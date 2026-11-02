from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from . import config, prompts, s1pick, views
from .budget import Ledger
from .jsutil import extract, final, wrap

MODEL = "gpt-6.1-sol"
EFFORT = "medium"
REFINE_EFFORT = "high"
SERVICE_TIER = "default"
CALLS = 3                 # per chain: generation and two refinements
REPAIRS = 1
CHAINS = 3
STARTS = (prompts.GEN_USER, prompts.GEN_USER_ALT, prompts.GEN_USER)
PANEL_MODEL = "gpt-6-astra"
PANEL_EFFORT = "low"
PROVIDER_RETRIES = 2
BACKOFF_SECONDS = 5
EXPECTED_CALL_CREDITS = 450_000
EXPECTED_PANEL_CREDITS = 150_000
CHAIN_CREDITS = 1_350_000
CHAIN_CALLS = CALLS + REPAIRS + PROVIDER_RETRIES
ID_BLOCK = 100
PLAN_ID = 50
PLAN_MODEL = "gpt-6.1-sol"
PLAN_EFFORT = "medium"
PLAN_CREDITS = 120_000
FINAL_IDS = 900
FINAL_REFINE_ID = 950
PICK_IDS = 970            # 8 ids: angle * 2 + order
PICK_MODEL = "gpt-6-astra"
PICK_EFFORT = "low"
PICK_PX = 1024
PICK_CALL_CREDITS = 80_000
PICK_CREDITS = 8 * PICK_CALL_CREDITS
FINAL_REFINES = 2
FINAL_REFINE_CREDITS = 250_000  # reserve per final refinement; measured 122k mean / 187k max on the 15-prompt stand
TOTAL_CREDITS = 4_600_000  # the frozen budget is 5,000,000 credits and 60 calls per prompt
_CHARGE_LOCK = threading.Lock()


def _png(image: bytes) -> bytes:
    im = Image.open(io.BytesIO(image)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _call(witness, ledger: Ledger, rid: int, kind: str, text: str, pngs: list[bytes], *, model: str | None = None,
          effort: str | None = None, instructions: str | None = None, content: list[dict] | None = None) -> str | None:
    model, effort = model or MODEL, effort or EFFORT
    instructions = prompts.GEN_SYSTEM if instructions is None else instructions
    if content is None:
        content = [{"type": "input_text", "text": text}]
        content += [{"type": "input_image", "image_url": views.data_url(p), "detail": "high"} for p in pngs]
    try:
        result = witness.responses.create(
            request_id=rid, model=model, instructions=instructions,
            reasoning={"effort": effort}, service_tier=SERVICE_TIER,
            input=[{"role": "user", "content": content}],
        )
    except Exception as exc:
        if getattr(exc, "source", None) != "provider" and not _provider_failure(exc):
            raise
        if getattr(exc, "status_code", None) == 429 or type(exc).__name__ in ("APIConnectionError", "APITimeoutError"):
            raise  # rate limit or connection loss: leave the prompt unsubmitted; a later pass reuses saved calls
        with _CHARGE_LOCK:
            ledger.charge(model, None, kind)
            ledger.provider_errors += 1
        time.sleep(BACKOFF_SECONDS)
        return None
    usage = getattr(result, "usage", None) or {}
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    with _CHARGE_LOCK:
        ledger.charge(model, usage, kind)
    return getattr(result, "output_text", None)


def _provider_failure(exc) -> bool:
    try:
        import openai
    except ImportError:
        return False
    return isinstance(exc, openai.APIError)


def _rejection(exc) -> str:
    return str(getattr(exc, "detail_json", None) or getattr(exc, "detail", None) or exc)[:1500]


def _sheet_or_error(renderer, source: str) -> tuple[bytes | None, str | None]:
    try:
        return views.sheet(renderer.render, source), None
    except Exception as exc:
        if type(exc).__name__ != "RenderRejected":
            raise
        return None, _rejection(exc)


def _parallel(jobs: list[Callable[[], object]]) -> list:
    """One stage of model calls: all start together, the stage ends when all have returned."""
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), config.PARALLEL_CALLS))) as pool:
        futures = [pool.submit(job) for job in jobs]
        return [f.result() for f in futures]


class _Chain:
    """One chain (generate, then refine the latest valid module twice; repair rejections), one step per stage."""

    def __init__(self, idx: int, notes: str | None) -> None:
        self.idx, self.base, self.notes = idx, (idx + 1) * ID_BLOCK, notes
        self.ledger = Ledger(credit_cap=CHAIN_CREDITS, call_cap=CHAIN_CALLS)
        self.stop, self.spare, self.retries, self.seen = CALLS, REPAIRS, PROVIDER_RETRIES, 0
        self.best: str | None = None
        self.best_sheet: bytes | None = None
        self.js: str | None = None
        self.err: str | None = None
        self.step = 0
        self.done = False

    def settle(self, renderer) -> None:
        """Validate the module the last call returned (renders run in the main thread, chain by chain)."""
        self.err = None
        if self.js is None:
            return
        wrapped = wrap(self.js)
        if wrapped is None:
            self.err = "the module must contain `export default function generate(THREE)`"
            return
        sheet, self.err = _sheet_or_error(renderer, wrapped)
        if self.err is None:
            self.best, self.best_sheet = self.js, sheet

    def next_job(self, witness, ref: bytes) -> Callable | None:
        while self.ledger.provider_errors > self.seen:
            self.seen += 1
            if self.retries > 0:
                self.retries -= 1
                self.stop += 1
        if self.ledger.calls >= self.stop and self.best is None and self.spare > 0:
            self.spare -= 1
            self.stop += 1
        if self.ledger.calls >= self.stop or not self.ledger.can_call(EXPECTED_CALL_CREDITS):
            self.done = True
            return None
        if self.js is None and self.best is None:
            kind, pngs = "gen", [ref]
            text = STARTS[self.idx % len(STARTS)] if self.ledger.calls == 0 else prompts.GEN_USER
            if self.notes:
                text = prompts.PLAN_NOTES % self.notes + text
            effort = (REFINE_EFFORT if config.DRAFT_HIGH else EFFORT) if self.ledger.calls == 0 else REFINE_EFFORT
        elif self.err is not None:
            kind, text, pngs, effort = "repair", prompts.REPAIR_USER % (self.err, self.js), [ref], REFINE_EFFORT
        else:
            kind, text, pngs, effort = "refine", prompts.REFINE_USER % self.best, [ref, self.best_sheet], REFINE_EFFORT
            if self.notes:
                text += prompts.CHECK_REFINE % self.notes
        rid = self.base + self.ledger.calls

        def job():
            return extract(_call(witness, self.ledger, rid, kind, text, pngs, effort=effort))
        return job


def _plan(witness, ref: bytes) -> tuple[str | None, float]:
    """One text call on the photo: notes and a judge checklist, reused by every later call."""
    ledger = Ledger(credit_cap=2 * PLAN_CREDITS, call_cap=1)
    text = _call(witness, ledger, PLAN_ID, "plan", prompts.PLAN_USER, [ref],
                 model=PLAN_MODEL, effort=PLAN_EFFORT, instructions=prompts.PLAN_SYSTEM)
    notes = (text or "").strip()[:2000]
    return (notes or None), ledger.credits


def _penalties(text: str | None, n: int) -> list[float] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        pens = [float(x) for x in json.loads(m.group(0))["penalties"]] if m else []
    except (ValueError, KeyError, TypeError):
        pens = []
    return pens if len(pens) == n else None


def _panel(witness, renderer, ref: bytes, modules: list[str], notes: str | None) -> tuple[list[int], float]:
    """lumen-kit's panel, asked in both presentation orders at once; penalties are summed.
    Returns the candidate indices from best to worst and the credits spent."""
    sheets = [views.front_sheet(renderer.render, wrap(js)) for js in modules]
    n = len(modules)
    ledger = Ledger(credit_cap=2 * EXPECTED_PANEL_CREDITS + 1, call_cap=2)
    orders = (list(range(n)), list(range(n))[::-1])
    text = prompts.PANEL_USER % (n + 1)
    if notes:
        text = prompts.CHECK_PANEL % notes + text

    def job(k: int):
        def ask():
            order = orders[k]
            reply = _call(witness, ledger, FINAL_IDS + k, "pick", text,
                          [ref, *[sheets[i] for i in order]], model=PANEL_MODEL, effort=PANEL_EFFORT,
                          instructions=prompts.PICK_SYSTEM)
            pens = _penalties(reply, n)
            if pens is None:
                return None
            out = [0.0] * n
            for pos, i in enumerate(order):
                out[i] = pens[pos]
            return out
        return ask

    votes = [v for v in _parallel([job(0), job(1)]) if v is not None]
    if not votes:
        return list(range(n)), ledger.credits
    total = [sum(col) for col in zip(*votes)]
    return sorted(range(n), key=lambda i: (total[i], i)), ledger.credits


def _s1_pick(witness, renderer, ref: bytes, a: str, b: str) -> tuple[str, float]:
    """Duel two modules as the judge's Stage 1 does: four single front views of the final (tilted) module,
    both presentation orders, the judge's own wording. Returns ("A" | "B" | "draw", credits)."""
    fa, fb = final(a), final(b)
    if fa is None or fb is None:
        return "draw", 0.0
    photo = views.data_url(ref)
    urls = {}
    for theta, phi, label in s1pick.ANGLES:
        for side, src in (("a", fa), ("b", fb)):
            urls[side, label] = views.data_url(renderer.render(src, views.view_options(theta, phi, PICK_PX)))
    ledger = Ledger(credit_cap=PICK_CREDITS + 1, call_cap=8)

    def job(k: int, label: str, order: str):
        first, second = (urls["a", label], urls["b", label]) if order == "ab" else (urls["b", label], urls["a", label])

        def ask():
            return s1pick.parse(_call(witness, ledger, PICK_IDS + k, "duel", "", [], model=PICK_MODEL, effort=PICK_EFFORT,
                                      instructions=s1pick.SYSTEM, content=s1pick.content(photo, first, second, label)))
        return ask

    jobs = []
    for i, (_, _, label) in enumerate(s1pick.ANGLES):
        jobs += [job(2 * i, label, "ab"), job(2 * i + 1, label, "ba")]
    replies = _parallel(jobs)
    angles = [s1pick.angle_penalties(replies[2 * i], replies[2 * i + 1]) for i in range(len(s1pick.ANGLES))]
    choice, _gap = s1pick.aggregate(angles)
    return choice, ledger.credits


TOURNAMENT_IDS = 910
TOURNAMENT_MAX = 3
TOURNAMENT_EFFORT = "low"
TOURNAMENT_CALL_CREDITS = 40_000


GUARD_IDS = 860
SCORE_IDS = 800           # 4 ids per finalist (800 + 4i): single-view judge scores that narrow the finalists
CRITIC_IDS = 840          # 4 ids per final refinement (840 + 4k): one critic call per judge view, below the guard block


def _tournament_call_credits() -> float:
    return 2 * TOURNAMENT_CALL_CREDITS if config.TOURNAMENT_MODEL == "gpt-6-astra" else TOURNAMENT_CALL_CREDITS


def _tournament_cost(n: int) -> float:
    return n * (n - 1) * len(s1pick.ANGLES) * _tournament_call_credits()


def _tournament(witness, renderer, ref: bytes, modules: list[str], ids: int = TOURNAMENT_IDS) -> tuple[int, float, dict]:
    """Round-robin of the finalists on the judge's four single front views, both orders, low effort;
    penalties are summed per module. Returns (index of the best module, credits, penalties by index)."""
    shots: dict[int, dict[str, str]] = {}
    for i, js in enumerate(modules):
        src = final(js)
        if src is None:
            continue
        try:
            shots[i] = {label: views.data_url(renderer.render(src, views.view_options(t, p, PICK_PX)))
                        for t, p, label in s1pick.ANGLES}
        except Exception as exc:
            if type(exc).__name__ != "RenderRejected":
                raise
    alive = sorted(shots)
    if len(alive) < 2:
        return (alive[0] if alive else 0), 0.0, {}
    photo = views.data_url(ref)
    asks = [(x, y, label) for a in alive for b in alive if a < b for x, y in ((a, b), (b, a)) for _, _, label in s1pick.ANGLES]
    ledger = Ledger(credit_cap=_tournament_cost(len(alive)) + 1, call_cap=len(asks))

    def job(k: int, x: int, y: int, label: str):
        def ask():
            return s1pick.parse(_call(witness, ledger, ids + k, "duel", "", [], model=config.TOURNAMENT_MODEL,
                                      effort=TOURNAMENT_EFFORT, instructions=s1pick.SYSTEM,
                                      content=s1pick.content(photo, shots[x][label], shots[y][label], label)))
        return ask

    replies = _parallel([job(k, x, y, label) for k, (x, y, label) in enumerate(asks)])
    pen = {i: 0.0 for i in alive}
    for (x, y, _), (p1, p2) in zip(asks, replies):
        pen[x] += p1
        pen[y] += p2
    return min(alive, key=lambda i: (pen[i], i)), ledger.credits, pen


def _score(witness, renderer, ref: bytes, modules: list[str], ids: int = SCORE_IDS) -> tuple[list[float], float]:
    """Single-view judge score of every module: one low-effort critic call per judge view (the tournament's model),
    penalties summed per module (unrenderable modules score worst). Returns (penalties by index, credits)."""
    photo = views.data_url(ref)
    shots: dict[int, dict[str, str]] = {}
    for i, js in enumerate(modules):
        src = final(js)
        if src is None:
            continue
        try:
            shots[i] = {label: views.data_url(renderer.render(src, views.view_options(t, p, PICK_PX)))
                        for t, p, label in s1pick.ANGLES}
        except Exception as exc:
            if type(exc).__name__ != "RenderRejected":
                raise
    asks = [(i, label) for i in sorted(shots) for _, _, label in s1pick.ANGLES]
    ledger = Ledger(credit_cap=len(asks) * _tournament_call_credits() + 1, call_cap=max(len(asks), 1))

    def job(k: int, i: int, label: str):
        def ask():
            return s1pick.parse_penalty(_call(witness, ledger, ids + k, "score", "", [], model=config.TOURNAMENT_MODEL,
                                             effort=TOURNAMENT_EFFORT, instructions=s1pick.SYSTEM,
                                             content=s1pick.critic_content(photo, shots[i][label], label)))
        return ask

    replies = _parallel([job(k, i, label) for k, (i, label) in enumerate(asks)])
    pen = [4 * 11.0] * len(modules)
    for i in shots:
        pen[i] = 0.0
    for (i, _), p in zip(asks, replies):
        pen[i] += p
    return pen, ledger.credits


def _aspect(png: bytes, photo: bool) -> float | None:
    """Width / height of the object's bounding box. Renders: anything darker than near-white. Photos: pixels that
    differ clearly from the border colour; None when the border is not a plain background."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    img.thumbnail((256, 256))
    w, h = img.size
    px = img.load()
    if photo:
        border = [px[x, y] for x in range(w) for y in (0, h - 1)] + [px[x, y] for y in range(h) for x in (0, w - 1)]
        mean = [sum(c[k] for c in border) / len(border) for k in range(3)]
        spread = sum(abs(c[k] - mean[k]) for c in border for k in range(3)) / len(border)
        if spread > 45:
            return None
        hit = lambda c: sum(abs(c[k] - mean[k]) for k in range(3)) > 90  # noqa: E731
    else:
        hit = lambda c: sum(c) < 3 * 245  # noqa: E731
    xs, ys = [], []
    for y in range(h):
        for x in range(w):
            if hit(px[x, y]):
                xs.append(x)
                ys.append(y)
    if len(xs) < 50:
        return None
    xs.sort(); ys.sort()
    lo, hi = len(xs) // 100, len(xs) - 1 - len(xs) // 100  # trim 1 percent of outliers each side
    bw, bh = xs[hi] - xs[lo] + 1, ys[hi] - ys[lo] + 1
    return bw / bh if bh > 0 else None


def _proportion_note(renderer, ref: bytes, js: str) -> str | None:
    """Photo versus near-frontal render (the judge's slightly-below camera) width:height; a note when they differ
    by more than 15 percent, None when either cannot be measured or they agree."""
    src = final(js)
    want = _aspect(ref, photo=True)
    if src is None or want is None:
        return None
    try:
        have = _aspect(renderer.render(src, views.view_options(0, 15, views.VIEW_PX)), photo=False)
    except Exception as exc:
        if type(exc).__name__ != "RenderRejected":
            raise
        return None
    if have is None or abs(have - want) <= 0.15 * max(want, have):
        return None
    return prompts.PROPORTION_REFINE.format(photo=want, model=have)


def _critique(witness, renderer, ref: bytes, js: str, ids: int) -> tuple[str | None, float]:
    """Judge-style critic on the four single front views of the final module: one low-effort call per view,
    the tournament's model; returns the issues as notes (one line per view) and the credits used."""
    src = final(js)
    if src is None:
        return None, 0.0
    try:
        shots = {label: views.data_url(renderer.render(src, views.view_options(t, p, PICK_PX))) for t, p, label in s1pick.ANGLES}
    except Exception as exc:
        if type(exc).__name__ != "RenderRejected":
            raise
        return None, 0.0
    photo = views.data_url(ref)
    ledger = Ledger(credit_cap=len(shots) * _tournament_call_credits() + 1, call_cap=len(shots))

    def job(k: int, label: str):
        def ask():
            return s1pick.parse_issues(_call(witness, ledger, ids + k, "critic", "", [], model=config.TOURNAMENT_MODEL,
                                            effort=TOURNAMENT_EFFORT, instructions=s1pick.SYSTEM,
                                            content=s1pick.critic_content(photo, shots[label], label)))
        return ask

    replies = _parallel([job(k, label) for k, (_, _, label) in enumerate(s1pick.ANGLES)])
    lines = [f"- seen from {s1pick.ANGLE_DESC[label]}: {issues}" for (_, _, label), issues in zip(s1pick.ANGLES, replies) if issues]
    return ("\n".join(lines) if lines else None), ledger.credits


def _final_refine(witness, renderer, ref: bytes, best: str, sheet: bytes, request_id: int, text: str,
                  notes: str | None) -> tuple[str, bytes]:
    """One more refinement of the picked module against the four camera views; kept only if it renders."""
    ledger = Ledger(credit_cap=FINAL_REFINE_CREDITS + 1, call_cap=1)
    text = text % best
    if notes:
        text += prompts.CHECK_REFINE % notes
    js = extract(_call(witness, ledger, request_id, "final", text, [ref, sheet], effort=REFINE_EFFORT))
    wrapped = wrap(js) if js is not None else None
    if wrapped is None:
        return best, sheet
    new_sheet, err = _sheet_or_error(renderer, wrapped)
    return (best, sheet) if err is not None else (js, new_sheet)


def run(image: bytes, seed: int, witness, renderer) -> str | None:
    ref = _png(image)
    notes, spent = None, 0.0
    if config.CHECKLIST:
        notes, spent = _plan(witness, ref)
    chains = [_Chain(i, notes) for i in range(min(config.CHAINS, (FINAL_IDS // ID_BLOCK) - 1))]
    while True:
        planned = [(c, c.next_job(witness, ref)) for c in chains if not c.done]
        planned = [(c, job) for c, job in planned if job is not None]
        if not planned:
            break
        results = _parallel([job for _, job in planned])
        for (c, _), js in zip(planned, results):
            c.js = js
            c.step += 1
            c.settle(renderer)
    found = [c for c in chains if c.best is not None]
    spent += sum(c.ledger.credits for c in chains)
    best_chain = found[0] if found else None
    if len(found) > 1 and spent + 2 * EXPECTED_PANEL_CREDITS <= TOTAL_CREDITS:
        finalists = found
        if config.TOURNAMENT_PICK and config.SCORE_NARROW and len(found) > config.SCORE_NARROW \
                and spent + 4 * len(found) * _tournament_call_credits() + _tournament_cost(config.SCORE_NARROW) + FINAL_REFINE_CREDITS <= TOTAL_CREDITS:
            pen, used = _score(witness, renderer, ref, [c.best for c in found])
            spent += used
            order = sorted(range(len(found)), key=lambda i: (pen[i], i))
            best_chain = found[order[0]]
            finalists = [found[i] for i in order[:config.SCORE_NARROW]]
        elif len(found) > TOURNAMENT_MAX or not config.TOURNAMENT_PICK:
            order, used = _panel(witness, renderer, ref, [c.best for c in found], notes)
            spent += max(used, 2 * EXPECTED_PANEL_CREDITS)
            best_chain = found[order[0]]
            finalists = [found[i] for i in order[:TOURNAMENT_MAX]]
        if config.TOURNAMENT_PICK and spent + _tournament_cost(len(finalists)) + FINAL_REFINE_CREDITS <= TOTAL_CREDITS:
            idx, used, _pen = _tournament(witness, renderer, ref, [c.best for c in finalists])
            spent += used
            best_chain = finalists[idx]
        elif config.S1_PICK and spent + PICK_CREDITS + FINAL_REFINE_CREDITS <= TOTAL_CREDITS:
            runner_up = found[order[1]]
            choice, used = _s1_pick(witness, renderer, ref, best_chain.best, runner_up.best)
            spent += used
            if choice == "B":
                best_chain = runner_up
    best = None
    if best_chain is not None:
        best, sheet = best_chain.best, best_chain.best_sheet
        n_refines = min(config.FINAL_REFINES, PICK_IDS - FINAL_REFINE_ID)
        for k in range(n_refines):
            if spent + (k + 1) * FINAL_REFINE_CREDITS > TOTAL_CREDITS:
                break
            text = prompts.FINAL_REFINE_USER if (config.STRUCTURE_REFINE and k == 0) else prompts.REFINE_USER
            if config.PROPORTION_NOTE:
                note = _proportion_note(renderer, ref, best)
                if note:
                    text += note  # numbers only, no percent signs: safe before the module substitution
            if config.JUDGE_NOTES and spent + 4 * _tournament_call_credits() + (k + 1) * FINAL_REFINE_CREDITS <= TOTAL_CREDITS:
                judge_notes, used = _critique(witness, renderer, ref, best, CRITIC_IDS + 4 * k)
                spent += used
                if judge_notes:
                    text += prompts.JUDGE_REFINE % judge_notes.replace("%", "%%")  # the module is substituted later
            before, before_sheet = best, sheet
            best, sheet = _final_refine(witness, renderer, ref, best, sheet, FINAL_REFINE_ID + k, text, notes)
            guard = config.GUARDED_REFINES or (config.GUARD_LAST and k == n_refines - 1)
            if guard and best is not before and spent + _tournament_cost(2) + FINAL_REFINE_CREDITS <= TOTAL_CREDITS:
                _idx, used, pen = _tournament(witness, renderer, ref, [before, best], ids=GUARD_IDS + 8 * k)
                spent += used
                if pen and pen.get(0, 0.0) < pen.get(1, 0.0):  # the refinement lost its duel: keep the module before it
                    best, sheet = before, before_sheet
    out = final(best) if best is not None else None
    if out is None:
        raise RuntimeError("no valid module")  # never submit an empty result; the runner retries the prompt
    return out
