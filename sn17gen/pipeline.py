from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from . import prompts, views
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
FINAL_IDS = 900
FINAL_REFINE_ID = 950
FINAL_REFINES = 2
FINAL_REFINE_CREDITS = 600_000
TOTAL_CREDITS = 4_600_000  # the frozen budget is 5,000,000 credits and 60 calls per prompt
_CHARGE_LOCK = threading.Lock()
_TRACE_LOCK = threading.Lock()


def _trace(key: str, stage: str, source: str | None) -> None:
    """Development only: with SN17_TRACE_DIR set, append each stage's module for offline stage evaluation."""
    folder = os.environ.get("SN17_TRACE_DIR")
    if not folder or source is None:
        return
    path = os.path.join(folder, key)
    with _TRACE_LOCK:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "stages.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"stage": stage, "source": source}) + "\n")


def _png(image: bytes) -> bytes:
    im = Image.open(io.BytesIO(image)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _call(witness, ledger: Ledger, rid: int, kind: str, text: str, pngs: list[bytes], *, model: str | None = None,
          effort: str | None = None, instructions: str | None = None) -> str | None:
    model, effort = model or MODEL, effort or EFFORT
    instructions = prompts.GEN_SYSTEM if instructions is None else instructions
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


def _parallel(jobs: list[Callable[[], str | None]]) -> list[str | None]:
    """One stage of model calls: all start together, the stage ends when all have returned."""
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(job) for job in jobs]
        return [f.result() for f in futures]


class _Chain:
    """One chain (generate, then refine the latest valid module twice; repair rejections), one step per stage."""

    def __init__(self, idx: int, key: str) -> None:
        self.idx, self.key, self.base = idx, key, (idx + 1) * ID_BLOCK
        self.ledger = Ledger(credit_cap=CHAIN_CREDITS, call_cap=CHAIN_CALLS)
        self.stop, self.spare, self.retries, self.seen = CALLS, REPAIRS, PROVIDER_RETRIES, 0
        self.best: str | None = None
        self.best_sheet: bytes | None = None
        self.js: str | None = None
        self.err: str | None = None
        self.step = 0
        self.done = False

    def settle(self, renderer, kind: str) -> None:
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
            _trace(self.key, f"c{self.idx}.{self.step}.{kind}", wrapped)

    def next_job(self, witness, ref: bytes) -> tuple[Callable, str] | None:
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
            effort = EFFORT if self.ledger.calls == 0 else REFINE_EFFORT
        elif self.err is not None:
            kind, text, pngs, effort = "repair", prompts.REPAIR_USER % (self.err, self.js), [ref], REFINE_EFFORT
        else:
            kind, text, pngs, effort = "refine", prompts.REFINE_USER % self.best, [ref, self.best_sheet], REFINE_EFFORT
        rid = self.base + self.ledger.calls

        def job():
            return extract(_call(witness, self.ledger, rid, kind, text, pngs, effort=effort))
        return job, kind


def _penalties(text: str | None, n: int) -> list[float] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        pens = [float(x) for x in json.loads(m.group(0))["penalties"]] if m else []
    except (ValueError, KeyError, TypeError):
        pens = []
    return pens if len(pens) == n else None


def _panel(witness, renderer, ref: bytes, modules: list[str]) -> str:
    """lumen-kit's panel, asked in both presentation orders at once; penalties are summed."""
    sheets = [views.front_sheet(renderer.render, wrap(js)) for js in modules]
    n = len(modules)
    ledger = Ledger(credit_cap=2 * EXPECTED_PANEL_CREDITS + 1, call_cap=2)
    orders = (list(range(n)), list(range(n))[::-1])

    def job(k: int):
        def ask():
            order = orders[k]
            text = _call(witness, ledger, FINAL_IDS + k, "pick", prompts.PANEL_USER % (n + 1),
                         [ref, *[sheets[i] for i in order]], model=PANEL_MODEL, effort=PANEL_EFFORT,
                         instructions=prompts.PICK_SYSTEM)
            pens = _penalties(text, n)
            if pens is None:
                return None
            out = [0.0] * n
            for pos, i in enumerate(order):
                out[i] = pens[pos]
            return out
        return ask

    votes = [v for v in _parallel([job(0), job(1)]) if v is not None]
    if not votes:
        return modules[0]
    total = [sum(col) for col in zip(*votes)]
    return modules[min(range(n), key=lambda i: (total[i], i))]


def _final_refine(witness, renderer, ref: bytes, best: str, sheet: bytes, request_id: int) -> tuple[str, bytes]:
    """One more refinement of the picked module against the four camera views; kept only if it renders."""
    ledger = Ledger(credit_cap=FINAL_REFINE_CREDITS + 1, call_cap=1)
    js = extract(_call(witness, ledger, request_id, "final", prompts.REFINE_USER % best, [ref, sheet],
                       effort=REFINE_EFFORT))
    wrapped = wrap(js) if js is not None else None
    if wrapped is None:
        return best, sheet
    new_sheet, err = _sheet_or_error(renderer, wrapped)
    return (best, sheet) if err is not None else (js, new_sheet)


def run(image: bytes, seed: int, witness, renderer) -> str | None:
    ref = _png(image)
    key = hashlib.sha256(image).hexdigest()[:16]
    chains = [_Chain(i, key) for i in range(CHAINS)]
    while True:
        planned = [(c, c.next_job(witness, ref)) for c in chains if not c.done]
        planned = [(c, p) for c, p in planned if p is not None]
        if not planned:
            break
        results = _parallel([job for _, (job, _) in planned])
        for (c, (_, kind)), js in zip(planned, results):
            c.js = js
            c.step += 1
            c.settle(renderer, kind)
    found = [c.best for c in chains if c.best is not None]
    spent = sum(c.ledger.credits for c in chains)
    if len(found) > 1 and spent + 2 * EXPECTED_PANEL_CREDITS <= TOTAL_CREDITS:
        best = _panel(witness, renderer, ref, found)
    else:
        best = found[0] if found else None
    if best is not None:
        sheet = next(c.best_sheet for c in chains if c.best is best)
        for k in range(FINAL_REFINES):
            if spent + 2 * EXPECTED_PANEL_CREDITS + (k + 1) * FINAL_REFINE_CREDITS > TOTAL_CREDITS:
                break
            best, sheet = _final_refine(witness, renderer, ref, best, sheet, FINAL_REFINE_ID + k)
    out = final(best) if best is not None else None
    _trace(key, "final", out)
    if out is None:
        raise RuntimeError("no valid module")  # never submit an empty result; the runner retries the prompt
    return out
