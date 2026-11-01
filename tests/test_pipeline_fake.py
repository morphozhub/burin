"""The pipeline against a fake witness and renderer: stage order, request ids, call counts, flag paths."""
import io
import json
import sys
import types
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sn17gen import config, pipeline  # noqa: E402

JS = "export default function generate(THREE) {\n  const root = new THREE.Group();\n  return root;\n}"


def _png(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 200, 200)).save(buf, format="PNG")
    return buf.getvalue()


class FakeRenderer:
    def __init__(self):
        self.calls = 0

    def render(self, source, options):
        self.calls += 1
        px = getattr(options, "img_size", None) or (options.get("img_size") if isinstance(options, dict) else 512)
        n = len(getattr(options, "thetas", None) or (options.get("thetas") if isinstance(options, dict) else [0]))
        side = 2 * px if n > 1 else px
        return _png((side, side))


class FakeWitness:
    """Replies by request id: code calls get a module, picks get penalties, duels get a verdict."""

    def __init__(self, duel_reply='{"penalty_1": 2, "penalty_2": 6, "issues": "x"}', panel='{"penalties": [3, 1, 5]}'):
        self.ids = []
        self.duel_reply, self.panel = duel_reply, panel
        self.responses = self

    def create(self, *, request_id, model, instructions, reasoning, service_tier, input):
        self.ids.append(request_id)
        if pipeline.FINAL_IDS <= request_id < pipeline.FINAL_IDS + 2:
            text = self.panel
        elif pipeline.PICK_IDS <= request_id < pipeline.PICK_IDS + 8:
            text = self.duel_reply
        elif request_id == pipeline.PLAN_ID:
            text = "Object: cup.\nChecklist:\n- one handle"
        else:
            text = f"```js\n{JS} // {request_id}\n```"
        return types.SimpleNamespace(output_text=text, usage={"input_tokens": 1000, "output_tokens": 500})


def _run(monkeypatch, **flags):
    for k, v in flags.items():
        monkeypatch.setattr(config, k, v)
    w, r = FakeWitness(), FakeRenderer()
    out = pipeline.run(_png(), 1, w, r)
    return out, w, r


def test_base_stage_ids_and_counts(monkeypatch):
    out, w, _ = _run(monkeypatch, STRUCTURE_REFINE=False, METAL_TABLE=False, GLASS_RULES=False, S1_PICK=False, CHECKLIST=False)
    assert out and "__inner" in out
    chain = sorted(i for i in w.ids if i < pipeline.FINAL_IDS)
    assert chain == [100, 101, 102, 200, 201, 202, 300, 301, 302]
    assert sorted(i for i in w.ids if i >= pipeline.FINAL_IDS) == [900, 901, 950, 951]
    assert len(w.ids) == 13


class SidedWitness(FakeWitness):
    """Duel replies that favour the runner-up (B) in both presentation orders."""

    def create(self, *, request_id, **kw):
        if pipeline.PICK_IDS <= request_id < pipeline.PICK_IDS + 8:
            self.ids.append(request_id)
            text = '{"penalty_1": 6, "penalty_2": 1}' if request_id % 2 == 0 else '{"penalty_1": 1, "penalty_2": 6}'
            return types.SimpleNamespace(output_text=text, usage={"input_tokens": 1, "output_tokens": 1})
        return super().create(request_id=request_id, **kw)


def _spy_first_final(monkeypatch, seen):
    real = pipeline._final_refine

    def spy(witness, renderer, ref, best, sheet, request_id, text, notes):
        seen.append(best)
        return real(witness, renderer, ref, best, sheet, request_id, text, notes)
    monkeypatch.setattr(pipeline, "_final_refine", spy)


def test_s1_pick_runs_eight_duel_calls_and_draw_keeps_the_panel_pick(monkeypatch):
    # panel penalties [3, 1, 5] -> chain 2 (ids 200..202) first, chain 1 (ids 100..102) runner-up;
    # the same reply in both orders mirrors to a zero gap -> draw -> the panel pick stands
    seen = []
    _spy_first_final(monkeypatch, seen)
    _, w, _ = _run(monkeypatch, S1_PICK=True, CHECKLIST=False, STRUCTURE_REFINE=False)
    assert sorted(i for i in w.ids if pipeline.PICK_IDS <= i < pipeline.PICK_IDS + 8) == list(range(970, 978))
    assert len(w.ids) == 21
    assert "// 202" in seen[0]


def test_s1_pick_one_sided_duel_flips_to_the_runner_up(monkeypatch):
    seen = []
    _spy_first_final(monkeypatch, seen)
    for k, v in dict(S1_PICK=True, CHECKLIST=False, STRUCTURE_REFINE=False).items():
        monkeypatch.setattr(config, k, v)
    pipeline.run(_png(), 1, SidedWitness(), FakeRenderer())
    assert "// 102" in seen[0]


def test_checklist_adds_one_plan_call_first(monkeypatch):
    _, w, _ = _run(monkeypatch, CHECKLIST=True, S1_PICK=False)
    assert w.ids[0] == pipeline.PLAN_ID and len(w.ids) == 14


def test_structure_refine_uses_the_structure_text_once(monkeypatch):
    seen = []
    real = pipeline._final_refine

    def spy(witness, renderer, ref, best, sheet, request_id, text, notes):
        seen.append("STRUCTURE" if "exact count of every" in text else "plain")
        return real(witness, renderer, ref, best, sheet, request_id, text, notes)
    monkeypatch.setattr(pipeline, "_final_refine", spy)
    _run(monkeypatch, STRUCTURE_REFINE=True, S1_PICK=False, CHECKLIST=False)
    assert seen == ["STRUCTURE", "plain"]


def test_six_chains_use_six_id_blocks_below_the_panel(monkeypatch):
    _, w, _ = _run(monkeypatch, CHAINS=6, S1_PICK=False, CHECKLIST=False, PARALLEL_CALLS=3)
    chain = sorted(i for i in w.ids if i < pipeline.FINAL_IDS)
    assert chain == [b + k for b in (100, 200, 300, 400, 500, 600) for k in (0, 1, 2)]
    assert len(w.ids) == 18 + 4


class EffortWitness(FakeWitness):
    def create(self, *, request_id, reasoning, **kw):
        self.efforts = getattr(self, "efforts", {})
        self.efforts[request_id] = reasoning["effort"] if isinstance(reasoning, dict) else getattr(reasoning, "effort", reasoning)
        return super().create(request_id=request_id, reasoning=reasoning, **kw)


def test_draft_high_raises_only_the_first_call_of_each_chain(monkeypatch):
    for flag in (False, True):
        monkeypatch.setattr(config, "DRAFT_HIGH", flag)
        monkeypatch.setattr(config, "CHAINS", 3)
        monkeypatch.setattr(config, "S1_PICK", False)
        monkeypatch.setattr(config, "CHECKLIST", False)
        w = EffortWitness()
        pipeline.run(_png(), 1, w, FakeRenderer())
        assert w.efforts[100] == ("high" if flag else "medium") and w.efforts[101] == "high"


def test_flags_off_keep_the_base_system_prompt():
    assert "Metal shades" not in __import__("sn17gen.prompts", fromlist=["x"]).GEN_SYSTEM or config.METAL_TABLE
