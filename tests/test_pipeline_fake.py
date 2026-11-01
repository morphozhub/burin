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


BASE_FLAGS = dict(STRUCTURE_REFINE=False, METAL_TABLE=False, GLASS_RULES=False, S1_PICK=False, CHECKLIST=False,
                  DRAFT_HIGH=False, CHAINS=3, PARALLEL_CALLS=3, FINAL_REFINES=2, TOURNAMENT_PICK=False,
                  TOURNAMENT_MODEL=None, GUARDED_REFINES=False)


def _pin(monkeypatch, **flags):
    for k, v in {**BASE_FLAGS, **flags}.items():
        monkeypatch.setattr(config, k, v)


def _run(monkeypatch, **flags):
    _pin(monkeypatch, **flags)
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
    _pin(monkeypatch, S1_PICK=True)
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
        _pin(monkeypatch, DRAFT_HIGH=flag)
        w = EffortWitness()
        pipeline.run(_png(), 1, w, FakeRenderer())
        assert w.efforts[100] == ("high" if flag else "medium") and w.efforts[101] == "high"


class TournamentWitness(FakeWitness):
    """Duel replies that always favour the module shown second (model 2)."""

    def create(self, *, request_id, **kw):
        if pipeline.TOURNAMENT_IDS <= request_id < pipeline.FINAL_REFINE_ID:
            self.ids.append(request_id)
            return types.SimpleNamespace(output_text='{"penalty_1": 6, "penalty_2": 2}', usage={"input_tokens": 1, "output_tokens": 1})
        return super().create(request_id=request_id, **kw)


def test_tournament_with_three_chains_runs_24_duels_and_no_panel(monkeypatch):
    _pin(monkeypatch, TOURNAMENT_PICK=True, CHAINS=3)
    w = TournamentWitness()
    pipeline.run(_png(), 1, w, FakeRenderer())
    duels = sorted(i for i in w.ids if pipeline.TOURNAMENT_IDS <= i < pipeline.FINAL_REFINE_ID)
    assert duels == list(range(910, 934))
    assert not any(pipeline.FINAL_IDS <= i < pipeline.FINAL_IDS + 2 for i in w.ids)
    assert len(w.ids) == 9 + 24 + 2


def test_tournament_with_six_chains_narrows_by_panel_first(monkeypatch):
    seen = []
    _spy_first_final(monkeypatch, seen)
    _pin(monkeypatch, TOURNAMENT_PICK=True, CHAINS=6)
    w = TournamentWitness()
    pipeline.run(_png(), 1, w, FakeRenderer())
    assert sorted(i for i in w.ids if pipeline.FINAL_IDS <= i < pipeline.FINAL_IDS + 2) == [900, 901]
    assert len([i for i in w.ids if pipeline.TOURNAMENT_IDS <= i < pipeline.FINAL_REFINE_ID]) == 24
    assert len(w.ids) == 18 + 2 + 24 + 2
    assert seen and "// " in seen[0]


class ModelWitness(TournamentWitness):
    def create(self, *, request_id, model, **kw):
        self.models = getattr(self, "models", {})
        self.models[request_id] = model
        return super().create(request_id=request_id, model=model, **kw)


def test_tournament_model_switches_to_astra(monkeypatch):
    _pin(monkeypatch, TOURNAMENT_PICK=True, TOURNAMENT_MODEL="gpt-6-astra")
    w = ModelWitness()
    pipeline.run(_png(), 1, w, FakeRenderer())
    assert w.models[910] == "gpt-6-astra" and w.models[100] == pipeline.MODEL


class GuardWitness(FakeWitness):
    """Guard duels favour whichever side `winner` names (1 = before the refinement, 2 = after)."""

    def __init__(self, winner: int):
        super().__init__()
        self.winner = winner

    def create(self, *, request_id, **kw):
        if pipeline.GUARD_IDS <= request_id < pipeline.FINAL_IDS:
            self.ids.append(request_id)
            before_is_first = (request_id - pipeline.GUARD_IDS) % 8 < 4  # asks: 4 views of (before, after), then 4 of (after, before)
            before_wins = self.winner == 1
            p1, p2 = (2, 6) if before_is_first == before_wins else (6, 2)
            return types.SimpleNamespace(output_text=f'{{"penalty_1": {p1}, "penalty_2": {p2}}}', usage={"input_tokens": 1, "output_tokens": 1})
        return super().create(request_id=request_id, **kw)


def test_guarded_refines_keep_the_module_before_a_losing_refinement(monkeypatch):
    outs = {}
    for winner in (1, 2):
        _pin(monkeypatch, GUARDED_REFINES=True, FINAL_REFINES=1)
        w = GuardWitness(winner)
        outs[winner] = pipeline.run(_png(), 1, w, FakeRenderer())
        assert sorted(i for i in w.ids if pipeline.GUARD_IDS <= i < pipeline.FINAL_IDS) == list(range(860, 868))
    assert "// 950" not in outs[1] and "// 950" in outs[2]


def test_four_final_refines_use_ids_950_to_953(monkeypatch):
    _, w, _ = _run(monkeypatch, FINAL_REFINES=4)
    assert sorted(i for i in w.ids if pipeline.FINAL_REFINE_ID <= i < pipeline.PICK_IDS) == [950, 951, 952, 953]
    assert len(w.ids) == 9 + 2 + 4


def test_metal_table_text_follows_the_flag():
    assert ("Metal shades" in __import__("sn17gen.prompts", fromlist=["x"]).GEN_SYSTEM) == config.METAL_TABLE
