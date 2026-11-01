import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sn17gen import s1pick  # noqa: E402


def test_parse_is_tolerant():
    assert s1pick.parse('{"penalty_1": 2, "penalty_2": 5, "issues": "x"}') == (2.0, 5.0)
    assert s1pick.parse('before {"penalty_1": "3", "penalty_2": 4.5} after') == (3.0, 4.5)
    assert s1pick.parse("no json") == s1pick.NEUTRAL
    assert s1pick.parse('{"penalty_1": 11, "penalty_2": 1}') == s1pick.NEUTRAL
    assert s1pick.parse(None) == s1pick.NEUTRAL


def test_angle_penalties_mirror_the_second_order():
    # AB: A=2, B=4; BA: B=5 (model 1), A=1 (model 2) -> A = (2+1)/2, B = (4+5)/2
    assert s1pick.angle_penalties((2, 4), (5, 1)) == (1.5, 4.5)


def test_aggregate_draw_band_and_sides():
    assert s1pick.aggregate([(3, 3.5)] * 4)[0] == "draw"          # gap 0.5 <= 0.7
    assert s1pick.aggregate([(2, 3)] * 4)[0] == "A"               # A lower penalty
    assert s1pick.aggregate([(4, 3), (4, 3), (4, 3), (4, 2)])[0] == "B"
    assert s1pick.aggregate([])[0] == "draw"
    # a neutral (failed) angle contributes zero to the gap
    choice, gap = s1pick.aggregate([(2, 4), (2, 4), (2, 4), s1pick.NEUTRAL])
    assert choice == "A" and abs(gap - 1.5) < 1e-9


def test_content_layout_matches_the_judge():
    c = s1pick.content("p", "m1", "m2", "front_left")
    assert [x["type"] for x in c] == ["input_text", "input_image", "input_text", "input_image", "input_text", "input_image", "input_text"]
    assert c[1]["image_url"] == "p" and c[3]["image_url"] == "m1" and c[5]["image_url"] == "m2"
    assert "slightly left of the front" in c[-1]["text"] and "penalty_1" in c[-1]["text"]
