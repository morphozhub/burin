"""Pairwise pick on the judge's Stage-1 views: same four cameras, same wording, same penalty scale.

Calibrated offline on 120 round-0 duels (experiments/selector/calibration_r0.md): the mean gap over
all four angles with both presentation orders averaged, decided above a 0.7 band, agreed with the
judge's side on 40 of 40 clear-gap picks and abstained on the rest.
"""
from __future__ import annotations

import json
import re

ANGLES = ((30, 0, "front_left"), (330, 0, "front_right"), (0, 15, "front_below"), (0, -30, "front_above"))
ANGLE_DESC = {"front_left": "slightly left of the front", "front_right": "slightly right of the front",
              "front_below": "the front, from slightly below", "front_above": "the front, from slightly above"}
SYSTEM = ("You are a specialized 3D model evaluation system.\n"
          "Analyze visual quality and prompt adherence with expert precision.\n"
          "Always respond with valid JSON only.")
USER = ("You see two 3D models rendered from {angle_desc}.\n"
        "The reference image shows the target object.\n\n"
        "Which model is a more faithful 3D reproduction of the reference?\n\n"
        "Penalty 0-10:\n"
        "0 = Perfect match to reference\n"
        "3 = Minor issues (slight shape differences, missing small details)\n"
        "5 = Moderate issues (wrong style, significant details missing)\n"
        "7 = Major issues (wrong category but related, e.g. chair vs stool)\n"
        "10 = Completely wrong object\n\n"
        'Output: {{"penalty_1": <0-10>, "penalty_2": <0-10>, "issues": "<brief>"}}')
CRITIC_USER = ("You see one 3D model rendered from {angle_desc}.\n"
               "The reference image shows the target object.\n\n"
               "How faithful a 3D reproduction of the reference is the model?\n\n"
               "Penalty 0-10:\n"
               "0 = Perfect match to reference\n"
               "3 = Minor issues (slight shape differences, missing small details)\n"
               "5 = Moderate issues (wrong style, significant details missing)\n"
               "7 = Major issues (wrong category but related, e.g. chair vs stool)\n"
               "10 = Completely wrong object\n\n"
               'Output: {{"penalty": <0-10>, "issues": "<the visible differences from the reference, most important first>"}}')
DRAW_BAND = 0.7
NEUTRAL = (5.0, 5.0)


def content(photo_url: str, first_url: str, second_url: str, label: str) -> list[dict]:
    return [{"type": "input_text", "text": "Reference image (target object):"},
            {"type": "input_image", "image_url": photo_url, "detail": "high"},
            {"type": "input_text", "text": "3D model 1:"},
            {"type": "input_image", "image_url": first_url, "detail": "high"},
            {"type": "input_text", "text": "3D model 2:"},
            {"type": "input_image", "image_url": second_url, "detail": "high"},
            {"type": "input_text", "text": USER.format(angle_desc=ANGLE_DESC[label])}]


def critic_content(photo_url: str, model_url: str, label: str) -> list[dict]:
    return [{"type": "input_text", "text": "Reference image (target object):"},
            {"type": "input_image", "image_url": photo_url, "detail": "high"},
            {"type": "input_text", "text": "3D model:"},
            {"type": "input_image", "image_url": model_url, "detail": "high"},
            {"type": "input_text", "text": CRITIC_USER.format(angle_desc=ANGLE_DESC[label])}]


def parse_issues(text: str | None) -> str | None:
    """The critic's issues text, or None when the reply is unreadable or names no issue."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        issues = json.loads(m.group(0)).get("issues")
    except (ValueError, AttributeError):
        return None
    issues = str(issues or "").strip()
    return issues[:600] or None


def parse(text: str | None) -> tuple[float, float]:
    """(penalty of model 1, penalty of model 2); unreadable or out-of-range replies count as neutral."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return NEUTRAL
    try:
        j = json.loads(m.group(0))
        p1, p2 = float(j["penalty_1"]), float(j["penalty_2"])
    except (ValueError, KeyError, TypeError):
        return NEUTRAL
    if not (0 <= p1 <= 10 and 0 <= p2 <= 10):
        return NEUTRAL
    return p1, p2


def angle_penalties(ab: tuple[float, float], ba: tuple[float, float]) -> tuple[float, float]:
    """Per-angle (pen_a, pen_b): A was model 1 in the AB call and model 2 in the BA call."""
    return (ab[0] + ba[1]) / 2, (ab[1] + ba[0]) / 2


def aggregate(angles: list[tuple[float, float]]) -> tuple[str, float]:
    """Mean gap (pen_b - pen_a) over all angles: positive favours A. Within the band -> draw."""
    if not angles:
        return "draw", 0.0
    gap = sum(b - a for a, b in angles) / len(angles)
    if abs(gap) <= DRAW_BAND:
        return "draw", gap
    return ("A" if gap > 0 else "B"), gap


def parse_penalty(text: str | None) -> float:
    """Single-model critic penalty 0-10; unreadable or out-of-range replies count as a neutral 5."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        p = float(json.loads(m.group(0))["penalty"]) if m else 5.0
    except (ValueError, KeyError, TypeError):
        return 5.0
    return p if 0.0 <= p <= 10.0 else 5.0
