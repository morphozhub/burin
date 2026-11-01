"""Adapter so the shared batch runner and dry stand can drive the sn17gen generator.

``run(image, seed, calls, renderer)`` gives sn17gen a witness-like object whose ``responses``
attribute is the runner's call wrapper (same ``create(request_id=..., **params)`` interface
as ``witness.responses``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, Unpack

from openai.types.responses import Response
from openai.types.responses.response_create_params import ResponseCreateParamsBase
from renderer_client import Renderer

from sn17gen import pipeline as _generator
from sn17gen import prompts as _prompts

_ROOT = Path(__file__).resolve().parents[1]
# Identity of the generator for batch run.json: every sn17gen source file.
INSTRUCTIONS = _prompts.GEN_SYSTEM + "\n" + hashlib.sha256(
    b"".join(p.read_bytes() for p in sorted((_ROOT / "sn17gen").glob("*.py")))
).hexdigest()


class ModelCalls(Protocol):
    def create(self, *, request_id: int = 0, **params: Unpack[ResponseCreateParamsBase]) -> Response: ...


def run(image: bytes, seed: int, calls: ModelCalls, renderer: Renderer) -> str | None:
    witness = SimpleNamespace(responses=calls)
    return _generator.run(image, seed, witness, renderer)
