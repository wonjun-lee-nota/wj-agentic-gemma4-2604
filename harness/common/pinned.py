"""Load pinned measurement/eval parameters.

R5: measurement parameters (max_length, stride, dtype, attn_implementation,
seed, sample_n, max_tokens, temperature, extractor) are pinned and cannot be
overridden at runtime. The executor agent writes an intent.yaml but that file
describes quantization choices only; measurement knobs come from here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PINNED_PATH = Path(__file__).resolve().parents[2] / "configs" / "pinned.yaml"


def load() -> dict[str, Any]:
    with PINNED_PATH.open() as f:
        return yaml.safe_load(f)
