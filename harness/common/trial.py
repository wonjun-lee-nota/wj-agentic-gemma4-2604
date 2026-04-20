"""Trial directory contract."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TRIALS_DIR = REPO_ROOT / "trials"

REQUIRED_INTENT_FIELDS = {
    "trial_id",
    "method",
    "bit_width",
    "group_size",
    "calibration_set",
    "quant_target",
    "notes",
}

# Files the harness writes (agent must not fabricate):
HARNESS_OUTPUTS = {"metrics.json", "eval.json", "verify.json"}
# Files the agent must provide before metrics can run:
AGENT_INPUTS = {"intent.yaml", "weights", "config.json"}


def trial_dir(trial_id: str) -> Path:
    return TRIALS_DIR / trial_id


def load_intent(trial_id: str) -> dict:
    path = trial_dir(trial_id) / "intent.yaml"
    with path.open() as f:
        data = yaml.safe_load(f)
    missing = REQUIRED_INTENT_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"intent.yaml missing fields: {sorted(missing)}")
    if data["trial_id"] != trial_id:
        raise ValueError(f"trial_id mismatch: intent={data['trial_id']} dir={trial_id}")
    return data
