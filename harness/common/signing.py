"""Provenance signing for harness-produced artifacts.

Every metric/eval file written by the sealed harness carries an HMAC signature
computed over (trial_id, engine, timestamp, git_sha, payload). The verifier
recomputes the signature; trials that copy numbers from other runs fail R9.

The signing key lives in harness/common/.harness_key and is regenerated on
first use. Agents cannot read harness/** (blocked by PreToolUse hook), so they
cannot forge signatures.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any

KEY_PATH = Path(__file__).parent / ".harness_key"


def _load_or_create_key() -> bytes:
    if KEY_PATH.exists():
        return KEY_PATH.read_bytes()
    key = secrets.token_bytes(32)
    KEY_PATH.write_bytes(key)
    os.chmod(KEY_PATH, 0o600)
    return key


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def sign_payload(trial_id: str, engine: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return payload with harness_signature block attached.

    The signature binds (trial_id, engine, timestamp, git_sha, sha256(payload)).
    """
    key = _load_or_create_key()
    timestamp = time.time()
    git_sha = _git_sha()
    payload_bytes = json.dumps(payload, sort_keys=True).encode()
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    mac_input = f"{trial_id}|{engine}|{timestamp}|{git_sha}|{payload_hash}".encode()
    signature = hmac.new(key, mac_input, hashlib.sha256).hexdigest()
    return {
        **payload,
        "harness_signature": {
            "trial_id": trial_id,
            "engine": engine,
            "timestamp": timestamp,
            "git_sha": git_sha,
            "payload_sha256": payload_hash,
            "hmac_sha256": signature,
            "version": 1,
        },
    }


def verify_signature(signed: dict[str, Any], expected_trial_id: str) -> tuple[bool, str]:
    """Return (ok, reason). Fails if trial_id mismatches (cross-trial copy)."""
    sig = signed.get("harness_signature")
    if not sig:
        return False, "missing harness_signature"
    if sig.get("trial_id") != expected_trial_id:
        return False, f"trial_id mismatch: file={sig.get('trial_id')} dir={expected_trial_id}"
    key = _load_or_create_key()
    payload = {k: v for k, v in signed.items() if k != "harness_signature"}
    payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if payload_hash != sig.get("payload_sha256"):
        return False, "payload hash mismatch (file edited post-sign)"
    mac_input = (
        f"{sig['trial_id']}|{sig['engine']}|{sig['timestamp']}|{sig['git_sha']}|{payload_hash}"
    ).encode()
    expected = hmac.new(key, mac_input, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig.get("hmac_sha256", "")):
        return False, "hmac mismatch"
    return True, "ok"
