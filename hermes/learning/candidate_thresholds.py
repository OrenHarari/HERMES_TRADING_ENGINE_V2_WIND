"""Prompt 2 / Step 6B - Candidate Thresholds.

Stand-alone storage and promotion gate for candidate threshold proposals.

Files (per spec, all under `base_dir`):
  - active_thresholds.json      # used for live trading decisions
  - candidate_thresholds.json   # proposals only; never affect trading
  - threshold_adaptation_log.json  # append-only log of every event

The actual OOS validation logic is Step 6D's responsibility; here we accept
a `validation_result` dict (with `validation_passed` and `reason`) from any
upstream validator. Step 6B enforces:

  - candidate thresholds remain bounded (THRESHOLD_BOUNDS)
  - direct mutation of active is blocked unless promotion succeeds
  - edge_decay_alert blocks promotion
  - kill_switch_active blocks promotion
  - every event is logged

Existing Step 6 (`ThresholdAdapter`) is unchanged; its
`safety_addendum_active=True` mode already produces candidate-only proposals
that this module persists.
"""

import os
from datetime import datetime, timezone

from hermes.learning.threshold_adapter import THRESHOLD_BOUNDS
from hermes.utils.json_io import read_json, write_json_atomic

ACTIVE_FILENAME = "active_thresholds.json"
CANDIDATE_FILENAME = "candidate_thresholds.json"
LOG_FILENAME = "threshold_adaptation_log.json"

DEFAULT_THRESHOLD_SCHEMA_VERSION = 1

# Spec-required defaults for the active threshold structure (backward-
# compatible with the Phase-1 adapter's expected keys).
DEFAULT_ACTIVE_THRESHOLDS = {
    "min_confidence": 0.60,
    "allow_chop": False,
}

# Reasons exposed by Step 6B (canonical strings).
REASON_OK = ""
REASON_NO_CANDIDATE = "no_candidate_thresholds"
REASON_OUT_OF_BOUNDS = "candidate_out_of_bounds"
REASON_EDGE_DECAY_ACTIVE = "edge_decay_alert_active"
REASON_KILL_SWITCH_ACTIVE = "kill_switch_active"
REASON_CANDIDATE_REJECTED_OOS = "candidate_rejected_oos_validation"
REASON_PROMOTED = "candidate_promoted"


def _utc_iso_now():
    """Deterministic-by-construction wall-clock string. Tests provide their
    own log-entry payloads where strict determinism matters; this helper is
    only used for the implicit timestamp in adaptation log records.
    """
    return datetime.now(timezone.utc).isoformat()


def _within_bounds(thresholds):
    """Return (ok, offending_key, value, bounds) tuple."""
    for key, (lo, hi) in THRESHOLD_BOUNDS.items():
        if key in thresholds:
            v = thresholds[key]
            if v is None or not isinstance(v, (int, float)) or isinstance(v, bool):
                return False, key, v, (lo, hi)
            if v < lo or v > hi:
                return False, key, v, (lo, hi)
    return True, None, None, None


class ThresholdStore(object):
    """File-backed store for active / candidate thresholds + adaptation log."""

    __slots__ = ("_base_dir", "active_path", "candidate_path", "log_path")

    def __init__(self, base_dir):
        if not isinstance(base_dir, str) or not base_dir:
            raise ValueError("base_dir must be a non-empty string")
        self._base_dir = base_dir
        self.active_path = os.path.join(base_dir, ACTIVE_FILENAME)
        self.candidate_path = os.path.join(base_dir, CANDIDATE_FILENAME)
        self.log_path = os.path.join(base_dir, LOG_FILENAME)

    @property
    def base_dir(self):
        return self._base_dir

    # ---- active thresholds ----------------------------------------------
    def load_active(self):
        blob = read_json(self.active_path, default=None)
        if blob is None:
            return dict(DEFAULT_ACTIVE_THRESHOLDS)
        # Strip schema_version when handing the dict to callers; preserve
        # the raw payload on disk.
        out = {k: v for k, v in blob.items() if k != "schema_version"}
        # Ensure required defaults are present even if the file omits them.
        for k, v in DEFAULT_ACTIVE_THRESHOLDS.items():
            if k not in out:
                out[k] = v
        return out

    def save_active(self, thresholds, log_entry=None):
        if not isinstance(thresholds, dict):
            raise ValueError("thresholds must be a dict")
        ok, key, val, bounds = _within_bounds(thresholds)
        if not ok:
            raise ValueError(
                "active threshold {!r}={!r} out of bounds {!s}".format(
                    key, val, bounds
                )
            )
        blob = dict(thresholds)
        blob["schema_version"] = DEFAULT_THRESHOLD_SCHEMA_VERSION
        write_json_atomic(self.active_path, blob)
        if log_entry is not None:
            self.append_log(log_entry)

    # ---- candidate thresholds -------------------------------------------
    def load_candidate(self):
        blob = read_json(self.candidate_path, default=None)
        if blob is None:
            return None
        return {k: v for k, v in blob.items() if k != "schema_version"}

    def save_candidate(self, thresholds, log_entry=None):
        if not isinstance(thresholds, dict):
            raise ValueError("thresholds must be a dict")
        ok, key, val, bounds = _within_bounds(thresholds)
        if not ok:
            raise ValueError(
                "candidate threshold {!r}={!r} out of bounds {!s}".format(
                    key, val, bounds
                )
            )
        blob = dict(thresholds)
        blob["schema_version"] = DEFAULT_THRESHOLD_SCHEMA_VERSION
        write_json_atomic(self.candidate_path, blob)
        if log_entry is not None:
            self.append_log(log_entry)

    def clear_candidate(self, log_entry=None):
        if os.path.exists(self.candidate_path):
            os.remove(self.candidate_path)
        if log_entry is not None:
            self.append_log(log_entry)

    # ---- adaptation log -------------------------------------------------
    def append_log(self, entry):
        if not isinstance(entry, dict):
            raise ValueError("log entry must be a dict")
        log = read_json(self.log_path, default=[])
        if not isinstance(log, list):
            raise ValueError("adaptation log file is corrupt")
        record = dict(entry)
        record.setdefault("timestamp", _utc_iso_now())
        log.append(record)
        write_json_atomic(self.log_path, log)

    def get_log(self):
        log = read_json(self.log_path, default=[])
        if not isinstance(log, list):
            raise ValueError("adaptation log file is corrupt")
        return list(log)


def propose_candidate(adaptation_result, store):
    """Persist any proposals from a ThresholdAdapter result as a candidate.

    Returns:
      {
        "candidate_thresholds_created": bool,
        "thresholds_adapted": False,        # never mutates active here
        "active_thresholds_after": dict,    # unchanged copy
        "candidate_thresholds": dict | {},  # what was staged
        "reason": str,
      }
    """
    if not isinstance(adaptation_result, dict):
        raise ValueError("adaptation_result must be a dict")
    if not isinstance(store, ThresholdStore):
        raise ValueError("store must be a ThresholdStore instance")

    proposals = adaptation_result.get("proposals", {}) or {}
    active = store.load_active()

    if not proposals:
        return {
            "candidate_thresholds_created": False,
            "thresholds_adapted": False,
            "active_thresholds_after": active,
            "candidate_thresholds": {},
            "reason": "no_proposals",
        }

    # Build candidate dict by overlaying proposals on the current active.
    candidate = dict(active)
    for k, v in proposals.items():
        candidate[k] = v

    ok, key, val, bounds = _within_bounds(candidate)
    if not ok:
        store.append_log({
            "event": "candidate_rejected_out_of_bounds",
            "offending_key": key,
            "offending_value": val,
            "bounds": list(bounds) if bounds is not None else None,
            "proposals": dict(proposals),
        })
        return {
            "candidate_thresholds_created": False,
            "thresholds_adapted": False,
            "active_thresholds_after": active,
            "candidate_thresholds": {},
            "reason": REASON_OUT_OF_BOUNDS,
        }

    store.save_candidate(
        candidate,
        log_entry={
            "event": "candidate_proposed",
            "candidate_thresholds": dict(candidate),
            "proposals": dict(proposals),
            "source_reason": adaptation_result.get("reason", ""),
        },
    )
    return {
        "candidate_thresholds_created": True,
        "thresholds_adapted": False,
        "active_thresholds_after": active,
        "candidate_thresholds": candidate,
        "reason": "candidate_persisted",
    }


def promote_candidate(
    store,
    validation_result,
    edge_decay_alert=False,
    kill_switch_active=False,
):
    """Promote the staged candidate to active, gated by safety guards.

    `validation_result` must be a dict with at least
      {"validation_passed": bool, "reason": str}
    (this is provided by the OOS gate built in Step 6D).

    Hard guards (block promotion regardless of validation):
      - edge_decay_alert
      - kill_switch_active
      - validation_passed == False
      - no candidate present

    Returns:
      {
        "thresholds_adapted": bool,
        "candidate_thresholds_promoted": bool,
        "active_thresholds_after": dict,
        "reason": str,
        "candidate_rejection_reason": str | "",
      }
    """
    if not isinstance(store, ThresholdStore):
        raise ValueError("store must be a ThresholdStore instance")
    if not isinstance(validation_result, dict):
        raise ValueError("validation_result must be a dict")
    if "validation_passed" not in validation_result:
        raise ValueError(
            "validation_result missing required key 'validation_passed'"
        )
    if not isinstance(edge_decay_alert, bool):
        raise ValueError("edge_decay_alert must be bool")
    if not isinstance(kill_switch_active, bool):
        raise ValueError("kill_switch_active must be bool")

    candidate = store.load_candidate()
    active = store.load_active()

    if candidate is None:
        return {
            "thresholds_adapted": False,
            "candidate_thresholds_promoted": False,
            "active_thresholds_after": active,
            "reason": REASON_NO_CANDIDATE,
            "candidate_rejection_reason": "",
        }

    if kill_switch_active:
        store.append_log({
            "event": "candidate_rejected",
            "reason": REASON_KILL_SWITCH_ACTIVE,
            "candidate_thresholds": dict(candidate),
        })
        return {
            "thresholds_adapted": False,
            "candidate_thresholds_promoted": False,
            "active_thresholds_after": active,
            "reason": REASON_KILL_SWITCH_ACTIVE,
            "candidate_rejection_reason": REASON_KILL_SWITCH_ACTIVE,
        }

    if edge_decay_alert:
        store.append_log({
            "event": "candidate_rejected",
            "reason": REASON_EDGE_DECAY_ACTIVE,
            "candidate_thresholds": dict(candidate),
        })
        return {
            "thresholds_adapted": False,
            "candidate_thresholds_promoted": False,
            "active_thresholds_after": active,
            "reason": REASON_EDGE_DECAY_ACTIVE,
            "candidate_rejection_reason": REASON_EDGE_DECAY_ACTIVE,
        }

    if not validation_result.get("validation_passed", False):
        store.append_log({
            "event": "candidate_rejected",
            "reason": REASON_CANDIDATE_REJECTED_OOS,
            "candidate_thresholds": dict(candidate),
            "validation_reason": validation_result.get("reason", ""),
        })
        return {
            "thresholds_adapted": False,
            "candidate_thresholds_promoted": False,
            "active_thresholds_after": active,
            "reason": REASON_CANDIDATE_REJECTED_OOS,
            "candidate_rejection_reason": REASON_CANDIDATE_REJECTED_OOS,
        }

    # All guards passed -> promote.
    store.save_active(
        candidate,
        log_entry={
            "event": "candidate_promoted",
            "previous_active": dict(active),
            "new_active": dict(candidate),
            "validation_reason": validation_result.get("reason", ""),
        },
    )
    store.clear_candidate()
    return {
        "thresholds_adapted": True,
        "candidate_thresholds_promoted": True,
        "active_thresholds_after": store.load_active(),
        "reason": REASON_PROMOTED,
        "candidate_rejection_reason": "",
    }


__all__ = [
    "ACTIVE_FILENAME",
    "CANDIDATE_FILENAME",
    "DEFAULT_ACTIVE_THRESHOLDS",
    "DEFAULT_THRESHOLD_SCHEMA_VERSION",
    "LOG_FILENAME",
    "REASON_CANDIDATE_REJECTED_OOS",
    "REASON_EDGE_DECAY_ACTIVE",
    "REASON_KILL_SWITCH_ACTIVE",
    "REASON_NO_CANDIDATE",
    "REASON_OK",
    "REASON_OUT_OF_BOUNDS",
    "REASON_PROMOTED",
    "ThresholdStore",
    "promote_candidate",
    "propose_candidate",
]
