"""Step 6 / Part 3 - Dynamic Threshold Adaptation - Baseline.

Behavior contract:

  - Never adapts before MIN_TRADES_FOR_ADAPTATION completed trades total.
  - Never adapts a bucket/regime cell below its min sample size (already
    enforced in attribution.compute_attribution).
  - min_confidence proposals always bounded by THRESHOLD_BOUNDS.
  - Every proposal/application is captured in the returned 'log'.

Phase 1 / Phase 2 bridge: the constructor flag `safety_addendum_active`
controls whether proposals are applied or kept as candidates only.

  safety_addendum_active=False (Phase 1 default):
    proposed updates are applied directly to the active thresholds dict and
    the result has thresholds_adapted=True when something changed.

  safety_addendum_active=True (Phase 2 will set this via wrapper):
    proposed updates are NOT applied to active thresholds; instead they are
    returned as 'candidate_thresholds' and the result has
    thresholds_adapted=False, candidate_thresholds_created=True.

Required output keys (always present, additive in Phase 2):
  - thresholds_adapted: bool
  - candidate_thresholds_created: bool
  - reason: str
  - active_thresholds_after: dict (current active thresholds; may equal input)
  - proposals: dict (proposed values, possibly empty)
  - log: list of dicts capturing the decision trace
"""

from hermes.learning.attribution import (
    BUCKET_BOUNDARIES,
    bucket_lower_bound_for_label,
)

THRESHOLD_BOUNDS = {
    "min_confidence": (0.40, 0.90),
}

MIN_TRADES_FOR_ADAPTATION = 100
DEFAULT_TARGET_WIN_RATE = 0.55
DEFAULT_CHOP_DISALLOW_THRESHOLD = 0.45
DEFAULT_CHOP_ALLOW_THRESHOLD = 0.55


def _clamp(name, value):
    lo, hi = THRESHOLD_BOUNDS[name]
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _result(adapted, candidate, reason, active_after, proposals, log):
    return {
        "thresholds_adapted": adapted,
        "candidate_thresholds_created": candidate,
        "reason": reason,
        "active_thresholds_after": active_after,
        "proposals": proposals,
        "log": log,
    }


class ThresholdAdapter(object):
    def __init__(
        self,
        safety_addendum_active=False,
        target_win_rate=DEFAULT_TARGET_WIN_RATE,
        min_trades_for_adaptation=MIN_TRADES_FOR_ADAPTATION,
        chop_disallow_threshold=DEFAULT_CHOP_DISALLOW_THRESHOLD,
        chop_allow_threshold=DEFAULT_CHOP_ALLOW_THRESHOLD,
    ):
        if not isinstance(safety_addendum_active, bool):
            raise ValueError("safety_addendum_active must be bool")
        if not (0.0 <= target_win_rate <= 1.0):
            raise ValueError("target_win_rate must be in [0,1]")
        if not isinstance(min_trades_for_adaptation, int) or min_trades_for_adaptation < 0:
            raise ValueError("min_trades_for_adaptation must be int >= 0")
        if not (0.0 <= chop_disallow_threshold <= 1.0):
            raise ValueError("chop_disallow_threshold must be in [0,1]")
        if not (0.0 <= chop_allow_threshold <= 1.0):
            raise ValueError("chop_allow_threshold must be in [0,1]")
        if chop_disallow_threshold > chop_allow_threshold:
            raise ValueError(
                "chop_disallow_threshold must be <= chop_allow_threshold"
            )
        self._addendum = safety_addendum_active
        self._target_win_rate = float(target_win_rate)
        self._min_trades = int(min_trades_for_adaptation)
        self._chop_disallow = float(chop_disallow_threshold)
        self._chop_allow = float(chop_allow_threshold)

    @property
    def safety_addendum_active(self):
        return self._addendum

    def propose(self, attribution_result, total_trades, active_thresholds):
        """Compute and (in non-addendum mode) apply threshold adaptations.

        `attribution_result`: output of compute_attribution.
        `total_trades`: count of all completed trades (gate for >=100).
        `active_thresholds`: dict containing at least 'min_confidence' and
                             'allow_chop'. Not mutated.

        Returns a result dict (see module docstring).
        """
        if not isinstance(active_thresholds, dict):
            raise ValueError("active_thresholds must be a dict")
        if "min_confidence" not in active_thresholds:
            raise ValueError("active_thresholds missing 'min_confidence'")
        if "allow_chop" not in active_thresholds:
            raise ValueError("active_thresholds missing 'allow_chop'")
        if not isinstance(total_trades, int) or total_trades < 0:
            raise ValueError("total_trades must be int >= 0")

        log = []
        proposals = {}
        active_after = dict(active_thresholds)

        if total_trades < self._min_trades:
            log.append(
                {
                    "decision": "no_adaptation",
                    "reason": "insufficient_total_trades",
                    "total_trades": total_trades,
                    "required": self._min_trades,
                }
            )
            return _result(
                False, False, "insufficient_total_trades",
                active_after, proposals, log,
            )

        # --- min_confidence proposal -----------------------------------
        # Find lowest bucket whose win_rate >= target_win_rate.
        eligible = []
        for entry in attribution_result.get("by_bucket", []):
            if entry["win_rate"] >= self._target_win_rate:
                eligible.append(entry)
        if eligible:
            # Map condition string back to lower bound and pick lowest.
            def _lower(entry):
                return bucket_lower_bound_for_label(entry["condition"])

            eligible.sort(key=_lower)
            chosen = eligible[0]
            proposed_min_conf = _clamp("min_confidence", _lower(chosen))
            if proposed_min_conf != active_after["min_confidence"]:
                proposals["min_confidence"] = proposed_min_conf
                log.append(
                    {
                        "decision": "propose_min_confidence",
                        "from": active_after["min_confidence"],
                        "to": proposed_min_conf,
                        "based_on_bucket": chosen["condition"],
                        "bucket_win_rate": chosen["win_rate"],
                        "bucket_trade_count": chosen["trade_count"],
                    }
                )
        else:
            log.append(
                {
                    "decision": "no_min_confidence_proposal",
                    "reason": "no_bucket_meets_target_win_rate",
                    "target_win_rate": self._target_win_rate,
                }
            )

        # --- allow_chop proposal ---------------------------------------
        chop_entry = None
        for entry in attribution_result.get("by_regime", []):
            if entry["condition"] == "regime=chop":
                chop_entry = entry
                break
        if chop_entry is not None:
            wr = chop_entry["win_rate"]
            if wr < self._chop_disallow and active_after["allow_chop"] is True:
                proposals["allow_chop"] = False
                log.append(
                    {
                        "decision": "propose_disallow_chop",
                        "chop_win_rate": wr,
                        "trade_count": chop_entry["trade_count"],
                    }
                )
            elif wr >= self._chop_allow and active_after["allow_chop"] is False:
                proposals["allow_chop"] = True
                log.append(
                    {
                        "decision": "propose_allow_chop",
                        "chop_win_rate": wr,
                        "trade_count": chop_entry["trade_count"],
                    }
                )

        if not proposals:
            log.append({"decision": "no_change_needed"})
            return _result(
                False, False, "no_change_needed",
                active_after, proposals, log,
            )

        if self._addendum:
            # Phase 2 mode: do NOT touch active thresholds.
            log.append({"decision": "candidate_only_addendum_active"})
            return _result(
                False, True, "candidate_only_addendum_active",
                active_after, proposals, log,
            )

        # Phase 1 mode: apply directly.
        for k, v in proposals.items():
            active_after[k] = v
        log.append({"decision": "applied_directly", "applied": dict(proposals)})
        return _result(
            True, False, "applied_directly",
            active_after, proposals, log,
        )
