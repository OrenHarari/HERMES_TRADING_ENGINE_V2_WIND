"""Prompt 2 / Step 6E - Performance Claim Safety.

Pure observer that converts a pair of validation-window performance reports
plus an OOS validation result into one of five honest claims:

  - "improved"               : OOS passed AND a robustness metric strictly
                               improved (PF up, MDD down, or stability up)
                               with no robustness metric materially worsened
  - "preserved"              : OOS passed and no material change in
                               robustness
  - "worsened"               : OOS passed (within tolerance) but at least one
                               robustness metric materially worsened. The
                               system reports this honestly even though the
                               OOS gate technically allowed promotion.
  - "rejected"               : OOS validation_passed is False
  - "insufficient_evidence"  : no OOS result, or candidate trade count too
                               small to support any claim

Spec rules embedded:
  - Do not force improvement
  - Do not fabricate improvement (no claim without OOS evidence)
  - Do not optimize only for win_rate (win_rate alone is NOT a robustness
    metric here; it is reported as evidence but never drives a claim)
  - A lower win_rate with better profit_factor + lower drawdown may be
    acceptable -> still classified as "improved"
  - If evidence is insufficient, report it clearly
  - If performance worsens, report it clearly

This module is a *pure observer*: it never mutates threshold state. The
candidate-thresholds workflow continues to be controlled by Step 6B's
`promote_candidate`, which independently consumes the OOS validation_result.
"""

from hermes.utils.bounds import is_real_number

# ---- canonical claim strings (do not rename) ---------------------------
CLAIM_IMPROVED = "improved"
CLAIM_PRESERVED = "preserved"
CLAIM_WORSENED = "worsened"
CLAIM_REJECTED = "rejected"
CLAIM_INSUFFICIENT_EVIDENCE = "insufficient_evidence"

VALID_CLAIMS = (
    CLAIM_IMPROVED,
    CLAIM_PRESERVED,
    CLAIM_WORSENED,
    CLAIM_REJECTED,
    CLAIM_INSUFFICIENT_EVIDENCE,
)

REASON_OK = ""
REASON_NO_VALIDATION_EVIDENCE = "no_validation_evidence"
REASON_INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class PerformanceClaimConfig(object):
    """Tunable thresholds for performance-claim classification."""

    __slots__ = (
        "min_evidence_trades",
        "profit_factor_epsilon",
        "max_drawdown_epsilon_pct",
        "stability_epsilon",
    )

    def __init__(
        self,
        min_evidence_trades=30,
        profit_factor_epsilon=0.05,
        max_drawdown_epsilon_pct=0.05,
        stability_epsilon=0.05,
    ):
        if not isinstance(min_evidence_trades, int) or min_evidence_trades < 0:
            raise ValueError("min_evidence_trades must be int >= 0")
        for name, val in (
            ("profit_factor_epsilon", profit_factor_epsilon),
            ("max_drawdown_epsilon_pct", max_drawdown_epsilon_pct),
            ("stability_epsilon", stability_epsilon),
        ):
            if not is_real_number(val) or val < 0.0:
                raise ValueError("{!s} must be a non-negative number".format(name))
        self.min_evidence_trades = min_evidence_trades
        self.profit_factor_epsilon = float(profit_factor_epsilon)
        self.max_drawdown_epsilon_pct = float(max_drawdown_epsilon_pct)
        self.stability_epsilon = float(stability_epsilon)


def _claim(claim, reason, evidence):
    return {"claim": claim, "reason": reason, "evidence": dict(evidence)}


def evaluate_performance_claim(
    active_report,
    candidate_report,
    oos_validation_result,
    config=None,
):
    """Pure classification of a candidate-vs-active comparison.

    Returns:
      {"claim": str, "reason": str, "evidence": dict}
    """
    if config is None:
        config = PerformanceClaimConfig()
    if not isinstance(active_report, dict):
        raise ValueError("active_report must be a dict")
    if not isinstance(candidate_report, dict):
        raise ValueError("candidate_report must be a dict")
    if oos_validation_result is not None and not isinstance(
        oos_validation_result, dict
    ):
        raise ValueError("oos_validation_result must be a dict or None")

    active_pf = float(active_report.get("profit_factor", 0.0))
    cand_pf = float(candidate_report.get("profit_factor", 0.0))
    active_mdd = float(active_report.get("max_drawdown", 0.0))
    cand_mdd = float(candidate_report.get("max_drawdown", 0.0))
    active_stab = float(active_report.get("stability_score", 0.0))
    cand_stab = float(candidate_report.get("stability_score", 0.0))
    active_wr = float(active_report.get("win_rate", 0.0))
    cand_wr = float(candidate_report.get("win_rate", 0.0))
    cand_count = int(candidate_report.get("trade_count", 0))
    active_count = int(active_report.get("trade_count", 0))

    evidence = {
        "active_profit_factor": active_pf,
        "candidate_profit_factor": cand_pf,
        "active_max_drawdown": active_mdd,
        "candidate_max_drawdown": cand_mdd,
        "active_stability": active_stab,
        "candidate_stability": cand_stab,
        "active_win_rate": active_wr,
        "candidate_win_rate": cand_wr,
        "active_trade_count": active_count,
        "candidate_trade_count": cand_count,
    }

    # --- 1) No OOS evidence at all -> insufficient evidence -------------
    if oos_validation_result is None:
        return _claim(
            CLAIM_INSUFFICIENT_EVIDENCE,
            REASON_NO_VALIDATION_EVIDENCE,
            evidence,
        )

    # --- 2) OOS rejected -> claim rejected, preserve the original reason -
    if not oos_validation_result.get("validation_passed", False):
        return _claim(
            CLAIM_REJECTED,
            oos_validation_result.get("reason", "oos_validation_failed"),
            evidence,
        )

    # --- 3) Candidate sample too small -> insufficient evidence ----------
    if cand_count < config.min_evidence_trades:
        evidence["min_evidence_trades"] = config.min_evidence_trades
        return _claim(
            CLAIM_INSUFFICIENT_EVIDENCE,
            REASON_INSUFFICIENT_EVIDENCE,
            evidence,
        )

    # --- 4) Compute robustness deltas (NOT win_rate) --------------------
    pf_improved = (cand_pf - active_pf) > config.profit_factor_epsilon
    pf_worsened = (active_pf - cand_pf) > config.profit_factor_epsilon

    if active_mdd > 0.0:
        mdd_pct_change = (cand_mdd - active_mdd) / active_mdd
    elif cand_mdd > 0.0:
        mdd_pct_change = float("inf")
    else:
        mdd_pct_change = 0.0
    mdd_improved = mdd_pct_change < -config.max_drawdown_epsilon_pct
    mdd_worsened = mdd_pct_change > config.max_drawdown_epsilon_pct

    stab_improved = (cand_stab - active_stab) > config.stability_epsilon
    stab_worsened = (active_stab - cand_stab) > config.stability_epsilon

    evidence["pf_improved"] = pf_improved
    evidence["pf_worsened"] = pf_worsened
    evidence["mdd_improved"] = mdd_improved
    evidence["mdd_worsened"] = mdd_worsened
    evidence["stability_improved"] = stab_improved
    evidence["stability_worsened"] = stab_worsened

    any_worsened = pf_worsened or mdd_worsened or stab_worsened
    any_improved = pf_improved or mdd_improved or stab_improved

    # --- 5) Honest classification ----------------------------------------
    # If anything materially worsened (even though OOS allowed it within
    # tolerance), report "worsened" -- never claim improvement.
    if any_worsened:
        return _claim(CLAIM_WORSENED, "robustness_metric_worsened", evidence)
    if any_improved:
        return _claim(CLAIM_IMPROVED, "robustness_metric_improved", evidence)
    return _claim(CLAIM_PRESERVED, "no_material_change", evidence)


__all__ = [
    "CLAIM_IMPROVED",
    "CLAIM_INSUFFICIENT_EVIDENCE",
    "CLAIM_PRESERVED",
    "CLAIM_REJECTED",
    "CLAIM_WORSENED",
    "PerformanceClaimConfig",
    "REASON_INSUFFICIENT_EVIDENCE",
    "REASON_NO_VALIDATION_EVIDENCE",
    "REASON_OK",
    "VALID_CLAIMS",
    "evaluate_performance_claim",
]
