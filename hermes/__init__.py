"""HERMES Trading Engine v2.

Deterministic, research-grade intraday trading engine.

Phase 1 (Prompt 1): core engine - signals, orchestrator, market intelligence,
decision/risk, learning loop.

Phase 2 (Prompt 2): non-destructive safety hardening - data contract, cost
model, lifecycle, sizing safety, candidate thresholds, OOS gate, paper gate,
kill switch.

Integration layer: `safe_make_decision` wires the safety modules around the
core engine without modifying it.

Pure Python standard library only. No randomness. No hidden state.
"""

from hermes.integration import safe_make_decision

__all__ = ["safe_make_decision"]
