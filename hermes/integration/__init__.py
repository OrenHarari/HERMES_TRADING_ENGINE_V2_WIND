"""Integration layer.

Wires the Prompt-2 safety modules around the Prompt-1 core engine WITHOUT
modifying any existing module. The single public entrypoint is
`safe_make_decision` which preserves the canonical Phase-1 6-key output and
adds the explicitly-allowed Phase-2 additive keys.
"""

from hermes.integration.safe_decision import (
    INTEGRATION_OUTPUT_KEYS,
    PHASE2_ADDITIVE_KEYS,
    STAGE_CORE,
    STAGE_COST_MODEL,
    STAGE_DATA_CONTRACT,
    STAGE_KILL_SWITCH,
    STAGE_LIVE_GATE,
    STAGE_NONE,
    STAGE_SIZING_SAFETY,
    safe_make_decision,
)

__all__ = [
    "INTEGRATION_OUTPUT_KEYS",
    "PHASE2_ADDITIVE_KEYS",
    "STAGE_CORE",
    "STAGE_COST_MODEL",
    "STAGE_DATA_CONTRACT",
    "STAGE_KILL_SWITCH",
    "STAGE_LIVE_GATE",
    "STAGE_NONE",
    "STAGE_SIZING_SAFETY",
    "safe_make_decision",
]
