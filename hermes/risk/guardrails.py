"""Step 5 / Part 3 - Risk Guardrails.

Stateful guardrail tracker. Deterministic. The state is explicit and lives on
the RiskState instance; no module-level globals.

Block reasons (deterministic priority order in `check`):
  1. max_trades_per_day_reached
  2. max_daily_loss_reached
  3. max_consecutive_losses_reached
  4. cooldown_active
"""

from hermes.risk.config import RiskConfig
from hermes.utils.bounds import is_numeric

REASON_OK = ""
REASON_MAX_TRADES = "max_trades_per_day_reached"
REASON_DAILY_LOSS = "max_daily_loss_reached"
REASON_MAX_CONSECUTIVE_LOSSES = "max_consecutive_losses_reached"
REASON_COOLDOWN = "cooldown_active"


def _coerce_numeric(name, value):
    if not is_numeric(value):
        raise ValueError("{!s} must be numeric (non-bool, non-NaN); got {!r}".format(name, value))
    return float(value)


class RiskState(object):
    """Mutable, explicit risk-tracking state.

    Conventions:
      - now_ts is an integer/float "seconds since epoch" or any monotonic
        scalar that supports subtraction. The class never reads system time.
      - day_key is whatever caller-defined hashable identifies a session day
        (e.g. an ISO date string). When it changes, the per-day counters reset.
    """

    __slots__ = (
        "_config",
        "trades_today",
        "daily_pnl",
        "consecutive_losses",
        "current_day",
        "last_entry_ts",
    )

    def __init__(self, config=None):
        if config is None:
            config = RiskConfig()
        self._config = config
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.current_day = None
        self.last_entry_ts = None

    @property
    def config(self):
        return self._config

    def start_new_day(self, day_key):
        """Reset per-day counters and set the current day key."""
        self.current_day = day_key
        self.trades_today = 0
        self.daily_pnl = 0.0
        # consecutive_losses persists across days (it's a streak counter).

    def _ensure_day(self, day_key):
        if self.current_day != day_key:
            self.start_new_day(day_key)

    def check(self, now_ts, day_key):
        """Return {"allowed": bool, "reason": str} for entering a new trade.

        Does not mutate state. Idempotent.
        """
        cfg = self._config
        # Local view: if day changed, the counters that matter for blocking
        # would be reset; reflect that without mutating until register_*.
        trades_today = self.trades_today if self.current_day == day_key else 0
        daily_pnl = self.daily_pnl if self.current_day == day_key else 0.0

        if cfg.max_trades_per_day == 0 or trades_today >= cfg.max_trades_per_day:
            return {"allowed": False, "reason": REASON_MAX_TRADES}
        if (
            cfg.max_daily_loss > 0.0
            and daily_pnl <= -cfg.max_daily_loss
        ):
            return {"allowed": False, "reason": REASON_DAILY_LOSS}
        if (
            cfg.max_consecutive_losses > 0
            and self.consecutive_losses >= cfg.max_consecutive_losses
        ):
            return {"allowed": False, "reason": REASON_MAX_CONSECUTIVE_LOSSES}
        if (
            cfg.cooldown_seconds > 0
            and self.last_entry_ts is not None
            and (now_ts - self.last_entry_ts) < cfg.cooldown_seconds
        ):
            return {"allowed": False, "reason": REASON_COOLDOWN}
        return {"allowed": True, "reason": REASON_OK}

    def register_trade_entry(self, now_ts, day_key):
        """Record that a new trade was entered."""
        self._ensure_day(day_key)
        self.trades_today += 1
        self.last_entry_ts = now_ts

    def register_trade_outcome(self, pnl, day_key):
        """Record the (signed) net pnl of a completed trade.

        Conventions:
          pnl > 0  -> win  (resets consecutive_losses)
          pnl < 0  -> loss (increments consecutive_losses)
          pnl == 0 -> breakeven (resets consecutive_losses, neutral on daily_pnl)
        """
        pnl = _coerce_numeric("pnl", pnl)
        self._ensure_day(day_key)
        self.daily_pnl += pnl
        if pnl > 0.0:
            self.consecutive_losses = 0
        elif pnl < 0.0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
