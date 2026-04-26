"""Risk configuration."""


class RiskConfig(object):
    """Baseline risk parameters.

    Attributes:
      max_trades_per_day:        hard cap on number of trades per session day
      max_daily_loss:            absolute (positive) cap on cumulative loss
                                 per day; once daily_pnl <= -max_daily_loss,
                                 further trades are blocked
      max_consecutive_losses:    blocks trading after N losses in a row
      cooldown_seconds:          minimum gap between trade entries
    """

    __slots__ = (
        "max_trades_per_day",
        "max_daily_loss",
        "max_consecutive_losses",
        "cooldown_seconds",
    )

    def __init__(
        self,
        max_trades_per_day=10,
        max_daily_loss=1000.0,
        max_consecutive_losses=4,
        cooldown_seconds=60,
    ):
        if not isinstance(max_trades_per_day, int) or max_trades_per_day < 0:
            raise ValueError("max_trades_per_day must be int >= 0")
        if isinstance(max_daily_loss, bool) or not isinstance(
            max_daily_loss, (int, float)
        ):
            raise ValueError("max_daily_loss must be numeric")
        if max_daily_loss < 0.0:
            raise ValueError("max_daily_loss must be >= 0")
        if (
            not isinstance(max_consecutive_losses, int)
            or max_consecutive_losses < 0
        ):
            raise ValueError("max_consecutive_losses must be int >= 0")
        if not isinstance(cooldown_seconds, int) or cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be int >= 0")
        self.max_trades_per_day = max_trades_per_day
        self.max_daily_loss = float(max_daily_loss)
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_seconds = cooldown_seconds
