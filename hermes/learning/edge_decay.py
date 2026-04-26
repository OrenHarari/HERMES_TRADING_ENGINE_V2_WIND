"""Step 6 / Part 5 - Edge Decay Detection.

Rolling-window monitor over completed-trade outcomes.

State machine:
  - Each completed trade is recorded as a 'win' (pnl > 0) or 'loss' (pnl <= 0).
    Breakeven is counted as 'loss' for the alert math (conservative).
  - Every WINDOW_SIZE (default 20) completed trades, compute that window's
    win_rate.
  - If two consecutive windows have win_rate < ALERT_THRESHOLD (default 0.45):
        edge_decay_alert = True
        log proposed +0.05 on min_confidence
  - Recovery: while alerted, watch a forward streak of completed trades; if
    win_rate over the latest RECOVERY_WINDOW (default 10) trades > 0.50, clear
    the alert.
"""

from hermes.utils.bounds import is_numeric

WINDOW_SIZE = 20
ALERT_THRESHOLD = 0.45
RECOVERY_WINDOW = 10
RECOVERY_THRESHOLD = 0.50
PROPOSED_MIN_CONFIDENCE_BUMP = 0.05


def _is_win(pnl):
    return pnl > 0.0


class EdgeDecayMonitor(object):
    """Stateful, deterministic edge-decay state machine.

    All transitions go through `update_with_pnl` (one call per completed trade)
    and are reflected in `state()`.
    """

    __slots__ = (
        "_outcomes",
        "_window_win_rates",
        "_alert",
        "_proposed_bumps",
        "_log",
    )

    def __init__(self):
        self._outcomes = []          # list of bool: True if win
        self._window_win_rates = []  # win_rate per completed window
        self._alert = False
        self._proposed_bumps = 0
        self._log = []

    @property
    def edge_decay_alert(self):
        return self._alert

    @property
    def total_observations(self):
        return len(self._outcomes)

    @property
    def proposed_min_confidence_bumps(self):
        return self._proposed_bumps

    def state(self):
        return {
            "edge_decay_alert": self._alert,
            "total_observations": len(self._outcomes),
            "completed_windows": len(self._window_win_rates),
            "proposed_min_confidence_bumps": self._proposed_bumps,
            "last_window_win_rates": list(self._window_win_rates[-2:]),
            "log": list(self._log),
        }

    def update_with_pnl(self, pnl):
        """Record one completed trade outcome and update state.

        Returns the post-update state dict.
        """
        if not is_numeric(pnl):
            raise ValueError("pnl must be numeric (non-bool, non-NaN)")
        self._outcomes.append(_is_win(pnl))

        n = len(self._outcomes)
        # Did we just complete a fresh window?
        if n % WINDOW_SIZE == 0:
            window = self._outcomes[n - WINDOW_SIZE : n]
            win_rate = sum(1 for w in window if w) / float(WINDOW_SIZE)
            self._window_win_rates.append(win_rate)
            self._evaluate_alert()

        if self._alert:
            self._evaluate_recovery()

        return self.state()

    def _evaluate_alert(self):
        if self._alert:
            return
        if len(self._window_win_rates) < 2:
            return
        last_two = self._window_win_rates[-2:]
        if all(wr < ALERT_THRESHOLD for wr in last_two):
            self._alert = True
            self._proposed_bumps += 1
            self._log.append(
                {
                    "event": "edge_decay_alert_raised",
                    "last_two_window_win_rates": list(last_two),
                    "alert_threshold": ALERT_THRESHOLD,
                    "proposed_min_confidence_bump": PROPOSED_MIN_CONFIDENCE_BUMP,
                }
            )

    def _evaluate_recovery(self):
        if not self._alert:
            return
        if len(self._outcomes) < RECOVERY_WINDOW:
            return
        recent = self._outcomes[-RECOVERY_WINDOW:]
        recovery_rate = sum(1 for w in recent if w) / float(RECOVERY_WINDOW)
        if recovery_rate > RECOVERY_THRESHOLD:
            self._alert = False
            self._log.append(
                {
                    "event": "edge_decay_alert_cleared",
                    "recovery_rate": recovery_rate,
                    "recovery_threshold": RECOVERY_THRESHOLD,
                    "recovery_window": RECOVERY_WINDOW,
                }
            )
