"""Tests for Prompt 1 / Step 4 - Market Intelligence Layer."""

import unittest

from hermes.market import (
    MarketIntelligenceConfig,
    REGIME_CHOP,
    REGIME_HIGH_VOLATILITY,
    REGIME_LOW_VOLATILITY,
    REGIME_TREND_DOWN,
    REGIME_TREND_UP,
    REGIME_VALUES,
    REQUIRED_INTELLIGENCE_KEYS,
    assemble_intelligence,
    classify_regime,
    compute_momentum_score,
    compute_volatility_score,
)


def _candle(o, h, l, c, v=None):
    d = {"open": float(o), "high": float(h), "low": float(l), "close": float(c)}
    if v is not None:
        d["volume"] = float(v)
    return d


def _flat_series(price, n):
    return [_candle(price, price, price, price) for _ in range(n)]


def _trend_up_series(start, step, n):
    out = []
    p = float(start)
    for _ in range(n):
        out.append(_candle(p, p + 0.001, p - 0.001, p))
        p += step
    return out


def _trend_down_series(start, step, n):
    out = []
    p = float(start)
    for _ in range(n):
        out.append(_candle(p, p + 0.001, p - 0.001, p))
        p -= step
    return out


def _high_vol_series(base, n, swing=0.05):
    out = []
    for i in range(n):
        # Alternate big up/down moves keeping mean close ~ base.
        if i % 2 == 0:
            out.append(_candle(base, base * (1 + swing), base * (1 - swing), base))
        else:
            out.append(
                _candle(base, base * (1 + swing), base * (1 - swing), base * 0.999)
            )
    return out


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = MarketIntelligenceConfig()
        self.assertEqual(c.lookback, 20)
        self.assertEqual(c.volatility_cap, 0.05)
        self.assertEqual(c.momentum_cap, 0.05)
        self.assertEqual(c.high_volatility_threshold, 0.80)
        self.assertEqual(c.low_volatility_threshold, 0.20)
        self.assertEqual(c.trend_up_threshold, 0.65)
        self.assertEqual(c.trend_down_threshold, 0.35)

    def test_rejects_bad_thresholds(self):
        with self.assertRaises(ValueError):
            MarketIntelligenceConfig(lookback=1)
        with self.assertRaises(ValueError):
            MarketIntelligenceConfig(volatility_cap=0.0)
        with self.assertRaises(ValueError):
            MarketIntelligenceConfig(momentum_cap=-0.01)
        with self.assertRaises(ValueError):
            MarketIntelligenceConfig(
                high_volatility_threshold=0.5, low_volatility_threshold=0.6
            )
        with self.assertRaises(ValueError):
            MarketIntelligenceConfig(
                trend_up_threshold=0.4, trend_down_threshold=0.5
            )


class TestVolatilityScore(unittest.TestCase):
    def test_flat_series_score_zero(self):
        candles = _flat_series(100.0, 25)
        self.assertEqual(compute_volatility_score(candles, 24), 0.0)

    def test_score_in_unit_interval(self):
        for series in (
            _flat_series(50.0, 25),
            _trend_up_series(100.0, 0.5, 25),
            _high_vol_series(200.0, 25, swing=0.1),
        ):
            for idx in range(len(series)):
                v = compute_volatility_score(series, idx)
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_high_vol_saturates(self):
        candles = _high_vol_series(100.0, 25, swing=0.20)
        self.assertEqual(compute_volatility_score(candles, 24), 1.0)

    def test_uses_only_past_and_present(self):
        candles = _flat_series(100.0, 30)
        idx = 14
        before = compute_volatility_score(candles, idx)
        # Mutate FUTURE candles to wild values; result must not change.
        for j in range(idx + 1, len(candles)):
            candles[j] = _candle(1e9, 1e9, 1e9, 1e9)
        after = compute_volatility_score(candles, idx)
        self.assertEqual(before, after)

    def test_deterministic(self):
        candles = _trend_up_series(100.0, 0.3, 25)
        a = compute_volatility_score(candles, 24)
        b = compute_volatility_score(candles, 24)
        self.assertEqual(a, b)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            compute_volatility_score([], 0)
        with self.assertRaises(ValueError):
            compute_volatility_score(_flat_series(1, 5), 5)
        with self.assertRaises(ValueError):
            compute_volatility_score(_flat_series(1, 5), -1)
        with self.assertRaises(ValueError):
            compute_volatility_score("not a list", 0)


class TestMomentumScore(unittest.TestCase):
    def test_flat_series_neutral(self):
        candles = _flat_series(100.0, 25)
        self.assertEqual(compute_momentum_score(candles, 24), 0.5)

    def test_score_in_unit_interval(self):
        for series in (
            _flat_series(50.0, 25),
            _trend_up_series(100.0, 0.5, 25),
            _trend_down_series(100.0, 0.5, 25),
            _high_vol_series(200.0, 25, swing=0.1),
        ):
            for idx in range(len(series)):
                m = compute_momentum_score(series, idx)
                self.assertGreaterEqual(m, 0.0)
                self.assertLessEqual(m, 1.0)

    def test_strong_up_moves_above_neutral(self):
        candles = _trend_up_series(100.0, 0.5, 25)  # ~12% rise over 20
        self.assertGreater(compute_momentum_score(candles, 24), 0.5)

    def test_strong_down_moves_below_neutral(self):
        candles = _trend_down_series(100.0, 0.5, 25)
        self.assertLess(compute_momentum_score(candles, 24), 0.5)

    def test_single_candle_window_returns_neutral(self):
        candles = _flat_series(100.0, 1)
        self.assertEqual(compute_momentum_score(candles, 0), 0.5)

    def test_uses_only_past_and_present(self):
        candles = _trend_up_series(100.0, 0.3, 30)
        idx = 14
        before = compute_momentum_score(candles, idx)
        for j in range(idx + 1, len(candles)):
            candles[j] = _candle(1e6, 1e6, 1e6, 1e6)
        after = compute_momentum_score(candles, idx)
        self.assertEqual(before, after)


class TestClassifyRegime(unittest.TestCase):
    def test_returns_valid_enum(self):
        for v in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            for m in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
                self.assertIn(classify_regime(v, m), REGIME_VALUES)

    def test_high_volatility_takes_precedence(self):
        # High vol overrides momentum.
        self.assertEqual(classify_regime(0.95, 0.99), REGIME_HIGH_VOLATILITY)
        self.assertEqual(classify_regime(0.95, 0.01), REGIME_HIGH_VOLATILITY)

    def test_low_volatility_takes_precedence_over_momentum(self):
        self.assertEqual(classify_regime(0.05, 0.99), REGIME_LOW_VOLATILITY)
        self.assertEqual(classify_regime(0.05, 0.01), REGIME_LOW_VOLATILITY)

    def test_trend_up_when_mid_vol_high_momentum(self):
        self.assertEqual(classify_regime(0.5, 0.80), REGIME_TREND_UP)

    def test_trend_down_when_mid_vol_low_momentum(self):
        self.assertEqual(classify_regime(0.5, 0.20), REGIME_TREND_DOWN)

    def test_chop_when_mid_everything(self):
        self.assertEqual(classify_regime(0.5, 0.5), REGIME_CHOP)

    def test_boundary_inclusivity(self):
        # At exact thresholds, the upper-bucket side wins (deterministic).
        self.assertEqual(classify_regime(0.80, 0.5), REGIME_HIGH_VOLATILITY)
        self.assertEqual(classify_regime(0.20, 0.5), REGIME_LOW_VOLATILITY)
        self.assertEqual(classify_regime(0.5, 0.65), REGIME_TREND_UP)
        self.assertEqual(classify_regime(0.5, 0.35), REGIME_TREND_DOWN)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            classify_regime(-0.1, 0.5)
        with self.assertRaises(ValueError):
            classify_regime(0.5, 1.1)
        with self.assertRaises(ValueError):
            classify_regime(True, 0.5)
        with self.assertRaises(ValueError):
            classify_regime(0.5, "0.5")


class TestAssembleIntelligence(unittest.TestCase):
    def test_required_keys_match_spec(self):
        self.assertEqual(
            set(REQUIRED_INTELLIGENCE_KEYS),
            {"regime", "volatility_score", "momentum_score"},
        )

    def test_returns_exact_key_set(self):
        candles = _trend_up_series(100.0, 0.1, 25)
        out = assemble_intelligence(candles, 24)
        self.assertEqual(set(out.keys()), set(REQUIRED_INTELLIGENCE_KEYS))

    def test_regime_always_valid(self):
        for series in (
            _flat_series(100.0, 25),
            _trend_up_series(100.0, 0.5, 25),
            _trend_down_series(100.0, 0.5, 25),
            _high_vol_series(200.0, 25, swing=0.1),
        ):
            for idx in range(len(series)):
                out = assemble_intelligence(series, idx)
                self.assertIn(out["regime"], REGIME_VALUES)

    def test_scores_in_unit_interval(self):
        candles = _high_vol_series(150.0, 25, swing=0.08)
        for idx in range(len(candles)):
            out = assemble_intelligence(candles, idx)
            self.assertGreaterEqual(out["volatility_score"], 0.0)
            self.assertLessEqual(out["volatility_score"], 1.0)
            self.assertGreaterEqual(out["momentum_score"], 0.0)
            self.assertLessEqual(out["momentum_score"], 1.0)

    def test_no_future_data_used(self):
        candles = _trend_up_series(100.0, 0.3, 30)
        idx = 12
        before = assemble_intelligence(candles, idx)
        # Replace all FUTURE candles with garbage; output must be unchanged.
        for j in range(idx + 1, len(candles)):
            candles[j] = _candle(9.99e9, 9.99e9, 9.99e9, 9.99e9)
        after = assemble_intelligence(candles, idx)
        self.assertEqual(before, after)

    def test_deterministic(self):
        candles = _trend_up_series(100.0, 0.5, 25)
        a = assemble_intelligence(candles, 24)
        b = assemble_intelligence(candles, 24)
        self.assertEqual(a, b)
        # And once more from a freshly-constructed equal series.
        candles2 = _trend_up_series(100.0, 0.5, 25)
        c = assemble_intelligence(candles2, 24)
        self.assertEqual(a, c)

    def test_missing_volume_is_tolerated(self):
        # Per spec: if volume is missing, do not fail and do not invent volume.
        candles = _trend_up_series(100.0, 0.3, 25)
        for c in candles:
            self.assertNotIn("volume", c)
        out = assemble_intelligence(candles, 24)
        self.assertIn(out["regime"], REGIME_VALUES)

    def test_volume_when_present_is_accepted(self):
        candles = [_candle(100, 100.5, 99.5, 100, v=1234.0) for _ in range(25)]
        out = assemble_intelligence(candles, 24)
        self.assertIn(out["regime"], REGIME_VALUES)

    def test_moderate_uptrend_classified_as_trend_up(self):
        # Gentle uptrend: ~2% rise over 20 candles -> momentum>=0.65,
        # vol score moderate (~0.4), so regime = trend_up.
        candles = _trend_up_series(100.0, 0.1, 25)
        out = assemble_intelligence(candles, 24)
        self.assertGreaterEqual(out["momentum_score"], 0.65)
        self.assertLess(out["volatility_score"], 0.80)
        self.assertGreater(out["volatility_score"], 0.20)
        self.assertEqual(out["regime"], REGIME_TREND_UP)

    def test_moderate_downtrend_classified_as_trend_down(self):
        candles = _trend_down_series(100.0, 0.1, 25)
        out = assemble_intelligence(candles, 24)
        self.assertLessEqual(out["momentum_score"], 0.35)
        self.assertLess(out["volatility_score"], 0.80)
        self.assertGreater(out["volatility_score"], 0.20)
        self.assertEqual(out["regime"], REGIME_TREND_DOWN)

    def test_steep_trend_with_wide_range_is_high_volatility(self):
        # Steep trend that also has a wide range -> high_volatility wins
        # because the regime classifier checks volatility first by design.
        candles = _trend_up_series(100.0, 0.5, 25)
        out = assemble_intelligence(candles, 24)
        self.assertEqual(out["regime"], REGIME_HIGH_VOLATILITY)

    def test_flat_classified_as_low_volatility(self):
        candles = _flat_series(100.0, 25)
        out = assemble_intelligence(candles, 24)
        self.assertEqual(out["regime"], REGIME_LOW_VOLATILITY)


if __name__ == "__main__":
    unittest.main()
