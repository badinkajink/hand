import unittest
from datetime import datetime, timedelta, timezone

from claude_pulse.blocks import Block, TokenCounts
from claude_pulse.config import Config
from claude_pulse.policy import Snapshot, decide, in_quiet_hours

UTC = timezone.utc
NOW = datetime(2026, 7, 3, 14, 0, tzinfo=UTC)


def _block(start_h=10, tokens=0, last_h=13):
    b = Block(
        start=datetime(2026, 7, 3, start_h, 0, tzinfo=UTC),
        duration=timedelta(hours=5),
        first_entry=datetime(2026, 7, 3, start_h, 5, tzinfo=UTC),
        last_entry=datetime(2026, 7, 3, last_h, 0, tzinfo=UTC),
    )
    b.tokens = TokenCounts(input_tokens=tokens)
    return b


def _snap(**kw):
    defaults = dict(
        now=NOW,
        active_block=_block(),
        last_usage_time=NOW - timedelta(hours=3),
        last_trigger_time=None,
        pokes_this_window=0,
    )
    defaults.update(kw)
    return Snapshot(**defaults)


class TestPolicy(unittest.TestCase):
    def test_pokes_when_idle_enough(self):
        cfg = Config(poke_after_idle_minutes=120)
        d = decide(cfg, _snap(last_usage_time=NOW - timedelta(hours=3)))
        self.assertTrue(d.should_trigger, d.reason)

    def test_skips_when_recently_active(self):
        cfg = Config(poke_after_idle_minutes=120)
        d = decide(cfg, _snap(last_usage_time=NOW - timedelta(minutes=30)))
        self.assertFalse(d.should_trigger)
        self.assertIn("idle", d.reason)

    def test_own_poke_counts_as_activity(self):
        cfg = Config(poke_after_idle_minutes=120)
        # last real usage long ago, but we poked 30m ago -> should defer
        d = decide(cfg, _snap(last_usage_time=NOW - timedelta(hours=4),
                              last_trigger_time=NOW - timedelta(minutes=30)))
        self.assertFalse(d.should_trigger)

    def test_budget_blocks(self):
        cfg = Config(poke_after_idle_minutes=120, window_token_budget=1000)
        d = decide(cfg, _snap(active_block=_block(tokens=1500),
                              last_usage_time=NOW - timedelta(hours=3)))
        self.assertFalse(d.should_trigger)
        self.assertIn("budget", d.reason)

    def test_max_pokes_blocks(self):
        cfg = Config(poke_after_idle_minutes=120, max_pokes_per_window=2)
        d = decide(cfg, _snap(pokes_this_window=2, last_usage_time=NOW - timedelta(hours=3)))
        self.assertFalse(d.should_trigger)
        self.assertIn("poked", d.reason)

    def test_min_window_left_blocks(self):
        cfg = Config(poke_after_idle_minutes=120, min_window_left_minutes=20)
        # window started at 10:00 resets 15:00; at 14:50 only 10m left
        now = datetime(2026, 7, 3, 14, 50, tzinfo=UTC)
        snap = _snap(now=now, last_usage_time=now - timedelta(hours=3))
        d = decide(cfg, snap)
        self.assertFalse(d.should_trigger)
        self.assertIn("resets", d.reason)

    def test_no_active_window_requires_flag(self):
        cfg = Config(require_active_window=True, poke_after_idle_minutes=120)
        d = decide(cfg, _snap(active_block=None, last_usage_time=NOW - timedelta(hours=8)))
        self.assertFalse(d.should_trigger)
        self.assertIn("no active", d.reason)

    def test_no_active_window_allowed_by_default(self):
        cfg = Config(require_active_window=False, poke_after_idle_minutes=120)
        d = decide(cfg, _snap(active_block=None, last_usage_time=NOW - timedelta(hours=8)))
        self.assertTrue(d.should_trigger, d.reason)

    def test_quiet_hours_blocks(self):
        cfg = Config(poke_after_idle_minutes=120, quiet_hours=["00:00-23:59"])
        d = decide(cfg, _snap(last_usage_time=NOW - timedelta(hours=3)))
        self.assertFalse(d.should_trigger)
        self.assertIn("quiet", d.reason)


class TestQuietHours(unittest.TestCase):
    def test_wraps_midnight(self):
        rng = ["23:00-07:00"]
        self.assertTrue(in_quiet_hours(datetime(2026, 7, 3, 2, 0), rng))
        self.assertTrue(in_quiet_hours(datetime(2026, 7, 3, 23, 30), rng))
        self.assertFalse(in_quiet_hours(datetime(2026, 7, 3, 12, 0), rng))

    def test_same_day_range(self):
        rng = ["09:00-17:00"]
        self.assertTrue(in_quiet_hours(datetime(2026, 7, 3, 12, 0), rng))
        self.assertFalse(in_quiet_hours(datetime(2026, 7, 3, 20, 0), rng))


if __name__ == "__main__":
    unittest.main()
