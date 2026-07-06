import unittest
from datetime import datetime, timedelta, timezone

from claude_pulse.blocks import active_block, compute_blocks
from claude_pulse.usage import UsageEntry


def _entry(ts: datetime, out=10) -> UsageEntry:
    return UsageEntry(
        timestamp=ts, input_tokens=100, output_tokens=out,
        cache_creation_tokens=0, cache_read_tokens=0, model="m",
        message_id=f"m{ts.isoformat()}", request_id=f"r{ts.isoformat()}",
        session_id="s", project="p", source_file="f",
    )


UTC = timezone.utc


class TestBlocks(unittest.TestCase):
    def test_single_window(self):
        base = datetime(2026, 7, 3, 10, 15, tzinfo=UTC)
        entries = [_entry(base + timedelta(minutes=i * 30)) for i in range(4)]  # 0..90m
        blocks = compute_blocks(entries, session_hours=5)
        self.assertEqual(len(blocks), 1)
        # start floored to the hour
        self.assertEqual(blocks[0].start, datetime(2026, 7, 3, 10, 0, tzinfo=UTC))
        self.assertEqual(blocks[0].resets_at, datetime(2026, 7, 3, 15, 0, tzinfo=UTC))
        self.assertEqual(blocks[0].tokens.messages, 4)

    def test_new_window_after_5h_gap(self):
        a = datetime(2026, 7, 3, 10, 0, tzinfo=UTC)
        b = a + timedelta(hours=6)  # >5h inactivity
        blocks = compute_blocks([_entry(a), _entry(b)], session_hours=5)
        self.assertEqual(len(blocks), 2)

    def test_new_window_after_5h_duration(self):
        a = datetime(2026, 7, 3, 10, 0, tzinfo=UTC)
        entries = [_entry(a + timedelta(hours=h)) for h in range(0, 7)]  # every hour, 0..6h
        blocks = compute_blocks(entries, session_hours=5)
        # window starts at 10:00; entry at 15:00 is >=5h from start -> new window
        self.assertGreaterEqual(len(blocks), 2)

    def test_active_and_remaining(self):
        start = datetime(2026, 7, 3, 10, 0, tzinfo=UTC)
        now = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)  # 2h in
        blocks = compute_blocks([_entry(start + timedelta(minutes=5))], session_hours=5)
        blk = active_block(blocks, now)
        self.assertIsNotNone(blk)
        self.assertEqual(blk.time_remaining(now), timedelta(hours=3))

    def test_no_active_when_idle_past_reset(self):
        start = datetime(2026, 7, 3, 10, 0, tzinfo=UTC)
        now = datetime(2026, 7, 3, 16, 0, tzinfo=UTC)  # 6h later, window done
        blocks = compute_blocks([_entry(start + timedelta(minutes=5))], session_hours=5)
        self.assertIsNone(active_block(blocks, now))


if __name__ == "__main__":
    unittest.main()
