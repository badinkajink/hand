import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from claude_pulse.usage import load_usage_entries


def _line(ts, msg_id, req_id, out=10, model="claude-opus-4-8"):
    return json.dumps({
        "type": "assistant",
        "timestamp": ts,
        "requestId": req_id,
        "sessionId": "sess-1",
        "cwd": "/home/x/proj",
        "message": {
            "id": msg_id,
            "model": model,
            "usage": {
                "input_tokens": 100,
                "cache_creation_input_tokens": 200,
                "cache_read_input_tokens": 50,
                "output_tokens": out,
            },
        },
    })


class TestUsage(unittest.TestCase):
    def _write(self, d: Path, name: str, lines):
        (d / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_parse_and_totals(self):
        with TemporaryDirectory() as td:
            d = Path(td)
            self._write(d, "a.jsonl", [
                _line("2026-07-03T10:00:00.000Z", "m1", "r1"),
                _line("2026-07-03T10:05:00.000Z", "m2", "r2", out=20),
            ])
            entries = load_usage_entries([d])
        self.assertEqual(len(entries), 2)
        e = entries[0]
        self.assertEqual(e.input_tokens, 100)
        self.assertEqual(e.output_tokens, 10)
        self.assertEqual(e.cache_creation_tokens, 200)
        self.assertEqual(e.cache_read_tokens, 50)
        self.assertEqual(e.total_tokens, 360)
        self.assertEqual(e.timestamp.tzinfo, timezone.utc)

    def test_dedup_across_files(self):
        with TemporaryDirectory() as td:
            d = Path(td)
            self._write(d, "a.jsonl", [_line("2026-07-03T10:00:00.000Z", "m1", "r1")])
            # same message.id + requestId duplicated in a resumed-session file
            self._write(d, "b.jsonl", [_line("2026-07-03T10:00:00.000Z", "m1", "r1")])
            entries = load_usage_entries([d])
        self.assertEqual(len(entries), 1)

    def test_skips_non_usage_and_malformed(self):
        with TemporaryDirectory() as td:
            d = Path(td)
            self._write(d, "a.jsonl", [
                json.dumps({"type": "user", "timestamp": "2026-07-03T10:00:00Z", "message": {"role": "user"}}),
                "this is not json{{{",
                json.dumps({"type": "summary", "summary": "x"}),
                _line("2026-07-03T10:01:00.000Z", "m1", "r1"),
            ])
            entries = load_usage_entries([d])
        self.assertEqual(len(entries), 1)

    def test_since_filter(self):
        with TemporaryDirectory() as td:
            d = Path(td)
            self._write(d, "a.jsonl", [
                _line("2026-07-03T08:00:00.000Z", "m1", "r1"),
                _line("2026-07-03T12:00:00.000Z", "m2", "r2"),
            ])
            since = datetime(2026, 7, 3, 10, 0, tzinfo=timezone.utc)
            entries = load_usage_entries([d], since=since)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].message_id, "m2")


if __name__ == "__main__":
    unittest.main()
