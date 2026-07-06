import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from claude_pulse.config import Config, load_config, load_env


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg, source = load_config(None, environ={})
        self.assertIsNone(source)
        self.assertEqual(cfg.session_hours, 5.0)
        self.assertEqual(cfg.poke_after_idle_minutes, 120.0)
        self.assertEqual(cfg.command[0], "claude")

    def test_toml_file(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "cfg.toml"
            p.write_text(
                'poke_after_idle_minutes = 90\n'
                'session_hours = 5\n'
                'quiet_hours = ["23:00-07:00"]\n'
                'command = ["bash", "job.sh"]\n'
                'window_token_budget = 500000\n',
                encoding="utf-8",
            )
            cfg, source = load_config(str(p), environ={})
        self.assertEqual(source, p)
        self.assertEqual(cfg.poke_after_idle_minutes, 90.0)
        self.assertEqual(cfg.quiet_hours, ["23:00-07:00"])
        self.assertEqual(cfg.command, ["bash", "job.sh"])
        self.assertEqual(cfg.window_token_budget, 500000)

    def test_toml_section(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "cfg.toml"
            p.write_text('[claude-pulse]\npoke_after_idle_minutes = 45\n', encoding="utf-8")
            cfg, _ = load_config(str(p), environ={})
        self.assertEqual(cfg.poke_after_idle_minutes, 45.0)

    def test_env_overrides(self):
        env = {
            "CLAUDE_PULSE_POKE_AFTER_IDLE_MINUTES": "77",
            "CLAUDE_PULSE_DRY_RUN": "true",
            "CLAUDE_PULSE_DATA_DIRS": "/a,/b",
        }
        overrides = load_env(env)
        self.assertEqual(overrides["poke_after_idle_minutes"], 77.0)
        self.assertTrue(overrides["dry_run"])
        self.assertEqual(overrides["data_dirs"], ["/a", "/b"])

    def test_precedence_cli_over_env_over_file(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "cfg.toml"
            p.write_text("poke_after_idle_minutes = 30\n", encoding="utf-8")
            env = {"CLAUDE_PULSE_POKE_AFTER_IDLE_MINUTES": "60"}
            cfg, _ = load_config(str(p), cli_overrides={"poke_after_idle_minutes": 90.0}, environ=env)
        self.assertEqual(cfg.poke_after_idle_minutes, 90.0)  # CLI wins

    def test_budget_none_parsing(self):
        overrides = load_env({"CLAUDE_PULSE_WINDOW_TOKEN_BUDGET": "none"})
        self.assertIsNone(overrides["window_token_budget"])


if __name__ == "__main__":
    unittest.main()
