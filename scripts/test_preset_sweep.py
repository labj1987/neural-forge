#!/usr/bin/env python3
"""Evidence classification and preference-restoration regressions; no GPU required."""
import csv
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("preset_sweep", Path(__file__).with_name("preset-sweep.py"))
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


class PresetSweepTests(unittest.TestCase):
    def test_echoes_do_not_count_as_warmup_and_unchanged_evaluations_do(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.csv"
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["round", "evaluated", "pixels_changed", "roundtrip_ms", "eval_ms", "busy_ms", "vram_mb"])
                for i, (evaluated, value) in enumerate([(False, 999), (True, 99), (True, 3), (True, 5), (True, 101)]):
                    writer.writerow([i + 1, str(evaluated).lower(), "false", value, value, value, 200])
            summary = sweep.summarise(path, 1, 2)
            self.assertEqual(summary["busy_ms_median"], 4)
            self.assertEqual(summary["echoes"], 1)
            with self.assertRaisesRegex(RuntimeError, "need 10"):
                sweep.summarise(path, 1, 10)

    def test_live_game_lease_prevents_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            shm = Path(directory) / "shm.bin"
            with sweep.lease(shm):
                with self.assertRaisesRegex(RuntimeError, "owned by a game"):
                    with sweep.lease(shm):
                        self.fail("second writer acquired channel")
            with sweep.lease(shm):
                pass
            self.assertTrue(Path(str(shm) + ".owner").exists())

    def test_preferences_restored_when_roundtrip_fails_or_is_interrupted(self):
        for failure in (RuntimeError("GPU unavailable"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as directory:
                initial = {"preset": "7", "passes": "3", "enabled": "0", "apply_model": "0",
                           "pass0_override_mask": "0", "pass0_effective_preset": "7"}
                live = initial.copy()
                restarts = []
                def fake_run(argv):
                    if argv[0] == "roundtrip":
                        raise failure
                    if argv[1] == "restart":
                        restarts.append(1)
                    elif argv[1:3] == ["shmctl", "set"]:
                        live[argv[3]] = str(argv[4])
                        live["pass0_effective_preset"] = live["preset"]
                    return ""
                args = SimpleNamespace(cli="cli", roundtrip="roundtrip", out=Path(directory),
                                       frame="frame", format="rgba16f", width=64, height=64,
                                       presets=[0, 1], trials=1, warmup=1, samples=2)
                with patch.object(sweep, "status", side_effect=lambda _cli: live.copy()), patch.object(sweep, "run", side_effect=fake_run):
                    with self.assertRaises(type(failure)):
                        sweep.experiment(args)
                self.assertEqual(live, initial)
                self.assertEqual(len(restarts), 2)

    def test_a_preset_that_never_evaluates_is_recorded_and_the_sweep_goes_on(self):
        with tempfile.TemporaryDirectory() as directory:
            live = {"preset": "0", "passes": "1", "enabled": "1", "apply_model": "1",
                    "pass0_override_mask": "0", "pass0_effective_preset": "0"}
            def fake_run(argv):
                argv = list(map(str, argv))
                if argv[0] == "roundtrip":
                    stem = argv[argv.index("--csv") + 1]
                    with open(stem, "w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(["round", "evaluated", "pixels_changed", "roundtrip_ms", "eval_ms", "busy_ms", "vram_mb"])
                        evaluated = "false" if live["preset"] == "1" else "true"
                        for i in range(4):
                            writer.writerow([i + 1, evaluated, "true", 1, 1, 1, 0])
                    Path(argv[argv.index("--out") + 1]).write_bytes(b"answer")
                elif argv[1:3] == ["shmctl", "set"]:
                    live[argv[3]] = argv[4]
                    live["pass0_effective_preset"] = live["preset"]
                return ""
            args = SimpleNamespace(cli="cli", roundtrip="roundtrip", out=Path(directory),
                                   frame="frame", format="rgba16f", width=64, height=64,
                                   presets=[0, 1, 2], trials=1, warmup=1, samples=2)
            with patch.object(sweep, "status", side_effect=lambda _cli: live.copy()), patch.object(sweep, "run", side_effect=fake_run):
                results = sweep.experiment(args)
            self.assertEqual([r["preset"] for r in results], [0, 1, 2])
            self.assertIn("error", results[1])
            self.assertIn("answer_sha256", results[2])

    def test_locations_reads_the_cli_channel_and_dll_directory(self):
        outputs = {"status": "helper not running\n  config: /c/config.ini\n  channel: /tmp/nf/shm.bin\n",
                   "config": "shm=/tmp/nf/shm.bin\nbinaries=/b/binaries\nhelper_exe=" + __file__ + "\n"}
        with patch.object(sweep, "run", side_effect=lambda argv: outputs[argv[1]]):
            self.assertEqual(sweep.locations("cli"), (Path("/tmp/nf/shm.bin"), Path("/b/binaries")))
        # A build-tree CLI that would stop the helper and fail to start it is refused up front.
        outputs["config"] = "shm=/tmp/nf/shm.bin\nbinaries=/b/binaries\nhelper_exe=missing\n"
        with patch.object(sweep, "run", side_effect=lambda argv: outputs[argv[1]]):
            with self.assertRaisesRegex(RuntimeError, "NEURAL_FORGE_INSTALL_DIR"):
                sweep.locations("cli")

    def test_pass_override_rejected_before_any_setting_changes(self):
        with patch.object(sweep, "status", return_value={"pass0_override_mask": "64"}), patch.object(sweep, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "clear pass 0 overrides"):
                sweep.experiment(SimpleNamespace(cli="cli"))
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
