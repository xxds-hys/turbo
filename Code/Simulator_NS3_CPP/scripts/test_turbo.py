"""Launcher contract tests; not a substitute for Linux compilation."""
import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
import runpy
from unittest.mock import patch

import turbo


class ReleaseInputs(unittest.TestCase):
    def test_all_experiments_have_sources_and_default_inputs(self):
        for experiment in turbo.EXPERIMENTS:
            self.assertTrue((turbo.DATA / "executableFiles" / experiment / "main.cc").is_file())
            for filename in ["TOPO.txt", "CONFIG_DCQCN.txt"] + [
                    w + ".txt" for w in turbo.workloads_for(experiment)]:
                self.assertTrue((turbo.DATA / "inputFiles" / experiment / filename).is_file())

    def test_legacy_entrypoints_resolve_project_without_working_directory(self):
        paths = [(turbo.DATA / "executableFiles" / e / "run.py", e) for e in turbo.EXPERIMENTS]
        paths.append((turbo.DATA / "executableFiles" / "runAll.py", "all"))
        for path, experiment in paths:
            with patch.object(turbo.subprocess, "call", return_value=7) as call, \
                    patch.object(turbo.sys, "argv", [str(path), "--help"]):
                with self.assertRaises(SystemExit) as result:
                    runpy.run_path(str(path), run_name="__main__")
            self.assertEqual(result.exception.code, 7)
            self.assertEqual(call.call_args[0][0], [turbo.sys.executable,
                             str(turbo.ROOT / "scripts" / "turbo.py"), "run", experiment, "--help"])


class Launcher(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ns3 = self.root / "ns-3.33"
        self.data = self.root / "file-in-ctr"
        self.ns3.mkdir()
        for name, value in [("ROOT", self.root), ("NS3", self.ns3), ("DATA", self.data)]:
            patcher = patch.object(turbo, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        (self.ns3 / "scratch").mkdir()
        (self.ns3 / "scratch" / "main.cc").write_text("original", encoding="utf-8")
        for experiment in ("A00095", "A00105"):
            source = self.data / "executableFiles" / experiment
            source.mkdir(parents=True)
            (source / "main.cc").write_text(experiment, encoding="utf-8")
            inputs = self.data / "inputFiles" / experiment
            inputs.mkdir(parents=True)
            for filename in ("TOPO.txt", "CONFIG_DCQCN.txt", "GPT_1.txt", "GPT_32.txt"):
                (inputs / filename).write_text("input", encoding="utf-8")
            binary = turbo.binary_for(experiment)
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.touch()

    def test_direct_run_bypasses_build_and_sets_paths(self):
        with patch.object(turbo, "build") as build, patch.object(turbo.subprocess, "run") as process:
            turbo.main(["run", "A00095", "--workload", "GPT_1", "--sim-end", "0.01"])
        build.assert_not_called()
        command = process.call_args[0][0]
        self.assertEqual(command[0], str(turbo.binary_for("A00095")))
        self.assertIn("--qlenMonitorIntervalInNs=100000", command)
        self.assertIn("--simEndTimeInSec=0.01", command)
        self.assertEqual(process.call_args[1]["env"]["LD_LIBRARY_PATH"].split(turbo.os.pathsep)[0],
                         str(self.ns3 / "build" / "lib"))
        self.assertTrue((self.root / "artifacts" / "results" / "A00095").is_dir())

    def test_monitor_experiment_preserves_workload_and_interval(self):
        with patch.object(turbo.subprocess, "run") as process:
            turbo.main(["run", "A00105"])
        process.assert_called_once()
        command = process.call_args[0][0]
        self.assertIn("--qlenMonitorIntervalInNs=1000", command)
        self.assertTrue(any(arg.endswith("GPT_32.txt") for arg in command))

    def test_rebuild_precedes_execution(self):
        events = []
        with patch.object(turbo, "build", side_effect=lambda *a: events.append("build")), \
                patch.object(turbo, "run", side_effect=lambda *a: events.append("run")):
            turbo.main(["run", "A00095", "--rebuild"])
        self.assertEqual(events, ["build", "run"])

    def test_run_all_preserves_all_102_default_workloads(self):
        with patch.object(turbo, "run") as run:
            turbo.main(["run", "all"])
        self.assertEqual(run.call_count, 12)
        self.assertEqual(sum(len(call[0][1]) for call in run.call_args_list), 102)
        self.assertEqual(run.call_args_list[-1][0][1], ("GPT_32",))

    def test_build_uses_distinct_targets_without_overwriting_main(self):
        with patch.object(turbo.sys, "platform", "linux"), patch.object(turbo.subprocess, "run") as process:
            turbo.build(("A00095", "A00105"), 2)
        self.assertIn("--disable-python", process.call_args_list[0][0][0])
        self.assertFalse(any(arg.startswith("--targets")
                             for arg in process.call_args_list[1][0][0]))
        self.assertEqual((self.ns3 / "scratch" / "turbo-A00095.cc").read_text(), "A00095")
        self.assertEqual((self.ns3 / "scratch" / "main.cc").read_text(), "original")

    def test_configure_failure_stops_build(self):
        with patch.object(turbo.sys, "platform", "linux"), patch.object(
                turbo.subprocess, "run", side_effect=subprocess.CalledProcessError(3, "waf")) as process:
            with self.assertRaises(subprocess.CalledProcessError):
                turbo.build(("A00095",), 2)
        process.assert_called_once()

    def test_simulation_failure_stops_remaining_workloads(self):
        with patch.object(turbo.subprocess, "run", side_effect=subprocess.CalledProcessError(2, "binary")) as process:
            with self.assertRaises(subprocess.CalledProcessError):
                turbo.run("A00095", ("GPT_1", "GPT_32"), self.root / "results", 0, 1)
        process.assert_called_once()

    def test_missing_input_does_not_launch(self):
        (self.data / "inputFiles" / "A00095" / "TOPO.txt").unlink()
        with patch.object(turbo.subprocess, "run") as process, self.assertRaises(RuntimeError):
            turbo.run("A00095", ("GPT_1",), self.root / "results", 0, 1)
        process.assert_not_called()

    def test_invalid_time_or_jobs_rejected_before_execution(self):
        cases = [["run", "A00095", "--sim-end", value] for value in ("nan", "inf", "-1", "0")]
        cases.append(["build", "--jobs", "0"])
        with patch.object(turbo.subprocess, "run") as process, contextlib.redirect_stderr(io.StringIO()):
            for case in cases:
                with self.assertRaises(SystemExit):
                    turbo.main(case)
        process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
