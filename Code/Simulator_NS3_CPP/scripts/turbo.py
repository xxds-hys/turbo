#!/usr/bin/env python3
"""Build independent Turbo experiments or run them without invoking a compiler."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
NS3 = ROOT / "ns-3.33"
DATA = ROOT / "file-in-ctr"
EXPERIMENTS = tuple("A%05d" % i for i in range(95, 107))
WORKLOADS = tuple("%s_%d" % (model, size)
                 for model in ("GPT", "LLAM") for size in (1, 4, 8, 16, 32))


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_time(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return number


def workloads_for(experiment):
    return ("GPT_32",) if experiment in ("A00105", "A00106") else WORKLOADS


def binary_for(experiment):
    return NS3 / "build" / "scratch" / ("turbo-" + experiment)


@contextmanager
def simulator_paths(input_dir, output_dir):
    # ns-3.33 parses string options as whitespace-separated tokens.
    if sys.platform == "linux" and any(c.isspace() for c in str(input_dir) + str(output_dir)):
        with tempfile.TemporaryDirectory(prefix="turbo-paths-", dir="/tmp") as temporary:
            source, destination = Path(temporary) / "input", Path(temporary) / "output"
            source.symlink_to(input_dir, target_is_directory=True)
            destination.symlink_to(output_dir, target_is_directory=True)
            yield source, destination
    else:
        yield input_dir, output_dir


def build(experiments, jobs, configure=False):
    if sys.platform != "linux":
        raise RuntimeError("Build on Linux or inside the supplied Linux container.")
    # Waf discovers each scratch .cc as an independent executable.
    for experiment in experiments:
        source = DATA / "executableFiles" / experiment / "main.cc"
        target = NS3 / "scratch" / ("turbo-" + experiment + ".cc")
        if not target.exists() or source.read_bytes() != target.read_bytes():
            shutil.copyfile(str(source), str(target))
    waf = [sys.executable, str(NS3 / "waf")]
    if configure or not (NS3 / "build" / "c4che" / "_cache.py").is_file():
        subprocess.run(waf + ["configure", "-d", "release", "--enable-mpi",
                             "--disable-werror", "--disable-python",
                             "--disable-tests", "--disable-examples"],
                       cwd=str(NS3), check=True)
    # This ns-3 Waf needs its unselected header-generation tasks on fresh builds.
    subprocess.run(waf + ["build", "--jobs=%d" % jobs],
                   cwd=str(NS3), check=True)
    for experiment in experiments:
        if not binary_for(experiment).is_file():
            raise RuntimeError("Build did not produce " + str(binary_for(experiment)))


def run(experiment, workloads, output_root, sim_start, sim_end):
    executable = binary_for(experiment)
    if not executable.is_file():
        raise RuntimeError("Missing %s. Run the build command first." % executable)
    input_dir = DATA / "inputFiles" / experiment
    for filename in ["TOPO.txt", "CONFIG_DCQCN.txt"] + [w + ".txt" for w in workloads]:
        if not (input_dir / filename).is_file():
            raise RuntimeError("Missing input " + str(input_dir / filename))
    output_dir = output_root.resolve() / experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(NS3 / "build" / "lib") + (
        os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    with simulator_paths(input_dir, output_dir) as (runtime_input, runtime_output):
        for workload in workloads:
            command = [str(executable),
                   "--fileIdx=" + experiment + "_" + workload,
                   "--inputFileDir=" + str(runtime_input) + "/",
                   "--outputFileDir=" + str(runtime_output) + "/",
                   "--topoFileName=" + str(runtime_input / "TOPO.txt"),
                   "--configFileName=" + str(runtime_input / "CONFIG_DCQCN.txt"),
                   "--kvCacheFileName=" + str(runtime_input / (workload + ".txt")),
                   "--simStartTimeInSec=" + str(sim_start),
                   "--simEndTimeInSec=" + str(sim_end),
                   "--qlenMonitorIntervalInNs=" + (
                       "1000" if experiment in ("A00105", "A00106") else "100000")]
            print("Running %s / %s -> %s" % (experiment, workload, output_dir), flush=True)
            subprocess.run(command, cwd=str(NS3), env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list supported experiments and workloads")
    compiler = commands.add_parser("build", help="compile all or one experiment")
    compiler.add_argument("experiment", nargs="?", choices=EXPERIMENTS)
    compiler.add_argument("--jobs", type=positive_int, default=2)
    compiler.add_argument("--configure", action="store_true", help="rerun release configuration")
    runner = commands.add_parser("run", help="run a precompiled experiment")
    runner.add_argument("experiment", choices=EXPERIMENTS + ("all",))
    runner.add_argument("--workload", choices=WORKLOADS, help="default: original experiment workload set")
    runner.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "results")
    runner.add_argument("--sim-start", type=nonnegative_time, default=0.0)
    runner.add_argument("--sim-end", type=nonnegative_time, default=5000.0,
                        help="simulation seconds, not wall-clock seconds")
    runner.add_argument("--rebuild", action="store_true", help="compile this experiment before running")
    runner.add_argument("--jobs", type=positive_int, default=2)
    args = parser.parse_args(argv)
    if args.command == "list":
        for experiment in EXPERIMENTS:
            print(experiment + ": " + ", ".join(workloads_for(experiment)))
    elif args.command == "build":
        build((args.experiment,) if args.experiment else EXPERIMENTS, args.jobs, args.configure)
    else:
        if args.sim_end <= args.sim_start:
            parser.error("--sim-end must be greater than --sim-start")
        experiments = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
        if args.rebuild:
            build(experiments, args.jobs)
        for experiment in experiments:
            workloads = (args.workload,) if args.workload else workloads_for(experiment)
            run(experiment, workloads, args.output_dir, args.sim_start, args.sim_end)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Error: " + str(error), file=sys.stderr)
        sys.exit(error.returncode if isinstance(error, subprocess.CalledProcessError)
                 and error.returncode > 0 else 1)
