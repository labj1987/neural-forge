#!/usr/bin/env python3
"""Compare Feature 18 presets through the helper API, with no game running.

Requires trigger_helper_roundtrip built from this tree (the --csv option).
Restarts the helper before each trial to give every preset fresh model/flow history.
Identical outputs are candidates for equivalent behaviour, not proof of fallback.
"""
import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import statistics
import subprocess


def run(argv):
    result = subprocess.run(list(map(str, argv)), text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=180)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed ({result.returncode}):\n{result.stdout}")
    return result.stdout


def locations(cli):
    """The channel the CLI and helper use (config.ini's shm= wins over $NEURAL_FORGE_SHM) and the
    DLL directory the helper loads from."""
    channel = next((line.split(":", 1)[1].strip() for line in run([cli, "status"]).splitlines()
                    if line.strip().startswith("channel:")), None)
    config = dict(line.split("=", 1) for line in run([cli, "config"]).splitlines() if "=" in line)
    binaries, helper = config.get("binaries"), config.get("helper_exe")
    if not channel or not binaries:
        raise RuntimeError("the CLI did not report its channel and binaries directory")
    # A CLI run from the build tree finds no helper next to itself, so its `restart` would stop
    # the helper and fail to start it: refuse before anything is stopped.
    if not helper or helper == "missing" or not Path(helper).is_file():
        raise RuntimeError("the CLI cannot find neural-forge-helper.exe; set NEURAL_FORGE_INSTALL_DIR "
                           "to the installed lib directory (~/.local/share/neural-forge/lib/neural-forge)")
    return Path(channel), Path(binaries)


class NoEvaluations(RuntimeError):
    """A trial whose helper never evaluated enough frames (a preset the feature refuses)."""


def status(cli):
    return dict(line.split("=", 1) for line in run([cli, "shmctl", "status"]).splitlines()
                if "=" in line and not line.startswith("#"))


@contextlib.contextmanager
def lease(shm):
    # Same inode/lock as ownership.rs. Do not unlink: a game must not get a second inode.
    fd = os.open(str(shm) + ".owner", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        meta = os.fstat(fd)
        if not stat.S_ISREG(meta.st_mode) or meta.st_uid != os.getuid():
            raise RuntimeError("unsafe channel ownership file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("channel is owned by a game or another experiment; close it first") from exc
        yield
    finally:
        os.close(fd)


def digest(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def summarise(path, warmup, samples):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    # Warmup counts evaluated frames, not echoes while CreateFeature is pending.
    evaluated = [row for row in rows if row["evaluated"] == "true"]
    measured = evaluated[warmup:]
    if len(measured) < samples:
        raise NoEvaluations(f"only {len(measured)} measured evaluations; need {samples}")
    measured = measured[:samples]
    result = {"evaluated": len(evaluated), "echoes": len(rows) - len(evaluated),
              "samples": len(measured)}
    for key in ("eval_ms", "busy_ms", "roundtrip_ms"):
        values = [float(row[key]) for row in measured]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise RuntimeError(f"invalid {key}")
        result[key + "_median"] = statistics.median(values)
    peak = max(int(row["vram_mb"]) for row in measured)
    result["vram_mb_peak"] = peak if peak > 0 else None
    return result


def experiment(args):
    initial = status(args.cli)
    if initial.get("pass0_override_mask") != "0":
        raise RuntimeError("use the current CLI and clear pass 0 overrides before this experiment")
    # These are the only preferences modified by this experiment or the roundtrip tool.
    keys = ("preset", "passes", "enabled", "apply_model")
    saved = {key: initial[key] for key in keys}
    results = []
    try:
        for key in ("passes", "enabled", "apply_model"):
            run([args.cli, "shmctl", "set", key, "1"])
        for trial in range(args.trials):
            # Rotate order to reduce systematic warm-driver / temperature bias.
            order = args.presets[trial % len(args.presets):] + args.presets[:trial % len(args.presets)]
            for preset in order:
                stem = args.out / f"preset-{preset:02d}-trial-{trial + 1}"
                run([args.cli, "shmctl", "set", "preset", preset])
                run([args.cli, "restart"])
                current = status(args.cli)
                if any(current[key] != str(value) for key, value in
                       (("preset", preset), ("passes", 1), ("enabled", 1), ("apply_model", 1))):
                    raise RuntimeError("helper restarted with unexpected experiment settings")
                if current.get("pass0_effective_preset") != str(preset):
                    raise RuntimeError("pass 0 does not resolve to the requested preset")
                command = [args.roundtrip, "--" + args.format, args.frame,
                           "--width", args.width, "--height", args.height,
                           "--repeat", args.warmup + args.samples + 64,
                           "--out", str(stem) + ".raw", "--csv", str(stem) + ".csv"]
                log = run(command)
                Path(str(stem) + ".log").write_text(log)
                try:
                    row = {"preset": preset, "trial": trial + 1,
                           **summarise(str(stem) + ".csv", args.warmup, args.samples),
                           "answer_sha256": digest(str(stem) + ".raw")}
                    print(f"preset {preset}, trial {trial + 1}: busy {row['busy_ms_median']:.3f} ms", flush=True)
                except NoEvaluations as exc:
                    # Recorded, and the sweep goes on: one refused preset must not end the run.
                    row = {"preset": preset, "trial": trial + 1, "error": str(exc)}
                    print(f"preset {preset}, trial {trial + 1}: {exc}", flush=True)
                results.append(row)
                (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        # SIGINT/SIGTERM and failed subprocesses also pass through this restoration.
        failures = []
        for key, value in saved.items():
            try:
                run([args.cli, "shmctl", "set", key, value])
            except Exception as exc:
                failures.append(str(exc))
        try:
            run([args.cli, "restart"])
            restored = status(args.cli)
            if any(restored[key] != value for key, value in saved.items()):
                failures.append("restored settings differ from initial settings")
        except Exception as exc:
            failures.append(str(exc))
        if failures:
            raise RuntimeError("Could not restore preferences: " + "; ".join(failures))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default="neural-forge-cli")
    parser.add_argument("--roundtrip", type=Path, required=True)
    parser.add_argument("--shm", type=Path, required=True,
                        help="running helper channel; also used for its ownership lease")
    parser.add_argument("--dll", type=Path, required=True, help="DLL actually loaded by the helper")
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--format", choices=("rgba8", "rgba16f"), default="rgba16f")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--presets", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.width, args.height, args.trials, args.samples) < 1 or args.warmup < 1:
        parser.error("dimensions, trial/sample counts and warmup must be positive")
    if any(p < 0 or p > 15 for p in args.presets) or len(set(args.presets)) != len(args.presets):
        parser.error("presets must be distinct numbers in 0..15")
    if args.frame.stat().st_size != args.width * args.height * (8 if args.format == "rgba16f" else 4):
        parser.error("frame byte length does not match dimensions and format")
    if not args.shm.is_file() or not args.dll.is_file() or not args.roundtrip.is_file():
        parser.error("channel, DLL and roundtrip executable must exist")
    channel, binaries = locations(args.cli)
    if channel.resolve() != args.shm.resolve():
        parser.error(f"--shm is not the channel the CLI and helper use ({channel})")
    if args.dll.resolve() != (binaries / "nvngx_dlssnr.dll").resolve():
        parser.error(f"--dll is not the DLL the helper loads ({binaries / 'nvngx_dlssnr.dll'})")
    args.roundtrip = args.roundtrip.resolve()
    os.environ["NEURAL_FORGE_SHM"] = str(args.shm.resolve())
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    with lease(args.shm.resolve()):
        args.out.mkdir(parents=True, exist_ok=False)
        metadata = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "dll_sha256": digest(args.dll), "frame_sha256": digest(args.frame),
                    "roundtrip_sha256": digest(args.roundtrip), "initial_status": status(args.cli),
                    "scope": "still-frame helper comparison; not game FPS or temporal quality"}
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        results = experiment(args)
        groups = {}
        for row in results:
            groups.setdefault(row.get("answer_sha256", "no-evaluation"), []).append([row["preset"], row["trial"]])
        (args.out / "output-groups.json").write_text(json.dumps(groups, indent=2) + "\n")
        print("Saved measured helper timings and raw outputs. Matching hashes do not prove preset fallback.")


if __name__ == "__main__":
    main()
