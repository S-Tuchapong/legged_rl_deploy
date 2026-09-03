#!/usr/bin/env python3
"""Run the external-payload sweep (series A to E) end to end.

For every condition this script rewrites `payload.xml`, brings up a fresh
MuJoCo / preprocessor / deploy stack, hands over to `eval_beam_sim2sim.py`,
and tears the stack down again.  The model is only read at simulator start-up,
so each condition genuinely needs its own process; nothing here changes the
evaluation protocol itself.

Every condition gets its own results directory, which is what keeps the
evaluator's generated file names from colliding, and the applied `payload.xml`
is copied in beside them so a result can always be traced back to the exact
physics that produced it.

Run it from a shell prepared exactly as docs/sim2sim_beam_evaluation.md
describes:

    source src/unitree_lowlevel/scripts/setup.sh lo jazzy
    export ROS_DOMAIN_ID=1
    python3 src/legged_rl_deploy/scripts/run_payload_sweep.py --all --trials 32

To drive a single condition by hand instead, use `payload_conditions.py apply`
and follow the four-terminal recipe in that document; `--dry-run` below prints
the exact commands this script would use.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import payload_conditions as pc

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parents[1]
WORKSPACE = SRC_DIR.parent
POLICY_DIR = ("src/legged_rl_deploy/policies/go2/unitree_rl_mjlab/"
              "beam_depth_distillation")
EVAL_SCRIPT = SCRIPTS_DIR / "eval_beam_sim2sim.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------- environment

def check_environment(args) -> None:
    """Fail fast on the same conditions eval_beam_sim2sim.py enforces, so a
    long sweep cannot die on the very first evaluator invocation."""
    iface = args.iface
    problems: list[str] = []
    domain = os.environ.get("ROS_DOMAIN_ID")
    if domain is None or not domain.isdigit() or not 1 <= int(domain) <= 232:
        problems.append(f"ROS_DOMAIN_ID must be an integer in [1, 232] (got {domain!r})")
    if os.environ.get("RMW_IMPLEMENTATION") != "rmw_cyclonedds_cpp":
        problems.append("RMW_IMPLEMENTATION must be rmw_cyclonedds_cpp")
    if os.environ.get("NetworkInterface") != "lo":
        problems.append("NetworkInterface must be exactly 'lo'")
    uri = os.environ.get("CYCLONEDDS_URI")
    if not uri:
        problems.append("CYCLONEDDS_URI must bind CycloneDDS to 'lo'")
    else:
        try:
            root = ET.fromstring(uri)
        except ET.ParseError:
            problems.append("CYCLONEDDS_URI must contain inline CycloneDDS XML")
        else:
            names = [e.attrib.get("name") for e in root.iter()
                     if e.tag.rsplit("}", 1)[-1] == "NetworkInterface"]
            if not names or any(n != "lo" for n in names):
                problems.append("every CYCLONEDDS_URI NetworkInterface must name 'lo'")
    if iface != os.environ.get("NetworkInterface", iface):
        problems.append(f"--iface {iface} disagrees with NetworkInterface")
    if shutil.which("ros2") is None:
        problems.append("ros2 is not on PATH")
    # A stray virtualenv on PATH is the classic way to lose rclpy halfway
    # through an overnight sweep, so prove the interpreter works up front.
    if shutil.which(args.python) is None:
        problems.append(f"--python {args.python!r} is not on PATH")
    elif subprocess.run([args.python, "-c", "import rclpy, numpy"],
                        capture_output=True).returncode != 0:
        problems.append(f"{args.python!r} cannot import rclpy and numpy; it is "
                        f"{shutil.which(args.python)}. Deactivate any virtualenv "
                        "or pass --python explicitly")
    if not os.environ.get("DISPLAY"):
        problems.append("DISPLAY must be set: the depth camera needs a GL context "
                        "even with --no-viewer")
    if problems:
        raise SystemExit(
            "error: environment is not ready for an evaluation run:\n"
            + "".join(f"  - {p}\n" for p in problems)
            + "\nPrepare the shell first:\n"
            f"  cd {WORKSPACE}\n"
            "  source src/unitree_lowlevel/scripts/setup.sh lo jazzy\n"
            "  export ROS_DOMAIN_ID=1\n")


# --------------------------------------------------------- stale stack guard

# A second simulator on the same ROS domain is silently fatal: the evaluator
# sees two /unitree_mujoco/episode_status publishers and two reset services, so
# episode ids interleave and every trial dies with "episode_id changed while
# running".  Match the installed executables and the `ros2 run` wrappers, not
# the bare package names, which also appear in unrelated command lines.
STACK_PATTERNS = (
    "ros2 run unitree_mujoco",
    "ros2 run legged_rl_deploy",
    "/lib/unitree_mujoco/",
    "/lib/legged_rl_deploy/",
)


def ancestor_pids() -> set[int]:
    """The runner's own shell may carry these patterns in its command line."""
    pids, pid = set(), os.getpid()
    while pid > 1:
        pids.add(pid)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        except (OSError, IndexError):
            break
        pid = int(fields[1])
    return pids


def find_stack_processes(exclude: set[int] | None = None) -> list[tuple[int, str]]:
    exclude = (exclude or set()) | ancestor_pids()
    try:
        listing = subprocess.run(["ps", "-eo", "pid=,args="],
                                 capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    found = []
    for line in listing.splitlines():
        pid_text, _, args = line.strip().partition(" ")
        if not pid_text.isdigit() or int(pid_text) in exclude:
            continue
        if any(pattern in args for pattern in STACK_PATTERNS):
            found.append((int(pid_text), args.strip()))
    return found


def kill_stack_processes(processes: list[tuple[int, str]]) -> None:
    for pid, _ in processes:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + (5.0 if sig != signal.SIGKILL else 2.0)
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.2)
            else:
                continue
            break


def assert_no_stale_stack(args, *, when: str) -> None:
    stale = find_stack_processes()
    if not stale:
        return
    if args.kill_stale:
        print(f"  killing {len(stale)} stale process(es) {when}")
        kill_stack_processes(stale)
        time.sleep(2.0)
        stale = find_stack_processes()
        if not stale:
            return
    listing = "".join(f"  - pid {pid}: {cmd[:110]}\n" for pid, cmd in stale)
    raise SystemExit(
        f"error: a simulator stack is already running {when}:\n{listing}"
        "\nTwo simulators on one ROS domain make every trial fail with "
        "'episode_id changed while running'.\nStop them, or rerun with "
        "--kill-stale.\n")


# ------------------------------------------------------------ process control

class Stack:
    """The three resident processes an evaluation needs, as one unit."""

    def __init__(self, args, log_dir: Path) -> None:
        self.args = args
        self.log_dir = log_dir
        self.procs: list[tuple[str, subprocess.Popen, object]] = []

    def commands(self) -> list[tuple[str, list[str], str | None]]:
        """Third field is text to feed the process on stdin.  legged_rl_deploy_node
        opens with an interactive `Press Enter to continue...` guard
        (legged_rl_deploy_node.cpp, std::cin.ignore()), which blocks forever when
        the process is started detached, so the newline has to be supplied here."""
        return [
            ("mujoco", [
                "ros2", "run", "unitree_mujoco", "unitree_mujoco",
                "-r", "go2", "-s", self.args.scene,
                "--depth-camera", "--no-joystick", "--beam-monitor",
                *([] if self.args.viewer else ["--no-viewer"]),
                "--episode-timeout", f"{self.args.sim_episode_timeout}",
            ], None),
            ("preprocessor", [
                "ros2", "run", "legged_rl_deploy", "depth_image_preprocessor_node.py",
                "--ros-args", "--params-file",
                f"{POLICY_DIR}/depth_image_preprocessor.yaml",
            ], None),
            ("deploy", [
                "ros2", "run", "legged_rl_deploy", "legged_rl_deploy_node",
                self.args.iface, f"{POLICY_DIR}/config.yaml",
                "--ros-args", "-p", "evaluation_mode:=true",
            ], "\n"),
        ]

    def start(self) -> None:
        for name, command, stdin_text in self.commands():
            handle = (self.log_dir / f"{name}.log").open("w", buffering=1)
            handle.write(f"$ {' '.join(command)}\n\n")
            proc = subprocess.Popen(
                command, cwd=WORKSPACE, stdout=handle, stderr=subprocess.STDOUT,
                # start_new_session detaches from the controlling terminal, so a
                # child that reads stdin would stall on SIGTTIN rather than see
                # the operator's keyboard.  Give it a pipe or nothing at all.
                stdin=subprocess.PIPE if stdin_text else subprocess.DEVNULL,
                text=True, start_new_session=True,
            )
            if stdin_text:
                proc.stdin.write(stdin_text)
                proc.stdin.flush()
            self.procs.append((name, proc, handle))
            # The simulator has to own the DDS graph before the two clients look
            # for its topics and services.
            time.sleep(self.args.stagger if name == "mujoco" else 1.0)
            if proc.poll() is not None:
                self.stop()
                raise RuntimeError(
                    f"{name} exited immediately with code {proc.returncode}; "
                    f"see {self.log_dir / f'{name}.log'}")

    def check(self) -> None:
        for name, proc, _ in self.procs:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} died with code {proc.returncode}")

    def stop(self) -> None:
        for name, proc, handle in reversed(self.procs):
            if proc.poll() is None:
                for sig, grace in ((signal.SIGINT, 8.0), (signal.SIGTERM, 4.0)):
                    try:
                        os.killpg(os.getpgid(proc.pid), sig)
                    except (ProcessLookupError, PermissionError):
                        break
                    try:
                        proc.wait(timeout=grace)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                else:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    proc.wait(timeout=5.0)
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
            handle.close()
        self.procs.clear()

    def __enter__(self) -> "Stack":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()


# ------------------------------------------------------------------ one cell

def eval_command(args) -> list[str]:
    return [
        args.python, str(EVAL_SCRIPT),
        "--trials", str(args.trials),
        "--velocity", f"{args.velocity}",
        "--seed", str(args.seed),
        "--reset-position-jitter-m", f"{args.reset_position_jitter_m}",
        "--reset-yaw-jitter-rad", f"{args.reset_yaw_jitter_rad}",
        "--startup-timeout", f"{args.startup_timeout}",
        "--episode-timeout", f"{args.eval_episode_timeout}",
    ]


def find_summary(directory: Path) -> Path | None:
    found = sorted(directory.glob("*.summary.json"))
    return found[0] if found else None


def completed_summary(directory: Path) -> Path | None:
    """A summary only counts as done if every requested trial actually ran.  A
    crashed attempt still leaves a summary behind, and resuming past it would
    silently drop the condition from the sweep."""
    path = find_summary(directory)
    if path is None:
        return None
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if summary.get("fatal_error") is not None:
        return None
    if summary.get("completed_trials") != summary.get("requested_trials"):
        return None
    return path


def archive_previous_attempt(out_dir: Path) -> Path | None:
    """eval_beam_sim2sim.py refuses to overwrite its own output, so a retry into
    a directory that already holds a crashed attempt would die instantly.  Move
    the old artefacts aside rather than deleting them."""
    stale = [p for p in out_dir.glob("*")
             if p.is_file() and (p.suffix in {".jsonl", ".log", ".xml"}
                                 or p.name.endswith(".summary.json"))]
    if not stale:
        return None
    index = 1
    while (archive := out_dir / f"previous-attempt-{index}").exists():
        index += 1
    archive.mkdir()
    for path in stale:
        path.rename(archive / path.name)
    return archive


ACTIVE_STACK: "Stack | None" = None


def install_signal_handlers() -> None:
    """The children run in their own sessions so a terminal Ctrl-C never reaches
    them; without this the runner can die and leave a simulator orphaned, which
    then poisons every later run."""
    def handler(signum, _frame):
        if ACTIVE_STACK is not None:
            print(f"\nreceived signal {signum}, stopping the stack")
            ACTIVE_STACK.stop()
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, handler)


def run_condition(condition, args, out_dir: Path) -> dict:
    global ACTIVE_STACK
    out_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_previous_attempt(out_dir)
    if archived is not None:
        print(f"  moved a previous attempt into {archived.name}/")
    pc.apply(condition)
    shutil.copy(pc.PAYLOAD_XML, out_dir / "payload.xml")
    shutil.copy(pc.CONTACTS_XML, out_dir / "payload_contacts.xml")
    record: dict = {
        "condition": condition.cid,
        "series": condition.series,
        "title": condition.title,
        "started_at": utc_now(),
        "derived": pc.derived(condition),
    }
    started = time.monotonic()
    try:
        with Stack(args, out_dir) as stack:
            ACTIVE_STACK = stack
            completed = subprocess.run(eval_command(args), cwd=out_dir)
            record["eval_returncode"] = completed.returncode
            stack.check()
    except RuntimeError as error:
        record["eval_returncode"] = None
        record["error"] = str(error)
    finally:
        ACTIVE_STACK = None
    record["wall_duration_s"] = time.monotonic() - started
    record["finished_at"] = utc_now()

    summary_path = find_summary(out_dir)
    if summary_path is not None:
        summary = json.loads(summary_path.read_text())
        record["summary_file"] = summary_path.name
        for key in ("success_rate", "successes", "completed_trials", "result_counts",
                    "fatal_error", "run_id"):
            record[key] = summary.get(key)
    return record


# ---------------------------------------------------------------------- main

def select_conditions(args) -> list[pc.Condition]:
    if args.conditions:
        ids = [c.strip() for c in args.conditions.split(",") if c.strip()]
        return [pc.resolve(c) for c in ids]
    if args.series:
        wanted = {s.strip().upper() for s in args.series.split(",") if s.strip()}
        chosen = [c for c in pc.default_sweep() if c.series.upper() in wanted]
        if not chosen:
            raise SystemExit(f"error: no conditions in series {sorted(wanted)}")
        return chosen
    return pc.default_sweep()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pick = parser.add_mutually_exclusive_group()
    pick.add_argument("--all", action="store_true",
                      help="every condition in the default sweep (the default)")
    pick.add_argument("--series", help="comma separated series letters, e.g. A,B")
    pick.add_argument("--conditions", help="comma separated ids, e.g. S0,A1,B0.6")

    parser.add_argument("--trials", type=int, default=32,
                        help="trials per condition (default 32 for screening; "
                             "raise to 128 for the conditions that matter)")
    parser.add_argument("--velocity", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42,
                        help="keep this identical across conditions: trial i then "
                             "starts from the same pose everywhere, which makes the "
                             "comparison paired")
    parser.add_argument("--reset-position-jitter-m", type=float, default=0.10)
    parser.add_argument("--reset-yaw-jitter-rad", type=float, default=0.20)
    parser.add_argument("--results-dir", default=None,
                        help="default: payload-sweep-<timestamp> in the CWD")
    parser.add_argument("--scene", default="scene_bridge.xml")
    parser.add_argument("--viewer", action="store_true",
                        help="show the MuJoCo window; off by default because a "
                             "sweep runs unattended. A working DISPLAY is needed "
                             "either way, for the offscreen depth context")
    parser.add_argument("--iface", default="lo")
    parser.add_argument("--python", default="python3",
                        help="interpreter used for eval_beam_sim2sim.py; it must "
                             "be the ROS one that can import rclpy")
    parser.add_argument("--sim-episode-timeout", type=float, default=20.0)
    parser.add_argument("--eval-episode-timeout", type=float, default=25.0)
    parser.add_argument("--startup-timeout", type=float, default=60.0,
                        help="raised over the evaluator default because the stack "
                             "is cold at the start of every condition")
    parser.add_argument("--stagger", type=float, default=6.0,
                        help="seconds to let the simulator settle before its clients")
    parser.add_argument("--resume", action="store_true",
                        help="skip conditions whose results directory already holds "
                             "a summary")
    parser.add_argument("--kill-stale", action="store_true",
                        help="stop any simulator stack that is already running "
                             "instead of refusing to start")
    parser.add_argument("--keep-going", action="store_true",
                        help="carry on after a failed condition")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the exact commands, run nothing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    conditions = select_conditions(args)

    if args.results_dir:
        results_dir = Path(args.results_dir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = Path.cwd() / f"payload-sweep-{stamp}"

    print(f"conditions ({len(conditions)}): "
          f"{', '.join(c.cid for c in conditions)}")
    print(f"trials per condition: {args.trials}   seed: {args.seed} (paired)")
    print(f"results: {results_dir}")

    if args.dry_run:
        stack = Stack(args, results_dir)
        print("\nper condition, in its own results directory:")
        for name, command, stdin_text in stack.commands():
            note = "   (fed a newline on stdin)" if stdin_text else ""
            print(f"  [{name}]  (cwd {WORKSPACE}){note}\n    " + " ".join(command))
        print("  [eval]  (cwd <results-dir>/<condition>)\n    "
              + " ".join(eval_command(args)))
        print("\nand payload.xml is rewritten before each one:")
        for condition in conditions:
            d = condition.payload
            spec = "baseline" if d is None else (
                f"{d.mass:g} kg at ({d.com[0]:+.3f},{d.com[1]:+.3f},{d.com[2]:+.3f})")
            print(f"  {condition.cid:6s} {spec:34s} {condition.title}")
        return 0

    check_environment(args)
    install_signal_handlers()
    assert_no_stale_stack(args, when="before the sweep started")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "manifest.json").write_text(
        json.dumps({
            "started_at": utc_now(),
            "workspace": str(WORKSPACE),
            "args": vars(args),
            "conditions": [c.cid for c in conditions],
        }, indent=2) + "\n", encoding="utf-8")

    records: list[dict] = []
    sweep_path = results_dir / "sweep_summary.json"
    started = time.monotonic()
    failures = 0

    for index, condition in enumerate(conditions, start=1):
        out_dir = results_dir / condition.cid
        if args.resume and completed_summary(out_dir) is not None:
            print(f"\n[{index}/{len(conditions)}] {condition.cid}: already done, skipping")
            continue
        elapsed = time.monotonic() - started
        eta = ""
        if records:
            per = elapsed / len(records)
            eta = f"   eta {per * (len(conditions) - index + 1) / 60:.0f} min"
        print(f"\n[{index}/{len(conditions)}] {condition.cid}: {condition.title}{eta}")

        record = run_condition(condition, args, out_dir)
        records.append(record)
        # A leak here would corrupt every later condition, so catch it now
        # rather than after another six hours of results.
        assert_no_stale_stack(args, when=f"after condition {condition.cid}")
        rate = record.get("success_rate")
        if record.get("eval_returncode") == 0 and rate is not None:
            print(f"  -> success rate {rate:.3f} "
                  f"({record.get('successes')}/{record.get('completed_trials')}) "
                  f"in {record['wall_duration_s'] / 60:.1f} min")
        else:
            failures += 1
            print(f"  -> FAILED: returncode={record.get('eval_returncode')} "
                  f"{record.get('error', '')}")
            print(f"     logs in {out_dir}")
            if not args.keep_going:
                sweep_path.write_text(json.dumps(records, indent=2) + "\n",
                                      encoding="utf-8")
                print("\nstopping; pass --keep-going to carry on past a failure, "
                      "or --resume to pick up where this left off")
                return 1
        sweep_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {sweep_path}")
    print(f"total wall time {(time.monotonic() - started) / 60:.1f} min, "
          f"{failures} failed condition(s)")
    print(f"\n{'condition':10s} {'success':>8s}  {'dmass':>7s}  I ratio roll/pitch/yaw")
    for record in records:
        rate = record.get("success_rate")
        d = record["derived"]
        print(f"{record['condition']:10s} "
              f"{'  n/a  ' if rate is None else f'{rate:7.3f}'}  "
              f"{d['mass_increase_pct']:+6.1f}%  "
              "x{:.2f} x{:.2f} x{:.2f}".format(*d["inertia_ratio"]))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
