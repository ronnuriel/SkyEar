from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

DEFAULT_CONFIG = "configs/config_station.yaml"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path, argv = _extract_config(argv)
    if not argv or argv[0] in {"-h", "--help"}:
        _print_help()
        return 0

    command = argv.pop(0)
    if command == "station":
        return _run_station(config_path, argv)
    if command == "server":
        return _run_with_argv("skyear-server", argv, _server_main)
    if command == "dashboard":
        return _run_dashboard(argv)
    if command == "monitor":
        return _run_monitor(config_path, argv)
    if command == "rec":
        return _run_recording(config_path, argv)
    if command == "check":
        return _run_check(config_path, argv)
    if command == "setup":
        return _run_setup(config_path, argv)
    if command == "dev":
        return _run_dev(config_path, argv)
    if command == "release":
        return _run_release(argv)

    print(f"Unknown skyear command: {command}", file=sys.stderr)
    _print_help()
    return 2


def _extract_config(argv: list[str]) -> tuple[str, list[str]]:
    config = os.environ.get("SKYEAR_CONFIG", DEFAULT_CONFIG)
    remaining: list[str] = []
    idx = 0
    while idx < len(argv):
        item = argv[idx]
        if item == "--config":
            if idx + 1 >= len(argv):
                raise SystemExit("--config requires a path")
            config = argv[idx + 1]
            idx += 2
            continue
        if item.startswith("--config="):
            config = item.split("=", 1)[1]
            idx += 1
            continue
        remaining.append(item)
        idx += 1
    return config, remaining


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _ensure_config_arg(argv: list[str], config_path: str) -> list[str]:
    if any(item == "--config" or item.startswith("--config=") for item in argv):
        return list(argv)
    return ["--config", config_path, *argv]


@contextmanager
def _patched_argv(program: str, argv: list[str]) -> Iterator[None]:
    old_argv = sys.argv
    sys.argv = [program, *argv]
    try:
        yield
    finally:
        sys.argv = old_argv


def _run_with_argv(program: str, argv: list[str], func: Callable[[], Any]) -> int:
    with _patched_argv(program, argv):
        try:
            result = func()
        except SystemExit as exc:
            return int(exc.code or 0)
    return int(result or 0)


def _station_main() -> None:
    from station.station_agent import main as station_main

    station_main()


def _server_main() -> None:
    from server.cli import main as server_main

    server_main()


def _run_station(config_path: str, argv: list[str]) -> int:
    return _run_with_argv("skyear-station", _ensure_config_arg(argv, config_path), _station_main)


def _run_dashboard(argv: list[str]) -> int:
    from dashboard.cli import _run_streamlit

    try:
        _run_streamlit("app.py", extra_args=argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _run_monitor(config_path: str, argv: list[str]) -> int:
    from dashboard.cli import _run_streamlit
    from station.local_monitor import local_monitor_paths

    cfg = _load_config(config_path)
    station_cfg = cfg.get("station", {}) or {}
    station_id = str(station_cfg.get("station_id") or station_cfg.get("id") or "station_001")
    state_path, history_path = local_monitor_paths(cfg, station_id)
    extra = list(argv)
    if "--" not in extra:
        extra = ["--", *extra]
    if "--state" not in extra:
        extra.extend(["--state", str(state_path)])
    if "--history" not in extra:
        extra.extend(["--history", str(history_path)])
    try:
        _run_streamlit("local_station_app.py", extra_args=extra)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def _run_recording(config_path: str, argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: skyear rec {start|mark|stop|state|summary} ...")
        return 0
    action = argv.pop(0)
    if action == "start":
        parser = argparse.ArgumentParser(prog="skyear rec start")
        parser.add_argument("session_name", nargs="?", default="session")
        parser.add_argument("--label")
        parser.add_argument("--note")
        args = parser.parse_args(argv)
        from tools.recording_control import _post

        _post(config_path, "/recording/start", {"session_name": args.session_name, "label": args.label, "note": args.note})
        return 0
    if action == "mark":
        parser = argparse.ArgumentParser(prog="skyear rec mark")
        parser.add_argument("label")
        parser.add_argument("--note")
        parser.add_argument("--distance-m", type=float)
        parser.add_argument("--bearing-deg", type=float)
        parser.add_argument("--drone-model")
        args = parser.parse_args(argv)
        from tools.recording_control import _post

        _post(
            config_path,
            "/recording/mark",
            {
                "label": args.label,
                "note": args.note,
                "distance_m": args.distance_m,
                "bearing_deg": args.bearing_deg,
                "drone_model": args.drone_model,
                "source": "manual",
            },
        )
        return 0
    if action == "stop":
        from tools.recording_control import _post

        _post(config_path, "/recording/stop", {})
        return 0
    if action == "state":
        from tools.recording_control import _get

        _get(config_path, "/recording/state")
        return 0
    if action == "summary":
        cfg = _load_config(config_path)
        root = str((cfg.get("recording", {}) or {}).get("root", "runtime/recordings"))
        from tools.summarize_recording import main as summary_main

        return summary_main(["--root", root, *argv])
    print(f"Unknown recording command: {action}", file=sys.stderr)
    return 2


def _run_check(config_path: str, argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: skyear check {audio|two-mic|server|hf} ...")
        return 0
    action = argv.pop(0)
    if action == "audio":
        from tools.check_audio import main as check_audio_main

        return check_audio_main(_ensure_config_arg(argv, config_path))
    if action == "two-mic":
        from tools.check_two_mic_direction import main as check_two_mic_main

        return check_two_mic_main(_ensure_config_arg(argv, config_path))
    if action == "server":
        cfg = _load_config(config_path)
        url = str((cfg.get("server", {}) or {}).get("url", "http://127.0.0.1:8080/events"))
        return _run_with_argv("skyear-check-server", ["--url", url, *argv], _check_server_main)
    if action == "hf":
        return _run_station(config_path, ["--hf-smoke-test", *argv])
    print(f"Unknown check command: {action}", file=sys.stderr)
    return 2


def _run_setup(config_path: str, argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: skyear setup {audio|station|volt2|array} ...")
        return 0
    action = argv.pop(0)
    profile = {
        "audio": "auto",
        "station": "auto",
        "volt2": "volt2_dual_mic",
        "array": "circular_clockwise",
    }.get(action)
    if profile is None:
        print(f"Unknown setup command: {action}", file=sys.stderr)
        return 2
    from tools.setup_audio import main as setup_audio_main

    return setup_audio_main(["--config", config_path, "--profile", profile, *argv])


def _check_server_main() -> None:
    from tools.check_server import main as check_server_main

    check_server_main()


def _run_dev(config_path: str, argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: skyear dev {debug-wav|benchmark|simulate} ...")
        return 0
    action = argv.pop(0)
    if action == "debug-wav":
        return _run_with_argv("skyear-debug-harmonic-wav", _ensure_config_arg(argv, config_path), _debug_wav_main)
    if action == "benchmark":
        return _run_with_argv("skyear-run-benchmarks", argv, _benchmark_main)
    if action == "simulate":
        if argv and argv[0] in {"fiber-grid", "fiber_grid"}:
            argv.pop(0)
            return _run_with_argv("skyear-simulate-fiber-grid", argv, _simulate_fiber_grid_main)
        if argv and argv[0] in {"multi-target", "multi_target"}:
            argv.pop(0)
            return _run_with_argv("skyear-simulate-multi-target", argv, _simulate_multi_target_main)
        if argv and argv[0] == "geo":
            argv.pop(0)
        return _run_with_argv("skyear-simulate-geo-events", argv, _simulate_main)
    print(f"Unknown dev command: {action}", file=sys.stderr)
    return 2


def _debug_wav_main() -> None:
    from tools.debug_harmonic_wav import main as debug_main

    debug_main()


def _benchmark_main() -> None:
    from tools.run_dataset_benchmarks import run_benchmarks, parse_args

    run_benchmarks(parse_args())


def _simulate_main() -> None:
    from tools.simulate_geo_events import main as simulate_main

    simulate_main()


def _simulate_fiber_grid_main() -> None:
    from tools.simulate_fiber_grid import main as simulate_main

    simulate_main()


def _simulate_multi_target_main() -> None:
    from tools.simulate_multi_target import main as simulate_main

    simulate_main()


def _run_release(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: skyear release {preflight|tag} ...")
        return 0
    action = argv.pop(0)
    if action == "preflight":
        return _release_preflight()
    if action == "tag":
        parser = argparse.ArgumentParser(prog="skyear release tag")
        parser.add_argument("version")
        parser.add_argument("--push", action="store_true")
        args = parser.parse_args(argv)
        return _release_tag(str(args.version), push=bool(args.push))
    print(f"Unknown release command: {action}", file=sys.stderr)
    return 2


def _release_preflight() -> int:
    commands = [
        ["git", "status", "--short"],
        [sys.executable, "-m", "compileall", "station", "server", "dashboard", "shared", "tools", "tests"],
        [sys.executable, "-m", "pytest", "-q"],
        ["bash", "scripts/release_field_alpha_check.sh"],
        [sys.executable, "-m", "build"],
    ]
    for command in commands:
        print("$ " + " ".join(command))
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            return int(result.returncode)
    return 0


def _release_tag(version: str, *, push: bool = False) -> int:
    status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    if status:
        print("Release tag requires a clean worktree.", file=sys.stderr)
        print(status, file=sys.stderr)
        return 1
    package_version = _pyproject_version()
    expected = _version_from_tag(version)
    if package_version != expected:
        print(
            f"pyproject.toml version is {package_version}, expected {expected} for tag {version}.",
            file=sys.stderr,
        )
        return 1
    subprocess.run(["git", "tag", "-a", version, "-m", f"SkyEar {version}"], check=True)
    if push:
        subprocess.run(["git", "push", "origin", version], check=True)
    else:
        print(f"Tag created locally. Push with: git push origin {version}")
    return 0


def _pyproject_version() -> str:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise SystemExit("Could not find project.version in pyproject.toml")
    return match.group(1)


def _version_from_tag(tag: str) -> str:
    normalized = tag[1:] if tag.startswith("v") else tag
    return normalized.split("-", 1)[0]


def _print_help() -> None:
    print(
        """
SkyEar command groups:
  skyear station                 Run station from active config
  skyear server                  Run central API server
  skyear dashboard               Run central dashboard
  skyear monitor                 Run local station monitor
  skyear rec start <session>     Start local station recording
  skyear rec mark <label>        Add recording marker
  skyear rec stop|state|summary  Control or inspect recording
  skyear check audio|two-mic|server|hf
  skyear setup audio|station|volt2|array
  skyear dev debug-wav|benchmark|simulate
  skyear release preflight
  skyear release tag <version>

Config: --config PATH, SKYEAR_CONFIG, or configs/config_station.yaml
""".strip()
    )


if __name__ == "__main__":
    raise SystemExit(main())
