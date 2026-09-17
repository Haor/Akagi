"""Akagi JSONL entrypoint; launch a separately installed native Python runtime."""
import os
from pathlib import Path
import subprocess
import sys


def runtime_python(root):
    supplied = os.environ.get("RIN_NATIVE_PYTHON")
    if supplied:
        python = Path(supplied).expanduser().resolve()
    else:
        runtime = Path(os.environ.get("RIN_NATIVE_RUNTIME", root / "runtime/python"))
        python = runtime / ("python.exe" if os.name == "nt" else "bin/python3")
    if not python.is_file():
        raise RuntimeError("Native Python is missing; install runtime/python or set RIN_NATIVE_PYTHON")
    return python


def main():
    # MJAI is UTF-8 regardless of the Windows console or launcher locale.
    sys.stdin.reconfigure(encoding="utf-8", errors="strict")
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:strict"
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(root / "src")
    command = [str(runtime_python(root)), "-m", "rin.inference.native.client", "--root", str(root),
               "--seat", sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AKAGI_PLAYER_ID", "0")]
    settings = os.environ.get("AKAGI_BOT_CONFIG")
    if not settings and (root / "runtime-settings.json").is_file():
        settings = str(root / "runtime-settings.json")
    if settings:
        command.extend(["--settings", settings])
    child = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=sys.stderr, text=True, encoding="utf-8", errors="strict", bufsize=1,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            child.stdin.write(line)
            child.stdin.flush()
            response = child.stdout.readline()
            if not response:
                raise RuntimeError(f"Native client exited before replying: {child.poll()}")
            sys.stdout.write(response)
            sys.stdout.flush()
        child.stdin.close()
        return child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"RIN native error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
