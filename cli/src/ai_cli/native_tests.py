"""Run registered CTest tests, or real GoogleTest executables for older repos."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("build", type=Path)
    parser.add_argument("--cpp-only", action="store_true", help="Python bindings are tested separately")
    args = parser.parse_args()
    build = args.build.resolve()
    probe = subprocess.run(["ctest", "--test-dir", str(build), "--show-only=json-v1"],
                           capture_output=True, text=True)
    if probe.returncode:
        print(probe.stderr, flush=True)
        return probe.returncode
    tests = json.loads(probe.stdout).get("tests", [])
    if args.cpp_only:
        tests = [test for test in tests if "python" not in test["name"].lower()]
    if tests:
        print(f"Running {len(tests)} registered CTest tests", flush=True)
        argv = ["ctest", "--test-dir", str(build), "--output-on-failure", "--no-tests=error"]
        if args.cpp_only:
            argv += ["-E", "[Pp][Yy][Tt][Hh][Oo][Nn]"]
        return subprocess.call(argv)
    binaries = sorted(p for p in build.glob("test_*") if p.is_file() and os.access(p, os.X_OK))
    if not binaries:
        print("ERROR: No CTest tests or test_* executables were built. Test job cannot pass.", flush=True)
        return 1
    for binary in binaries:
        print(f"Running {binary.name}", flush=True)
        code = subprocess.call([str(binary)], cwd=build)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
