"""Compile all phase scripts without running long experiments."""

from pathlib import Path
import py_compile


ROOT = Path(__file__).resolve().parents[1]
PHASES = ROOT / "src" / "phases"


def main() -> None:
    failures = []
    for path in sorted(PHASES.glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"OK {path.relative_to(ROOT)}")
        except Exception as exc:
            failures.append((path, exc))
            print(f"FAIL {path.relative_to(ROOT)}: {exc}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
