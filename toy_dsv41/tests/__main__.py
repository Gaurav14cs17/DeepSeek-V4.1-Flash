"""Run the full toy test suite: python -m toy_dsv41.tests"""

from __future__ import annotations

import traceback


def _collect():
    from . import test_components, test_smoke

    mods = [test_smoke, test_components]
    tests = []
    for mod in mods:
        for name in sorted(dir(mod)):
            if name.startswith("test_") and callable(getattr(mod, name)):
                tests.append((f"{mod.__name__.split('.')[-1]}.{name}", getattr(mod, name)))
    return tests


def main() -> int:
    tests = _collect()
    print(f"Running {len(tests)} tests…\n")
    failed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"  OK  {label}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {label}: {e}")
            traceback.print_exc()
    print()
    if failed:
        print(f"{failed}/{len(tests)} failed")
        return 1
    print(f"all {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
