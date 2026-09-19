"""Run: python -m toy_v4_flash.tests"""

from __future__ import annotations

import traceback

from . import test_smoke


def main() -> int:
    tests = [
        (n, getattr(test_smoke, n))
        for n in sorted(dir(test_smoke))
        if n.startswith("test_") and callable(getattr(test_smoke, n))
    ]
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
