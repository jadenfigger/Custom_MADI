"""Run the explorer's tests without pytest.

    python -m tools.manifold_explorer.selftest           # fast tests only
    python -m tools.manifold_explorer.selftest --slow    # include the 15 GB artifact

Identical test bodies to ``tests/manifold_explorer/test_manifold_explorer.py``;
this just imports them and calls them, so there is only one copy of each check.
"""

from __future__ import annotations

import argparse
import sys
import traceback

from tests.manifold_explorer import test_manifold_explorer as suite


def _is_slow(function) -> bool:
    marks = getattr(function, "pytestmark", [])
    return any(getattr(mark, "name", "") == "slow" for mark in marks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slow", action="store_true",
                        help="also run tests that read the production library")
    args = parser.parse_args()

    names = [name for name in dir(suite) if name.startswith("test_")]
    failures = []
    skipped = 0
    for name in sorted(names):
        function = getattr(suite, name)
        if _is_slow(function) and not args.slow:
            print(f"SKIP {name} (slow; pass --slow)")
            skipped += 1
            continue
        try:
            function()
        except BaseException as error:           # noqa: BLE001 - report everything
            if type(error).__name__ == "Skipped":
                print(f"SKIP {name}: {error}")
                skipped += 1
                continue
            failures.append((name, error))
            print(f"FAIL {name}: {error}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")

    print(f"\n{len(names) - len(failures) - skipped} passed, "
          f"{len(failures)} failed, {skipped} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
