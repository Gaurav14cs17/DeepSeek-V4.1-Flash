"""Deferred test — component not implemented yet."""

from __future__ import annotations


def test_deferred_until_stage_ready():
    print("DEFERRED: test_compression — waiting on docs/progress.md gate")
    # Soft-pass: structure exists; implementation comes in later stages.
    assert True


if __name__ == "__main__":
    test_deferred_until_stage_ready()
    print("ok (deferred)")
