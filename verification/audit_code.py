"""Automatic suspicious-pattern + deferred-module audit."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGES = [
    REPO / "DeepSeekFlashV4-Mini",
    REPO / "DeepSeekFlashV4.1-Mini",
]

SUSPICIOUS = re.compile(
    r"\b(TODO|FIXME|NotImplementedError|dummy|placeholder|fake_|mock_|pass\s*$)\b",
    re.I | re.M,
)


@dataclass
class Finding:
    path: str
    kind: str
    detail: str


def scan_deferred() -> list[Finding]:
    out: list[Finding] = []
    for pkg in PACKAGES:
        for path in pkg.rglob("*.py"):
            if "__pycache__" in path.parts or "oldDATA" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "DEFERRED = True" in text or "raise NotImplementedError" in text:
                out.append(
                    Finding(
                        str(path.relative_to(REPO)),
                        "DEFERRED_OR_NOT_IMPLEMENTED",
                        "Module raises NotImplementedError or marks DEFERRED=True",
                    )
                )
    return out


def scan_soft_pass_tests() -> list[Finding]:
    out: list[Finding] = []
    for pkg in PACKAGES:
        for path in (pkg / "tests").glob("test_*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "DEFERRED" in text and "assert True" in text:
                out.append(
                    Finding(
                        str(path.relative_to(REPO)),
                        "SOFT_PASS_TEST",
                        "Deferred test soft-passes with assert True",
                    )
                )
    return out


def scan_suspicious_in_model() -> list[Finding]:
    out: list[Finding] = []
    for pkg in PACKAGES:
        model_dir = pkg / "model"
        if not model_dir.exists():
            continue
        for path in model_dir.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(text.splitlines(), 1):
                if SUSPICIOUS.search(line) and "docstring" not in line.lower():
                    # skip deferred stubs' intentional NotImplementedError
                    if "NotImplementedError" in line and "DEFERRED" in text:
                        continue
                    out.append(
                        Finding(
                            f"{path.relative_to(REPO)}:{i}",
                            "SUSPICIOUS",
                            line.strip()[:160],
                        )
                    )
    return out


def main() -> int:
    findings = scan_deferred() + scan_soft_pass_tests() + scan_suspicious_in_model()
    results = REPO / "verification" / "results"
    results.mkdir(parents=True, exist_ok=True)
    payload = [asdict(f) for f in findings]
    (results / "code_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"code_audit: {len(findings)} findings → verification/results/code_audit.json")
    for f in findings[:40]:
        print(f"  [{f.kind}] {f.path}: {f.detail}")
    if len(findings) > 40:
        print(f"  ... +{len(findings) - 40} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
