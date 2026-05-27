from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 兼容直接运行 `python app\eval\run_eval.py` 的场景。
if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from app.api.server import AskRequest, ask
from app.memory.store import get_session_store


DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "cases.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "outputs" / "eval_report.json"


@dataclass(frozen=True, slots=True)
class EvalCheckResult:
    field: str
    expected: Any
    actual: Any
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "passed": self.passed,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Policy Agent 的最小评测集。")
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=f"评测样例文件路径，默认 {DEFAULT_CASES_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"评测结果输出路径，默认 {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        help="只运行指定 case id，便于快速抽样验证。",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    session_id = case.get("session_id") or f"eval-{case['id']}"
    session_store = get_session_store()
    session_store.sessions.pop(session_id, None)

    history = case.get("history", [])
    for turn in history:
        ask(
            AskRequest(
                query=turn["query"],
                top_k=int(turn.get("top_k", case.get("top_k", 5))),
                session_id=session_id,
            )
        )

    result = ask(
        AskRequest(
            query=case["query"],
            top_k=int(case.get("top_k", 5)),
            session_id=session_id,
        )
    )

    checks = evaluate_expected_fields(result, case.get("expected", {}))
    passed = all(item.passed for item in checks)

    return {
        "id": case["id"],
        "query": case["query"],
        "session_id": session_id,
        "passed": passed,
        "checks": [item.to_dict() for item in checks],
        "manual_checks": list(case.get("manual_checks", [])),
        "result": result,
    }


def evaluate_expected_fields(
    result: dict[str, Any],
    expected: dict[str, Any],
) -> list[EvalCheckResult]:
    checks: list[EvalCheckResult] = []

    field_mapping = {
        "resolved_action": ("resolved_action",),
        "response_mode": ("response_mode",),
        "retrieval_goal": ("retrieval_goal",),
        "focus": ("focus",),
        "route": ("route",),
        "strategy": ("strategy",),
    }

    for field, expected_value in expected.items():
        path = field_mapping.get(field, (field,))
        actual_value = extract_nested(result, path)
        checks.append(
            EvalCheckResult(
                field=field,
                expected=expected_value,
                actual=actual_value,
                passed=actual_value == expected_value,
            )
        )

    return checks


def extract_nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def build_summary(report: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(report)
    passed = sum(1 for item in report if item["passed"])
    failed = total - passed
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round((passed / total) * 100, 2) if total else 0.0,
    }


def write_report(report: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": build_summary(report),
        "cases": report,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def print_summary(report: list[dict[str, Any]]) -> None:
    summary = build_summary(report)
    print("=" * 72)
    print("Policy Agent Eval Summary")
    print("=" * 72)
    print(f"总样例数: {summary['total']}")
    print(f"通过: {summary['passed']}")
    print(f"失败: {summary['failed']}")
    print(f"通过率: {summary['pass_rate']}%")
    print()

    for item in report:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['id']} | {item['query']}")
        for check in item["checks"]:
            marker = "OK" if check["passed"] else "X"
            print(
                f"  - {marker} {check['field']}: expected={check['expected']} actual={check['actual']}"
            )
        if item["manual_checks"]:
            print("  - manual_checks:")
            for text in item["manual_checks"]:
                print(f"    * {text}")
        print()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    if args.case_ids:
        allowed_case_ids = set(args.case_ids)
        cases = [item for item in cases if item["id"] in allowed_case_ids]
    report = [run_case(case) for case in cases]
    output_path = write_report(report, args.output)
    print_summary(report)
    print(f"[OK] 评测报告已写入: {output_path}")


if __name__ == "__main__":
    main()
