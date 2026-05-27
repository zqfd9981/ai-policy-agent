from __future__ import annotations

import unittest

from app.eval.run_eval import evaluate_expected_fields


class EvalRunnerTests(unittest.TestCase):
    def test_evaluate_expected_fields_compares_core_state_fields(self) -> None:
        result = {
            "resolved_action": "compare",
            "response_mode": "scenario_advice_compare",
            "retrieval_goal": "compare_regions",
            "focus": "location_choice",
            "route": "compare",
            "strategy": "compare",
        }
        expected = {
            "resolved_action": "compare",
            "response_mode": "scenario_advice_compare",
            "retrieval_goal": "compare_regions",
            "focus": "location_choice",
            "route": "compare",
            "strategy": "compare",
        }

        checks = evaluate_expected_fields(result, expected)

        self.assertEqual(len(checks), 6)
        self.assertTrue(all(item.passed for item in checks))


if __name__ == "__main__":
    unittest.main()
