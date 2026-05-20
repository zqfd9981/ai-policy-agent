from __future__ import annotations

import unittest

from app.tools.summarize_policy import (
    extract_fallback_overview,
    passes_section_gate,
    trim_policy_preamble,
)


class SummarizePolicyRuleTests(unittest.TestCase):
    def test_trim_policy_preamble_keeps_policy_intro(self) -> None:
        raw_text = (
            "上海市经济和信息化委员会 文件\n"
            "沪经信智〔2025〕489号\n"
            "关于印发某措施的通知\n"
            "有关单位：\n"
            "为贯彻落实国家发展新一代人工智能的战略部署，"
            "加快实施人工智能行动，制定本措施。"
        )

        trimmed = trim_policy_preamble(raw_text)

        self.assertTrue(trimmed.startswith("为贯彻落实"))
        self.assertNotIn("沪经信智", trimmed)

    def test_target_audience_gate_requires_real_audience_signal(self) -> None:
        self.assertTrue(
            passes_section_gate(
                "申报主体为在上海市内注册、具备独立法人资格的企业。",
                title_path_str="四、申报主体要求",
                section="target_audiences",
            )
        )
        self.assertFalse(
            passes_section_gate(
                "发放6亿元算力券，加强算力调度平台建设。",
                title_path_str="1.降低智能算力使用成本",
                section="target_audiences",
            )
        )

    def test_application_condition_gate_requires_condition_signal(self) -> None:
        self.assertTrue(
            passes_section_gate(
                "申报主体应具备支撑应用示范场景建设和运营的必要基础。",
                title_path_str="四、申报主体要求",
                section="application_conditions",
            )
        )
        self.assertFalse(
            passes_section_gate(
                "支持战略性领军人才牵头组建人工智能新型研发机构。",
                title_path_str="6.打造产业创新服务平台",
                section="application_conditions",
            )
        )

    def test_extract_fallback_overview_skips_header_like_units(self) -> None:
        payloads = (
            {
                "chunk_id": "DOC001_0001",
                "doc_id": "DOC001",
                "title_path_str": "",
                "text": (
                    "上海市经济和信息化委员会 文件\n"
                    "沪经信智〔2025〕489号\n"
                    "关于印发某措施的通知\n"
                    "有关单位：\n"
                    "为贯彻落实国家发展新一代人工智能的战略部署，制定本措施。"
                ),
                "metadata": {"region": "上海"},
            },
        )

        overview = extract_fallback_overview(payloads, limit=1)

        self.assertEqual(len(overview), 0)


if __name__ == "__main__":
    unittest.main()
