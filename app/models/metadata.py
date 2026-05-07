from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PolicyMetadata:
    """表示一条政策文档的元数据。"""

    # 文档唯一编号，是整个项目里串联 metadata、原始文件、清洗结果和切片结果的主键。
    doc_id: str
    # 政策标题，通常用于展示、检索结果引用和摘要输出。
    title: str
    # 政策所属地区，例如上海、江苏、深圳。
    region: str
    # 政策层级，例如省级、市级。
    level: str
    # 发文机关或联合发文单位的主责任单位。
    issuer: str
    # 发布日期。当前先保留原始字符串，后续如有需要再统一转日期类型。
    publish_date: str
    # 政策类型，例如行动方案、若干措施、实施方案。
    policy_type: str
    # 政策主题标签，便于后续做检索过滤、比较和聚类分析。
    theme: str
    # 文档在项目中的定位，例如 core / supplement。
    tier: str
    # 数据状态标记，例如 official_text、official_pdf 等衍生状态对应的文本可信度说明。
    status: str
    # 原始文件格式，目前项目中主要是 pdf 和 txt。
    source_format: str
    # 文号。有些政策没有公开文号，因此保持可选。
    doc_no: str = ""
    # 原始政策来源链接，便于追溯和后续引用。
    source_url: str = ""
    # 备注信息，用来记录人工整理时的重要说明。
    notes: str = ""
