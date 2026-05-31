from typing import Any


WEAK_ANSWER_TYPES = ("decimal", "fraction", "mixed_fraction")
RATE_RATIO_KEYWORDS = (
    "比例",
    "百分比",
    "百分数",
    "倍数",
    "速度",
    "工程",
    "比例尺",
    "相遇",
    "平均速度",
)
UNIT_SCALE_KEYWORDS = (
    "长度",
    "面积",
    "体积",
    "重量",
    "容量",
    "千米",
    "公里",
    "米",
    "分米",
    "厘米",
    "毫米",
    "吨",
    "千克",
    "公斤",
    "克",
    "升",
    "毫升",
    "平方米",
    "平方厘米",
    "立方米",
    "立方厘米",
)
GEOMETRY_KEYWORDS = ("面积", "周长", "体积", "圆柱", "半径", "直径", "长方体", "正方体")
MULTI_STEP_KEYWORDS = ("先", "后", "剩下", "又", "比", "多次")

FEATURE_TAG_ORDER = ("weak_answer_type", "rate_ratio", "unit_scale", "geometry", "multi_step")
REQUIRED_SAMPLE_FIELDS = ("id", "question", "answer", "answer_type")


def _sample_prefix(row_index: int | None) -> str:
    if row_index is None:
        return "样本"
    return f"第 {row_index} 条样本"


def _has_any_keyword(question: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in question for keyword in keywords)


def validate_feature_sample(row: dict[str, Any], row_index: int | None = None) -> None:
    prefix = _sample_prefix(row_index)
    for key in REQUIRED_SAMPLE_FIELDS:
        if key not in row:
            raise KeyError(f"{prefix} 缺少字段：{key}")
    if not isinstance(row["question"], str) or not row["question"]:
        raise ValueError(f"{prefix} question 必须是非空字符串")
    if not isinstance(row["answer_type"], str) or not row["answer_type"]:
        raise ValueError(f"{prefix} answer_type 必须是非空字符串")


def build_feature_tags(row: dict[str, Any], row_index: int | None = None) -> list[str]:
    validate_feature_sample(row, row_index=row_index)

    question = row["question"]
    answer_type = row["answer_type"]
    tag_flags = {
        "weak_answer_type": answer_type in WEAK_ANSWER_TYPES,
        "rate_ratio": _has_any_keyword(question, RATE_RATIO_KEYWORDS),
        "unit_scale": _has_any_keyword(question, UNIT_SCALE_KEYWORDS),
        "geometry": _has_any_keyword(question, GEOMETRY_KEYWORDS),
        # multi_step 只作为分析标签，不单独触发复制。
        "multi_step": _has_any_keyword(question, MULTI_STEP_KEYWORDS),
    }
    # 固定标签输出顺序，保证报告和测试结果可复现。
    return [tag for tag in FEATURE_TAG_ORDER if tag_flags[tag]]
