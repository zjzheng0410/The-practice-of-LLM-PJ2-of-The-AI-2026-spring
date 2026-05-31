from collections import Counter, defaultdict
from fractions import Fraction
from typing import Any

from answer_utils import normalize_answer
from evaluation.metrics import LONG_QUESTION_LENGTH, safe_accuracy


NUMERIC_TYPES = {"integer", "decimal", "fraction", "mixed_fraction", "percent"}
GEOMETRY_KEYWORDS = ("面积", "周长", "体积", "圆柱", "半径", "直径")
RATE_RATIO_KEYWORDS = ("百分比", "百分数", "比例", "倍数", "速度", "工程")
MULTI_STEP_KEYWORDS = ("先", "后", "多次", "剩下", "又", "比")
SCALE_RATIOS = {
    Fraction(10, 1),
    Fraction(100, 1),
    Fraction(1000, 1),
    Fraction(1, 10),
    Fraction(1, 100),
    Fraction(1, 1000),
}


def _parse_numeric_answer(value: Any) -> Fraction | None:
    try:
        text = normalize_answer(value)
    except (TypeError, ValueError):
        return None

    try:
        if text.endswith("%"):
            return Fraction(text[:-1]) / 100
        if "_" in text and "/" in text:
            sign = -1 if text.startswith("-") else 1
            body = text[1:] if sign < 0 else text
            whole, fraction_text = body.split("_", 1)
            numerator, denominator = fraction_text.split("/", 1)
            return sign * (Fraction(int(whole), 1) + Fraction(int(numerator), int(denominator)))
        if "/" in text:
            return Fraction(text)
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def _has_any_keyword(question: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in question for keyword in keywords)


def build_attribution_tags(record: dict[str, Any], long_question_threshold: int = LONG_QUESTION_LENGTH) -> list[str]:
    tags: list[str] = []
    answer_type = str(record.get("answer_type", "unknown"))
    pred_answer_type = str(record.get("pred_answer_type", "unknown"))
    extract_status = str(record.get("extract_status", "unknown"))
    question = str(record.get("question", ""))

    if extract_status not in {"matched", "disabled"}:
        tags.append("postprocess_failure")
    if answer_type != pred_answer_type:
        tags.append("answer_type_mismatch")

    answer_value = _parse_numeric_answer(record.get("answer", ""))
    pred_value = _parse_numeric_answer(record.get("pred_answer", ""))
    if answer_value is not None and pred_value is not None:
        if answer_type == pred_answer_type and answer_type in NUMERIC_TYPES and answer_value != pred_value:
            tags.append("numeric_value_error")
        if answer_type in NUMERIC_TYPES and pred_answer_type in NUMERIC_TYPES and answer_value == pred_value:
            tags.append("fraction_percent_error")
        if answer_value != 0 and pred_value != 0:
            ratio = abs(pred_value / answer_value)
            if ratio in SCALE_RATIOS:
                tags.append("unit_scale_suspect")

    if len(question) > long_question_threshold:
        tags.append("long_question")
    if _has_any_keyword(question, GEOMETRY_KEYWORDS):
        tags.append("geometry_keyword")
    if _has_any_keyword(question, RATE_RATIO_KEYWORDS):
        tags.append("rate_ratio_keyword")
    if _has_any_keyword(question, MULTI_STEP_KEYWORDS):
        tags.append("multi_step_keyword")
    if not tags:
        tags.append("unknown_reason")
    return tags


def build_wrong_analysis(
    wrong_records: list[dict[str, Any]],
    long_question_threshold: int = LONG_QUESTION_LENGTH,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    annotated_records: list[dict[str, Any]] = []
    tag_counter: Counter[str] = Counter()
    answer_type_counter: Counter[str] = Counter()
    pred_answer_type_counter: Counter[str] = Counter()
    extract_status_counter: Counter[str] = Counter()
    cross_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for record in wrong_records:
        tags = build_attribution_tags(record, long_question_threshold=long_question_threshold)
        annotated = dict(record)
        annotated["attribution_tags"] = tags
        annotated_records.append(annotated)

        answer_type = str(record.get("answer_type", "unknown"))
        pred_answer_type = str(record.get("pred_answer_type", "unknown"))
        extract_status = str(record.get("extract_status", "unknown"))
        answer_type_counter[answer_type] += 1
        pred_answer_type_counter[pred_answer_type] += 1
        extract_status_counter[extract_status] += 1
        for tag in tags:
            tag_counter[tag] += 1
            cross_counter[answer_type][tag] += 1

    total_wrong = len(wrong_records)
    summary = {
        "total_wrong": total_wrong,
        "tag_counts": {
            tag: {
                "count": count,
                "ratio": safe_accuracy(count, total_wrong),
            }
            for tag, count in sorted(tag_counter.items())
        },
        "by_answer_type": dict(sorted(answer_type_counter.items())),
        "by_pred_answer_type": dict(sorted(pred_answer_type_counter.items())),
        "by_extract_status": dict(sorted(extract_status_counter.items())),
        "answer_type_x_attribution_tag": {
            answer_type: dict(sorted(counter.items()))
            for answer_type, counter in sorted(cross_counter.items())
        },
    }
    return summary, annotated_records

