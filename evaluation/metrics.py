from collections import Counter, defaultdict
from typing import Any


LONG_QUESTION_LENGTH = 100
FRACTION_TYPES = {"fraction", "mixed_fraction"}


def safe_accuracy(correct: int, total: int) -> float | None:
    if total == 0:
        return None
    return correct / total


def build_metrics(raw_records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(raw_records)
    correct = sum(1 for row in raw_records if row["correct"])

    by_type_total = Counter(row["answer_type"] for row in raw_records)
    by_type_correct = Counter(row["answer_type"] for row in raw_records if row["correct"])
    by_type = {
        answer_type: {
            "total": by_type_total[answer_type],
            "correct": by_type_correct[answer_type],
            "accuracy": safe_accuracy(by_type_correct[answer_type], by_type_total[answer_type]),
        }
        for answer_type in sorted(by_type_total)
    }

    special_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_records:
        if len(str(row["question"])) > LONG_QUESTION_LENGTH:
            special_groups["long_question"].append(row)
        if row["answer_type"] in FRACTION_TYPES:
            special_groups["fraction"].append(row)
        if row["answer_type"] == "percent":
            special_groups["percent"].append(row)

    special_metrics = {}
    for name, rows in special_groups.items():
        group_correct = sum(1 for row in rows if row["correct"])
        special_metrics[name] = {
            "total": len(rows),
            "correct": group_correct,
            "accuracy": safe_accuracy(group_correct, len(rows)),
        }

    postprocess_failure_count = sum(
        1 for row in raw_records if row["extract_status"] not in {"matched", "disabled"}
    )
    return {
        "total": total,
        "correct": correct,
        "accuracy": safe_accuracy(correct, total),
        "by_answer_type": by_type,
        "postprocess_failure_count": postprocess_failure_count,
        "postprocess_failure_rate": safe_accuracy(postprocess_failure_count, total),
        "special_metrics": special_metrics,
        "long_question_threshold": LONG_QUESTION_LENGTH,
    }

