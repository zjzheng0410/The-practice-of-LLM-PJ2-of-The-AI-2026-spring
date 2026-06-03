import argparse
import json
from pathlib import Path
from typing import Any

from answer_utils import compare_answers, extract_final_answer


DEFAULT_TRAIN_FILE = "data/clean_data/train_sft_cot_v1.json"
DEFAULT_ACCEPT_FILE = "data/clean_data/cot_source_deepseek_best_v1_accept_all.json"
DEFAULT_REJECT_FILE = "data/clean_data/cot_source_deepseek_best_v1_reject_all.json"
DEFAULT_REPORT_FILE = "data/clean_data/cot_report_v1.json"
REQUIRED_TRAIN_FIELDS = (
    "id",
    "question",
    "answer",
    "instruction",
    "answer_type",
    "cot_response",
    "cot_answer_marker",
    "cot_source_id",
)
MAX_FAILURE_SAMPLE = 20


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError(f"{path} 顶层结构必须是 list")
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise TypeError(f"{path} 第 {index} 条记录不是 dict")
    return data


def load_json_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"{path} 顶层结构必须是 dict")
    return data


def _index_unique_rows(rows: list[dict[str, Any]], key: str, name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for index, row in enumerate(rows):
        if key not in row:
            raise KeyError(f"{name} 第 {index} 条缺少字段：{key}")
        row_id = str(row[key])
        if row_id in indexed:
            duplicates.append(row_id)
        indexed[row_id] = row
    if duplicates:
        raise ValueError(f"{name} 存在重复 {key}：{sorted(set(duplicates))[:MAX_FAILURE_SAMPLE]}")
    return indexed


def _record_failure(failures: list[dict[str, Any]], row_id: str, reason: str) -> int:
    if len(failures) < MAX_FAILURE_SAMPLE:
        failures.append({"id": row_id, "reason": reason})
    return 1


def audit_cot_data(
    train_file: Path,
    accept_file: Path,
    reject_file: Path,
    report_file: Path,
) -> dict[str, Any]:
    train_rows = load_json_list(train_file)
    accept_rows = load_json_list(accept_file)
    reject_rows = load_json_list(reject_file)
    report = load_json_dict(report_file)

    accept_by_id = _index_unique_rows(accept_rows, "id", "accept 文件")
    reject_by_id = _index_unique_rows(reject_rows, "id", "reject 文件")

    failures: list[dict[str, Any]] = []
    failure_count = 0
    seen_train_ids: set[str] = set()
    duplicate_train_ids: set[str] = set()
    accept_reject_conflict_ids: set[str] = set()

    for row_index, row in enumerate(train_rows):
        row_id = str(row["id"]) if "id" in row else f"row_index={row_index}"
        missing_fields = [field for field in REQUIRED_TRAIN_FIELDS if field not in row]
        if missing_fields:
            failure_count += _record_failure(failures, row_id, f"训练样本缺少字段：{','.join(missing_fields)}")
            continue

        if row_id in seen_train_ids:
            duplicate_train_ids.add(row_id)
        seen_train_ids.add(row_id)

        cot_response = row["cot_response"]
        marker = row["cot_answer_marker"]
        source_id = str(row["cot_source_id"])
        if not isinstance(cot_response, str) or not cot_response.strip():
            failure_count += _record_failure(failures, row_id, "cot_response 必须是非空字符串")
            continue
        if not isinstance(marker, str) or not marker:
            failure_count += _record_failure(failures, row_id, "cot_answer_marker 必须是非空字符串")
            continue
        if marker not in cot_response:
            failure_count += _record_failure(failures, row_id, "cot_response 缺少最终答案标记")
            continue

        if source_id not in accept_by_id:
            failure_count += _record_failure(failures, row_id, f"cot_source_id 未进入 accept 集：{source_id}")
            continue
        if source_id in reject_by_id:
            accept_reject_conflict_ids.add(source_id)

        # 审计只关心标记后的最终答案，不解析 CoT 推理正文。
        try:
            extracted = extract_final_answer(cot_response)
            if not compare_answers(extracted.answer, row["answer"]):
                failure_count += _record_failure(
                    failures,
                    row_id,
                    f"标记后答案与 answer 不一致：{extracted.answer} != {row['answer']}",
                )
        except (TypeError, ValueError) as exc:
            failure_count += _record_failure(failures, row_id, str(exc))

    if duplicate_train_ids:
        failure_count += len(duplicate_train_ids)
        for row_id in sorted(duplicate_train_ids)[:MAX_FAILURE_SAMPLE]:
            if len(failures) < MAX_FAILURE_SAMPLE:
                failures.append({"id": row_id, "reason": "训练集 id 重复"})

    return {
        "train_file": str(train_file),
        "accept_file": str(accept_file),
        "reject_file": str(reject_file),
        "report_file": str(report_file),
        "train_count": len(train_rows),
        "accept_count": len(accept_rows),
        "reject_count": len(reject_rows),
        "unique_train_id_count": len(seen_train_ids),
        "failure_count": failure_count,
        "failure_sample": failures,
        "accept_reject_conflict_count": len(accept_reject_conflict_ids),
        "accept_reject_conflict_ids": sorted(accept_reject_conflict_ids),
        "report_output_train_cot_count": report["output_train_cot_count"],
        "report_output_duplicate_id_count": report["output_duplicate_id_count"],
        "report_final_marker_validation_failure_count": report["final_marker_validation_failure_count"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE, help="CoT 训练集 JSON 路径")
    parser.add_argument("--accept-file", default=DEFAULT_ACCEPT_FILE, help="CoT accept JSON 路径")
    parser.add_argument("--reject-file", default=DEFAULT_REJECT_FILE, help="CoT reject JSON 路径")
    parser.add_argument("--report-file", default=DEFAULT_REPORT_FILE, help="CoT 构建报告 JSON 路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_cot_data(
        train_file=Path(args.train_file),
        accept_file=Path(args.accept_file),
        reject_file=Path(args.reject_file),
        report_file=Path(args.report_file),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failure_count"] != 0:
        raise ValueError(f"CoT 审计失败：{summary['failure_count']} 条")


if __name__ == "__main__":
    main()
