import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from answer_utils import compare_answers
from ensemble.prediction_io import (
    PredictionIndex,
    SourceConfig,
    load_all_predictions,
    load_json_rows,
    parse_source_args,
    validate_prediction_ids,
)
from ensemble.voting import (
    SUPPORTED_VOTE_STRATEGY,
    VOTE_STATUS_ALL_INVALID_FALLBACK,
    require_vote_strategy,
    vote_sample,
)
from evaluation.config import DEFAULT_OUTPUT_DIR, DEFAULT_SPLIT_NAME, DEFAULT_VALID_FILE
from evaluation.metrics import safe_accuracy


VALID_ANSWER_FIELD = "pred_answer"
SUPPORTED_STRATEGY = SUPPORTED_VOTE_STRATEGY
TIE_BREAK_POLICY = "平票按 --source 输入顺序选择，全部无效回到首个 source"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _require_strategy(strategy: str) -> None:
    require_vote_strategy(strategy)


def _new_stats() -> dict[str, int]:
    return {"total": 0, "correct": 0}


def _add_stats(stats: dict[str, int], correct: bool) -> None:
    stats["total"] += 1
    if correct:
        stats["correct"] += 1


def _format_stats(stats: dict[str, int]) -> dict[str, int | float | None]:
    return {
        "total": stats["total"],
        "correct": stats["correct"],
        "accuracy": safe_accuracy(stats["correct"], stats["total"]),
    }


def _format_vote_pattern(vote_counts: dict[str, int]) -> str:
    if not vote_counts:
        return "all_invalid"
    counts = sorted(vote_counts.values(), reverse=True)
    if any(count <= 0 for count in counts):
        raise ValueError(f"vote_counts 存在非正票数：{vote_counts}")
    return "-".join(str(count) for count in counts)


def _require_valid_row(row: dict[str, Any], row_index: int) -> tuple[str, str]:
    for key in ("id", "question", "answer", "answer_type"):
        if key not in row:
            raise KeyError(f"验证集第 {row_index} 条缺少字段：{key}")
    question = row["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"验证集第 {row_index} 条 question 必须是非空字符串")
    return str(row["id"]), question


def build_vote_metrics(
    raw_records: list[dict[str, Any]],
    source_order: list[str],
    source_raw_files: dict[str, str] | None = None,
    strategy: str = SUPPORTED_STRATEGY,
    tie_break_policy: str = TIE_BREAK_POLICY,
) -> dict[str, Any]:
    _require_strategy(strategy)
    if not source_order:
        raise ValueError("source_order 不能为空")

    overall_stats = _new_stats()
    single_source_stats = {source: _new_stats() for source in source_order}
    tie_break_count = 0
    all_invalid_fallback_count = 0
    vote_pattern_counter: Counter[str] = Counter()
    selected_source_counter: Counter[str] = Counter()
    primary_source = source_order[0]
    same_as_primary = 0
    changed_vs_primary = 0
    improved_vs_primary = 0
    worsened_vs_primary = 0

    for row_index, row in enumerate(raw_records):
        for key in ("answer", "pred_answer", "correct", "selected_source", "vote_status", "tie_break_applied"):
            if key not in row:
                raise KeyError(f"投票 raw 第 {row_index} 条缺少字段：{key}")
        candidate_normalized_answers = row.get("candidate_normalized_answers")
        if not isinstance(candidate_normalized_answers, dict):
            raise TypeError(f"投票 raw 第 {row_index} 条 candidate_normalized_answers 必须是 dict")
        vote_counts = row.get("vote_counts")
        if not isinstance(vote_counts, dict):
            raise TypeError(f"投票 raw 第 {row_index} 条 vote_counts 必须是 dict")
        if not isinstance(row["correct"], bool):
            raise TypeError(f"投票 raw 第 {row_index} 条 correct 必须是 bool")

        _add_stats(overall_stats, row["correct"])
        selected_source_counter[str(row["selected_source"])] += 1
        if row["tie_break_applied"]:
            tie_break_count += 1
        if row["vote_status"] == VOTE_STATUS_ALL_INVALID_FALLBACK:
            all_invalid_fallback_count += 1
        vote_pattern_counter[_format_vote_pattern(vote_counts)] += 1

        for source in source_order:
            if source not in candidate_normalized_answers:
                raise KeyError(f"投票 raw 第 {row_index} 条缺少 source 答案：{source}")
            candidate_answer = candidate_normalized_answers[source]
            source_correct = False
            if candidate_answer is not None:
                source_correct = compare_answers(candidate_answer, row["answer"])
            _add_stats(
                single_source_stats[source],
                source_correct,
            )

        primary_answer = candidate_normalized_answers[primary_source]
        primary_correct = False
        if primary_answer is not None:
            primary_correct = compare_answers(primary_answer, row["answer"])
        if primary_answer is not None and row["pred_answer"] == primary_answer:
            same_as_primary += 1
        else:
            changed_vs_primary += 1
        if row["correct"] and not primary_correct:
            improved_vs_primary += 1
        if not row["correct"] and primary_correct:
            worsened_vs_primary += 1

    return {
        "overall": _format_stats(overall_stats),
        "single_source": {
            source: _format_stats(single_source_stats[source])
            for source in source_order
        },
        "vote_stats": {
            "tie_break_count": tie_break_count,
            "all_invalid_fallback_count": all_invalid_fallback_count,
            "vote_pattern_counts": dict(sorted(vote_pattern_counter.items())),
            "selected_source_counts": {
                source: selected_source_counter[source]
                for source in source_order
                if selected_source_counter[source] > 0
            },
        },
        "primary_comparison": {
            "primary_source": primary_source,
            "same_as_primary": same_as_primary,
            "changed_vs_primary": changed_vs_primary,
            "improved_vs_primary": improved_vs_primary,
            "worsened_vs_primary": worsened_vs_primary,
        },
        "source_raw_files": source_raw_files or {},
        "source_order": list(source_order),
        "strategy": strategy,
        "tie_break_policy": tie_break_policy,
    }


def build_checkpoint_vote_records(
    valid_rows: list[dict[str, Any]],
    predictions: dict[str, PredictionIndex],
    source_configs: list[SourceConfig],
    strategy: str = SUPPORTED_STRATEGY,
) -> list[dict[str, Any]]:
    _require_strategy(strategy)
    validate_prediction_ids(valid_rows, predictions, source_configs)

    raw_records: list[dict[str, Any]] = []
    for row_index, row in enumerate(valid_rows):
        row_id, question = _require_valid_row(row, row_index)
        records_by_source = {
            config.name: predictions[config.name][row_id]
            for config in source_configs
        }
        decision = vote_sample(
            records_by_source=records_by_source,
            source_configs=source_configs,
            answer_field=VALID_ANSWER_FIELD,
            row_id=row_id,
        )
        correct = compare_answers(decision.answer, row["answer"])
        raw_records.append(
            {
                "id": row_id,
                "question": question,
                "answer": row["answer"],
                "answer_type": row["answer_type"],
                "pred_answer": decision.answer,
                "correct": correct,
                "selected_source": decision.selected_source,
                "vote_status": decision.vote_status,
                "tie_break_applied": decision.tie_break_applied,
                "candidate_answers": decision.candidate_answers,
                "candidate_normalized_answers": decision.candidate_normalized_answers,
                "candidate_extract_status": decision.candidate_extract_status,
                "vote_counts": decision.vote_counts,
                "source_order": decision.source_order,
                "strategy": strategy,
            }
        )
    return raw_records


def evaluate_checkpoint_vote(
    valid_file: Path,
    source_configs: list[SourceConfig],
    experiment_id: str,
    output_dir: Path,
    split_name: str,
    strategy: str = SUPPORTED_STRATEGY,
) -> dict[str, Any]:
    _require_strategy(strategy)
    valid_rows = load_json_rows(valid_file)
    if not valid_rows:
        raise ValueError(f"{valid_file} 验证集为空")

    predictions = load_all_predictions(source_configs)
    raw_records = build_checkpoint_vote_records(
        valid_rows=valid_rows,
        predictions=predictions,
        source_configs=source_configs,
        strategy=strategy,
    )
    wrong_records = [row for row in raw_records if not row["correct"]]
    target_dir = Path(output_dir) / split_name / experiment_id
    metrics_path = target_dir / "metrics.json"
    raw_path = target_dir / "raw.jsonl"
    wrong_path = target_dir / "wrong.jsonl"

    source_order = [config.name for config in source_configs]
    source_raw_files = {
        config.name: str(config.raw_path)
        for config in source_configs
    }
    metrics = build_vote_metrics(
        raw_records=raw_records,
        source_order=source_order,
        source_raw_files=source_raw_files,
        strategy=strategy,
    )
    metrics.update(
        {
            "experiment_id": experiment_id,
            "valid_file": str(valid_file),
            "metrics_output": str(metrics_path),
            "raw_output": str(raw_path),
            "wrong_output": str(wrong_path),
        }
    )

    _write_json(metrics_path, metrics)
    _write_jsonl(raw_path, raw_records)
    _write_jsonl(wrong_path, wrong_records)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读评估 CoT 多 checkpoint 多数投票")
    parser.add_argument("--valid-file", default=DEFAULT_VALID_FILE, help="验证集 JSON 路径")
    parser.add_argument("--source", action="append", required=True, help="投票源，格式 name=raw.jsonl")
    parser.add_argument("--experiment-id", required=True, help="实验编号")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="评估输出目录")
    parser.add_argument("--split-name", default=DEFAULT_SPLIT_NAME, help="验证集版本目录名")
    parser.add_argument("--strategy", default=SUPPORTED_STRATEGY, help="投票策略，仅支持 majority")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_configs = parse_source_args(args.source, answer_field=VALID_ANSWER_FIELD)
    metrics = evaluate_checkpoint_vote(
        valid_file=Path(args.valid_file),
        source_configs=source_configs,
        experiment_id=args.experiment_id,
        output_dir=Path(args.output_dir),
        split_name=args.split_name,
        strategy=args.strategy,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
