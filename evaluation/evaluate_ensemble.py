import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from answer_utils import compare_answers
from ensemble.combiner import combine_predictions
from ensemble.prediction_io import (
    SourceConfig,
    build_source_configs,
    load_all_predictions,
    load_json_rows,
    read_answer,
)
from ensemble.selector import QUESTION_TAGS, SELECTION_RULE, SELECTOR_BUCKETS
from evaluation.config import DEFAULT_VALID_FILE
from evaluation.metrics import safe_accuracy


DEFAULT_CLEAN007_RAW = "eval_result/valid_v2/clean-007/checkpoint-3410/raw.jsonl"
DEFAULT_CLEAN008_RAW = "eval_result/valid_v2/clean-008/checkpoint-4280/raw.jsonl"
DEFAULT_CLEAN010_RAW = "eval_result/valid_v2/clean-010/checkpoint-3800/raw.jsonl"
VALID_ANSWER_FIELD = "pred_answer"


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


def _require_valid_answer(row: dict[str, Any], row_index: int) -> None:
    if "answer" not in row:
        raise KeyError(f"验证集第 {row_index} 条缺少字段：answer")


def _evaluate_single_models(
    row_id: str,
    reference_answer: Any,
    source_configs: list[SourceConfig],
    predictions: dict[str, dict[str, dict[str, Any]]],
    single_model_stats: dict[str, dict[str, int]],
) -> None:
    # 单模型指标用于核对组合收益，全部从同一份 valid 标注重新计算。
    for config in source_configs:
        prediction = read_answer(config, predictions[config.name][row_id], row_id)
        _add_stats(single_model_stats[config.name], compare_answers(prediction, reference_answer))


def evaluate_ensemble(
    valid_file: Path,
    clean007_raw: Path,
    clean008_raw: Path,
    clean010_raw: Path,
) -> dict[str, Any]:
    valid_rows = load_json_rows(valid_file)
    source_configs = build_source_configs(
        clean007_raw=clean007_raw,
        clean008_raw=clean008_raw,
        clean010_raw=clean010_raw,
        answer_field=VALID_ANSWER_FIELD,
    )
    predictions = load_all_predictions(source_configs)
    combined_rows = combine_predictions(valid_rows, predictions, source_configs)

    overall_stats = _new_stats()
    by_source = {config.name: _new_stats() for config in source_configs}
    by_question_tag = {tag: _new_stats() for tag in QUESTION_TAGS}
    by_selector_bucket = {bucket: _new_stats() for bucket in SELECTOR_BUCKETS}
    single_model_stats = {config.name: _new_stats() for config in source_configs}
    selected_source_counter: Counter[str] = Counter()

    for row_index, (valid_row, combined_row) in enumerate(zip(valid_rows, combined_rows)):
        _require_valid_answer(valid_row, row_index)
        row_id = str(valid_row["id"])
        if row_id != combined_row["id"]:
            raise ValueError(f"组合结果顺序错位：valid id={row_id}, combined id={combined_row['id']}")

        correct = compare_answers(combined_row["answer"], valid_row["answer"])
        selected_source = combined_row["selected_source"]
        bucket = combined_row["selector_bucket"]
        _add_stats(overall_stats, correct)
        _add_stats(by_source[selected_source], correct)
        _add_stats(by_selector_bucket[bucket], correct)
        selected_source_counter[selected_source] += 1
        for tag in combined_row["question_tags"]:
            _add_stats(by_question_tag[tag], correct)

        _evaluate_single_models(
            row_id=row_id,
            reference_answer=valid_row["answer"],
            source_configs=source_configs,
            predictions=predictions,
            single_model_stats=single_model_stats,
        )

    return {
        "selection_rule": SELECTION_RULE,
        "valid_file": str(valid_file),
        "source_raw_files": {
            config.name: str(config.raw_path)
            for config in source_configs
        },
        "overall": _format_stats(overall_stats),
        "single_model": {
            source: _format_stats(stats)
            for source, stats in sorted(single_model_stats.items())
        },
        "selected_source_counts": dict(sorted(selected_source_counter.items())),
        "by_selected_source": {
            source: _format_stats(stats)
            for source, stats in sorted(by_source.items())
        },
        "by_selector_bucket": {
            bucket: _format_stats(by_selector_bucket[bucket])
            for bucket in SELECTOR_BUCKETS
        },
        "by_question_tag": {
            tag: _format_stats(by_question_tag[tag])
            for tag in QUESTION_TAGS
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读评估现成 valid raw 预测的推理期组合效果")
    parser.add_argument("--valid-file", default=DEFAULT_VALID_FILE, help="验证集 JSON 路径")
    parser.add_argument("--clean007-raw", default=DEFAULT_CLEAN007_RAW, help="clean-007 valid raw.jsonl 路径")
    parser.add_argument("--clean008-raw", default=DEFAULT_CLEAN008_RAW, help="clean-008 valid raw.jsonl 路径")
    parser.add_argument("--clean010-raw", default=DEFAULT_CLEAN010_RAW, help="clean-010 valid raw.jsonl 路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_ensemble(
        valid_file=Path(args.valid_file),
        clean007_raw=Path(args.clean007_raw),
        clean008_raw=Path(args.clean008_raw),
        clean010_raw=Path(args.clean010_raw),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

