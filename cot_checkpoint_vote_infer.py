import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ensemble.prediction_io import (
    SourceConfig,
    load_all_predictions,
    load_json_rows,
    parse_source_args,
    validate_prediction_ids,
)
from ensemble.voting import SUPPORTED_VOTE_STRATEGY, require_vote_strategy, vote_sample


DEFAULT_EXPERIMENT_ID = "cot-001-vote-top4"
DEFAULT_TEST_FILE = "data/raw_data/test.json"
TEST_ANSWER_FIELD = "answer"
SUPPORTED_STRATEGY = SUPPORTED_VOTE_STRATEGY


def _resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    experiment_id = args.experiment_id or DEFAULT_EXPERIMENT_ID
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("submit") / f"{experiment_id}.csv"
    if not output_path.is_absolute() and output_path.parent == Path("."):
        output_path = Path("submit") / output_path

    if args.raw_output:
        raw_path = Path(args.raw_output)
    else:
        raw_path = Path("submit") / "raw" / f"{experiment_id}.jsonl"
    if not raw_path.is_absolute() and raw_path.parent == Path("."):
        raw_path = Path("submit") / "raw" / raw_path
    return output_path, raw_path


def _ensure_can_write(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("输出文件已存在，请使用 --overwrite：" + ", ".join(existing))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow([row["id"], row["answer"]])


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _require_test_row(row: dict[str, Any], row_index: int) -> tuple[str, str]:
    for key in ("id", "question"):
        if key not in row:
            raise KeyError(f"测试集第 {row_index} 条缺少字段：{key}")
    question = row["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"测试集第 {row_index} 条 question 必须是非空字符串")
    return str(row["id"]), question


def build_checkpoint_vote_submission_rows(
    test_rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, dict[str, Any]]],
    source_configs: list[SourceConfig],
    strategy: str = SUPPORTED_STRATEGY,
) -> list[dict[str, Any]]:
    require_vote_strategy(strategy)
    validate_prediction_ids(test_rows, predictions, source_configs)

    submission_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(test_rows):
        row_id, question = _require_test_row(row, row_index)
        records_by_source = {
            config.name: predictions[config.name][row_id]
            for config in source_configs
        }
        decision = vote_sample(
            records_by_source=records_by_source,
            source_configs=source_configs,
            answer_field=TEST_ANSWER_FIELD,
            row_id=row_id,
        )
        submission_rows.append(
            {
                "id": row_id,
                "question": question,
                "answer": decision.answer,
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
    return submission_rows


def run_checkpoint_vote_infer(
    test_file: Path,
    source_configs: list[SourceConfig],
    output_path: Path,
    raw_output_path: Path,
    strategy: str = SUPPORTED_STRATEGY,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    require_vote_strategy(strategy)
    _ensure_can_write([output_path, raw_output_path], overwrite=overwrite)
    test_rows = load_json_rows(test_file)
    if not test_rows:
        raise ValueError(f"{test_file} 测试集为空")
    predictions = load_all_predictions(source_configs)
    rows = build_checkpoint_vote_submission_rows(
        test_rows=test_rows,
        predictions=predictions,
        source_configs=source_configs,
        strategy=strategy,
    )
    _write_csv(output_path, rows)
    _write_jsonl(raw_output_path, rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并 CoT 多 checkpoint test raw，生成多数投票提交")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="测试集 JSON 路径")
    parser.add_argument("--source", action="append", required=True, help="投票源，格式 name=raw.jsonl")
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID, help="实验编号")
    parser.add_argument("--output", default=None, help="输出 CSV 文件名或路径")
    parser.add_argument("--raw-output", default=None, help="投票 provenance JSONL 文件名或路径")
    parser.add_argument("--strategy", default=SUPPORTED_STRATEGY, help="投票策略，仅支持 majority")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_configs = parse_source_args(args.source, answer_field=TEST_ANSWER_FIELD)
    output_path, raw_output_path = _resolve_output_paths(args)
    rows = run_checkpoint_vote_infer(
        test_file=Path(args.test_file),
        source_configs=source_configs,
        output_path=output_path,
        raw_output_path=raw_output_path,
        strategy=args.strategy,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "total": len(rows),
                "output": str(output_path),
                "raw_output": str(raw_output_path),
                "source_order": [config.name for config in source_configs],
                "strategy": args.strategy,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
