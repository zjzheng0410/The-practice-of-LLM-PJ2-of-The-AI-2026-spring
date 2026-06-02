import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ensemble.combiner import combine_predictions
from ensemble.prediction_io import build_source_configs, load_all_predictions, load_json_rows


DEFAULT_TEST_FILE = "data/raw_data/test.json"
TEST_ANSWER_FIELD = "answer"


def _resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.output and not args.experiment_id:
        raise ValueError("必须指定 --output 或 --experiment-id")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("submit") / f"{args.experiment_id}.csv"
    if not output_path.is_absolute() and output_path.parent == Path("."):
        output_path = Path("submit") / output_path

    if args.raw_output:
        raw_path = Path(args.raw_output)
    else:
        raw_stem = args.experiment_id if args.experiment_id else output_path.stem
        raw_path = Path("submit") / "raw" / f"{raw_stem}.jsonl"
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


def run_ensemble_infer(
    test_file: Path,
    clean007_raw: Path,
    clean008_raw: Path,
    clean010_raw: Path,
    output_path: Path,
    raw_output_path: Path,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    _ensure_can_write([output_path, raw_output_path], overwrite=overwrite)
    test_rows = load_json_rows(test_file)
    source_configs = build_source_configs(
        clean007_raw=clean007_raw,
        clean008_raw=clean008_raw,
        clean010_raw=clean010_raw,
        answer_field=TEST_ANSWER_FIELD,
    )
    predictions = load_all_predictions(source_configs)
    combined_rows = combine_predictions(test_rows, predictions, source_configs)
    _write_csv(output_path, combined_rows)
    _write_jsonl(raw_output_path, combined_rows)
    return combined_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并三份 test raw 预测，生成推理期组合提交")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="测试集 JSON 路径")
    parser.add_argument("--clean007-raw", required=True, help="clean-007 test raw.jsonl 路径")
    parser.add_argument("--clean008-raw", required=True, help="clean-008 test raw.jsonl 路径")
    parser.add_argument("--clean010-raw", required=True, help="clean-010 test raw.jsonl 路径")
    parser.add_argument("--output", default=None, help="输出 CSV 文件名或路径")
    parser.add_argument("--raw-output", default=None, help="组合 provenance JSONL 文件名或路径")
    parser.add_argument("--experiment-id", default=None, help="实验编号，用于自动命名输出文件")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path, raw_output_path = _resolve_output_paths(args)
    rows = run_ensemble_infer(
        test_file=Path(args.test_file),
        clean007_raw=Path(args.clean007_raw),
        clean008_raw=Path(args.clean008_raw),
        clean010_raw=Path(args.clean010_raw),
        output_path=output_path,
        raw_output_path=raw_output_path,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "total": len(rows),
                "output": str(output_path),
                "raw_output": str(raw_output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

