import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from evaluation.config import DEFAULT_BASE_MODEL, DEFAULT_OUTPUT_DIR, DEFAULT_SPLIT_NAME, DEFAULT_VALID_FILE

CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
TIE_BREAK_POLICY = "accuracy 降序，postprocess_failure_rate 升序，step 升序"


def parse_checkpoint_step(checkpoint_path: Path) -> int:
    match = CHECKPOINT_PATTERN.fullmatch(checkpoint_path.name)
    if match is None:
        raise ValueError(f"checkpoint 目录名不符合 checkpoint-<step>：{checkpoint_path}")
    return int(match.group(1))


def discover_checkpoints(experiment_dir: Path) -> list[Path]:
    if not experiment_dir.is_dir():
        raise NotADirectoryError(f"实验目录不存在：{experiment_dir}")
    checkpoints = [
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and CHECKPOINT_PATTERN.fullmatch(path.name)
    ]
    if not checkpoints:
        raise ValueError(f"实验目录下没有 checkpoint-*：{experiment_dir}")
    return sorted(checkpoints, key=parse_checkpoint_step)


def _require_number(metrics: dict[str, Any], key: str) -> int | float:
    if key not in metrics:
        raise ValueError(f"metrics 缺少数值字段：{key}")
    value = metrics[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"metrics 缺少数值字段：{key}")
    return value


def _require_string(metrics: dict[str, Any], key: str) -> str:
    if key not in metrics:
        raise ValueError(f"metrics 缺少字符串字段：{key}")
    value = metrics[key]
    if not isinstance(value, str):
        raise ValueError(f"metrics 缺少字符串字段：{key}")
    return value


def build_ranking_rows(metric_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics in metric_records:
        checkpoint = str(metrics["checkpoint"])
        step = parse_checkpoint_step(Path(checkpoint))
        rows.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_name": Path(checkpoint).name,
                "step": step,
                "accuracy": _require_number(metrics, "accuracy"),
                "postprocess_failure_rate": _require_number(metrics, "postprocess_failure_rate"),
                "correct": _require_number(metrics, "correct"),
                "total": _require_number(metrics, "total"),
                "prompt_profile": _require_string(metrics, "prompt_profile"),
                "metrics_file": str(metrics["metrics_output"]),
            }
        )
    return sorted(
        rows,
        key=lambda row: (-float(row["accuracy"]), float(row["postprocess_failure_rate"]), int(row["step"])),
    )


def build_best_checkpoint(
    ranking_rows: list[dict[str, Any]],
    experiment_id: str,
    valid_file: str,
    ranking_file: Path,
    prompt_profile: str,
) -> dict[str, Any]:
    if not ranking_rows:
        raise ValueError("ranking 为空，无法选择 best checkpoint")
    selected = ranking_rows[0]
    return {
        "experiment_id": experiment_id,
        "valid_file": valid_file,
        "prompt_profile": prompt_profile,
        "selection_metric": "valid_v2_generation_accuracy",
        "selected_checkpoint": selected["checkpoint"],
        "selected_step": selected["step"],
        "accuracy": selected["accuracy"],
        "correct": selected["correct"],
        "total": selected["total"],
        "tie_break_policy": TIE_BREAK_POLICY,
        "all_ranking_file": str(ranking_file),
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_ranking_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "checkpoint_name",
        "step",
        "accuracy",
        "postprocess_failure_rate",
        "correct",
        "total",
        "prompt_profile",
        "metrics_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_all_checkpoints(
    base_model: str,
    experiment_dir: str,
    experiment_id: str,
    valid_file: str,
    output_dir: str,
    max_new_tokens: int | None,
    split_name: str = DEFAULT_SPLIT_NAME,
    prompt_profile: str = "direct",
) -> dict[str, Any]:
    from evaluation.evaluate_checkpoint import evaluate_checkpoint

    checkpoints = discover_checkpoints(Path(experiment_dir))
    summary_dir = Path(output_dir) / split_name / experiment_id
    metric_records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        metric_records.append(
            evaluate_checkpoint(
                base_model=base_model,
                checkpoint=str(checkpoint),
                valid_file=valid_file,
                experiment_id=experiment_id,
                output_dir=output_dir,
                max_new_tokens=max_new_tokens,
                split_name=split_name,
                result_dir=summary_dir / checkpoint.name,
                prompt_profile=prompt_profile,
            )
        )

    ranking_rows = build_ranking_rows(metric_records)
    ranking_csv_path = summary_dir / "ranking.csv"
    ranking_json_path = summary_dir / "ranking.json"
    best_checkpoint_path = summary_dir / "best_checkpoint.json"
    best_checkpoint = build_best_checkpoint(
        ranking_rows=ranking_rows,
        experiment_id=experiment_id,
        valid_file=valid_file,
        ranking_file=ranking_json_path,
        prompt_profile=prompt_profile,
    )

    _write_ranking_csv(ranking_csv_path, ranking_rows)
    _write_json(ranking_json_path, ranking_rows)
    _write_json(best_checkpoint_path, best_checkpoint)
    return {
        "ranking_csv": str(ranking_csv_path),
        "ranking_json": str(ranking_json_path),
        "best_checkpoint": str(best_checkpoint_path),
        "selected_checkpoint": best_checkpoint["selected_checkpoint"],
        "prompt_profile": prompt_profile,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="基础模型路径")
    parser.add_argument("--experiment-dir", required=True, help="包含 checkpoint-* 的实验目录")
    parser.add_argument("--experiment-id", required=True, help="实验编号")
    parser.add_argument("--valid-file", default=DEFAULT_VALID_FILE, help="验证集 JSON 路径")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="最大生成 token 数")
    parser.add_argument("--prompt-profile", default="direct", help="提示词 profile：direct 或 cot")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="评估输出目录")
    parser.add_argument("--split-name", default=DEFAULT_SPLIT_NAME, help="验证集版本目录名")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_all_checkpoints(
        base_model=args.base_model,
        experiment_dir=args.experiment_dir,
        experiment_id=args.experiment_id,
        valid_file=args.valid_file,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        split_name=args.split_name,
        prompt_profile=args.prompt_profile,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
