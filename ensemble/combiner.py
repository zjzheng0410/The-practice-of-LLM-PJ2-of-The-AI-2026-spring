from typing import Any

from ensemble.prediction_io import (
    PredictionIndex,
    SourceConfig,
    read_answer,
    source_config_map,
    validate_prediction_ids,
)
from ensemble.selector import build_question_tags, select_source, selector_bucket


def _require_sample(row: dict[str, Any], row_index: int) -> tuple[str, str]:
    for key in ("id", "question"):
        if key not in row:
            raise KeyError(f"输入样本第 {row_index} 条缺少字段：{key}")
    question = row["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"输入样本第 {row_index} 条 question 必须是非空字符串")
    return str(row["id"]), question


def combine_predictions(
    sample_rows: list[dict[str, Any]],
    predictions: dict[str, PredictionIndex],
    source_configs: list[SourceConfig],
) -> list[dict[str, Any]]:
    validate_prediction_ids(sample_rows, predictions, source_configs)
    configs_by_source = source_config_map(source_configs)
    results: list[dict[str, Any]] = []

    for row_index, row in enumerate(sample_rows):
        row_id, question = _require_sample(row, row_index)
        question_tags = build_question_tags(question)
        selected_source = select_source(question_tags)
        if selected_source not in configs_by_source:
            raise KeyError(f"selector 返回未知 source：{selected_source}")

        candidate_answers = {
            config.name: read_answer(config, predictions[config.name][row_id], row_id)
            for config in source_configs
        }
        results.append(
            {
                "id": row_id,
                "question": question,
                "answer": candidate_answers[selected_source],
                "selected_source": selected_source,
                "selector_bucket": selector_bucket(question_tags),
                "question_tags": question_tags,
                "candidate_answers": candidate_answers,
            }
        )
    return results

