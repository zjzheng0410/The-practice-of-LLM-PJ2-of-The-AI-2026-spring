import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceConfig:
    name: str
    raw_path: Path
    answer_field: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("source name 必须是非空字符串")
        if not isinstance(self.answer_field, str) or not self.answer_field:
            raise ValueError(f"{self.name} answer_field 必须是非空字符串")
        object.__setattr__(self, "raw_path", Path(self.raw_path))


PredictionIndex = dict[str, dict[str, Any]]


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError(f"{path} 顶层结构必须是 list")
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise TypeError(f"{path} 第 {index} 条样本不是 dict")
    return data


def build_source_configs(
    clean007_raw: Path,
    clean008_raw: Path,
    clean010_raw: Path,
    answer_field: str,
) -> list[SourceConfig]:
    from ensemble.selector import SOURCE_CLEAN007, SOURCE_CLEAN008, SOURCE_CLEAN010

    return [
        SourceConfig(SOURCE_CLEAN007, Path(clean007_raw), answer_field),
        SourceConfig(SOURCE_CLEAN008, Path(clean008_raw), answer_field),
        SourceConfig(SOURCE_CLEAN010, Path(clean010_raw), answer_field),
    ]


def validate_source_configs(source_configs: list[SourceConfig]) -> None:
    if not source_configs:
        raise ValueError("source_configs 不能为空")
    seen_names: set[str] = set()
    for config in source_configs:
        if not isinstance(config, SourceConfig):
            raise TypeError("source_configs 必须全部是 SourceConfig")
        if config.name in seen_names:
            raise ValueError(f"source name 重复：{config.name}")
        seen_names.add(config.name)


def load_prediction_index(config: SourceConfig) -> PredictionIndex:
    records: PredictionIndex = {}
    with config.raw_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{config.name} {config.raw_path}:{line_number} 不是 dict")
            if "id" not in row:
                raise KeyError(f"{config.name} {config.raw_path}:{line_number} 缺少 id")
            if config.answer_field not in row:
                raise KeyError(
                    f"{config.name} {config.raw_path}:{line_number} 缺少 {config.answer_field}"
                )
            row_id = str(row["id"])
            if row_id in records:
                raise ValueError(f"{config.name} {config.raw_path} 存在重复 id：{row_id}")
            records[row_id] = row
    if not records:
        raise ValueError(f"{config.name} {config.raw_path} 没有可用记录")
    return records


def load_all_predictions(source_configs: list[SourceConfig]) -> dict[str, PredictionIndex]:
    validate_source_configs(source_configs)
    return {config.name: load_prediction_index(config) for config in source_configs}


def validate_prediction_ids(
    sample_rows: list[dict[str, Any]],
    predictions: dict[str, PredictionIndex],
    source_configs: list[SourceConfig],
) -> None:
    validate_source_configs(source_configs)
    sample_ids: list[str] = []
    seen_sample_ids: set[str] = set()
    for row_index, row in enumerate(sample_rows):
        if "id" not in row:
            raise KeyError(f"输入样本第 {row_index} 条缺少 id")
        row_id = str(row["id"])
        if row_id in seen_sample_ids:
            raise ValueError(f"输入样本存在重复 id：{row_id}")
        seen_sample_ids.add(row_id)
        sample_ids.append(row_id)

    sample_id_set = set(sample_ids)
    for config in source_configs:
        if config.name not in predictions:
            raise KeyError(f"缺少 source 预测：{config.name}")
        record_ids = set(predictions[config.name])
        missing_ids = sorted(sample_id_set - record_ids)
        extra_ids = sorted(record_ids - sample_id_set)
        if missing_ids:
            raise ValueError(f"{config.name} {config.raw_path} 缺少输入 id：{missing_ids[:10]}")
        if extra_ids:
            raise ValueError(f"{config.name} {config.raw_path} 存在多余 id：{extra_ids[:10]}")


def read_answer(config: SourceConfig, record: dict[str, Any], row_id: str) -> Any:
    # 答案字段必须由 SourceConfig 显式指定，禁止在 answer/pred_answer 之间猜测。
    if config.answer_field not in record:
        raise KeyError(f"{config.name} id={row_id} 缺少 {config.answer_field}")
    return record[config.answer_field]


def source_config_map(source_configs: list[SourceConfig]) -> dict[str, SourceConfig]:
    validate_source_configs(source_configs)
    return {config.name: config for config in source_configs}

