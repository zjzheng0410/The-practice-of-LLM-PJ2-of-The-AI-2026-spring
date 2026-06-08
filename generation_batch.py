from dataclasses import dataclass
from typing import Any, Iterator

from prompting.profiles import PromptProfile, select_instruction


DEFAULT_GENERATION_BATCH_SIZE = 32


@dataclass(frozen=True)
class BatchGenerationConfig:
    batch_size: int
    max_new_tokens: int
    prompt_profile: str


def iter_batches(
    rows: list[dict[str, Any]],
    batch_size: int,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    for start_index in range(0, len(rows), batch_size):
        yield start_index, rows[start_index : start_index + batch_size]


def build_messages_batch(
    rows: list[dict[str, Any]],
    profile: PromptProfile,
    required_fields: tuple[str, ...],
    start_index: int = 0,
) -> list[list[dict[str, str]]]:
    from generation import build_messages

    messages_batch: list[list[dict[str, str]]] = []
    for offset, row in enumerate(rows):
        index = start_index + offset
        for key in required_fields:
            if key not in row:
                raise KeyError(f"第 {index} 条样本缺少字段：{key}")

        instruction = select_instruction(profile, str(row["instruction"]))
        messages_batch.append(build_messages(instruction, str(row["question"])))
    return messages_batch


def generate_responses_for_rows(
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    profile: PromptProfile,
    max_new_tokens: int,
    batch_size: int,
    required_fields: tuple[str, ...] = ("id", "instruction", "question"),
) -> list[tuple[dict[str, Any], str]]:
    from generation import predict_responses

    generated: list[tuple[dict[str, Any], str]] = []
    for start_index, batch_rows in iter_batches(rows, batch_size):
        messages_batch = build_messages_batch(
            rows=batch_rows,
            profile=profile,
            required_fields=required_fields,
            start_index=start_index,
        )
        raw_responses = predict_responses(
            messages_batch=messages_batch,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        if len(raw_responses) != len(batch_rows):
            raise ValueError(f"batch 生成返回数量异常：{len(raw_responses)} != {len(batch_rows)}")
        generated.extend(zip(batch_rows, raw_responses))
    return generated
