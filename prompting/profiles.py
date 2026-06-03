from dataclasses import dataclass


DIRECT_INSTRUCTION = "这是小学数学1-6年级的校内题目，无需进行分析，请直接输出数字答案，不带单位。"
COT_INSTRUCTION = "这是小学数学1-6年级的校内题目，请先给出简短推理过程，最后单独输出“最终答案：<答案>”。最终答案不带单位。"


@dataclass(frozen=True)
class PromptProfile:
    name: str
    instruction: str
    target_field: str
    answer_marker: str
    default_max_new_tokens: int


_PROFILES = {
    "direct": PromptProfile(
        name="direct",
        instruction=DIRECT_INSTRUCTION,
        target_field="answer",
        answer_marker="",
        default_max_new_tokens=32,
    ),
    "cot": PromptProfile(
        name="cot",
        instruction=COT_INSTRUCTION,
        target_field="cot_response",
        answer_marker="最终答案：",
        default_max_new_tokens=128,
    ),
}


def get_prompt_profile(name: str) -> PromptProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"未知 prompt_profile：{name}") from exc


def select_instruction(profile: PromptProfile, row_instruction: str) -> str:
    if profile.name == "direct":
        if not row_instruction:
            raise ValueError("direct profile 需要样本 instruction")
        return row_instruction
    if profile.name == "cot":
        return profile.instruction
    raise ValueError(f"未知 prompt_profile：{profile.name}")


def resolve_max_new_tokens(profile: PromptProfile, cli_value: int | None) -> int:
    value = profile.default_max_new_tokens if cli_value is None else cli_value
    if value <= 0:
        raise ValueError("--max-new-tokens 必须为正整数")
    return value
