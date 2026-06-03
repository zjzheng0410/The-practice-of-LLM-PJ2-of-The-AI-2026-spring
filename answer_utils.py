import re
from dataclasses import asdict, dataclass
from typing import Any


YES_NO_ANSWERS = {"够", "不够", "可以", "不能", "能", "不能够", "是", "不是"}
# 短未知只允许非常短的模型输出；较长文本无规则命中时必须报错，便于定位后处理失败。
SHORT_UNKNOWN_LIMIT = 20

_INTEGER_RE = re.compile(r"^-?\d+$")
_DECIMAL_RE = re.compile(r"^-?\d+\.\d+$")
_FRACTION_RE = re.compile(r"^-?\d+/\d+$")
_MIXED_FRACTION_RE = re.compile(r"^-?\d+_\d+/\d+$")
_PERCENT_RE = re.compile(r"^-?\d+(?:\.\d+)?%$")

_ATOM_PATTERN = r"(?:-?\d+_\d+/\d+|-?\d+/\d+|-?\d+(?:\.\d+)?%?|-?\d+)"
_MULTI_EXTRACT_RE = re.compile(rf"(?<![\w/])({_ATOM_PATTERN}(?:;{_ATOM_PATTERN})+)(?![\w/%])")
_PERCENT_EXTRACT_RE = re.compile(r"(?<![\w/])-?\d+(?:\.\d+)?%(?![\w/%])")
_MIXED_FRACTION_EXTRACT_RE = re.compile(r"(?<![\w/])-?\d+_\d+/\d+(?![\w/%])")
_FRACTION_EXTRACT_RE = re.compile(r"(?<![\w/])-?\d+/\d+(?![\w/%])")
_NUMBER_EXTRACT_RE = re.compile(r"(?<![\w/])-?\d+(?:\.\d+)?(?![\w/%])")
_FINAL_MARKER_RE = re.compile(r"最终答案\s*[:：]")


@dataclass(frozen=True)
class ExtractionResult:
    answer: str
    answer_type: str
    status: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _strip_outer_quotes(text: str) -> str:
    quote_pairs = [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")]
    changed = True
    while changed and text:
        changed = False
        text = text.strip()
        for left, right in quote_pairs:
            if len(text) >= 2 and text.startswith(left) and text.endswith(right):
                text = text[1:-1].strip()
                changed = True
    return text


def normalize_answer(value: Any) -> str:
    if value is None:
        raise ValueError("答案为空：None")
    if isinstance(value, (list, dict, tuple, set)):
        raise TypeError(f"答案类型不支持：{type(value).__name__}")

    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    text = _strip_outer_quotes(text)
    text = text.replace("％", "%").replace("﹪", "%")
    text = text.replace("；", ";").replace("／", "/")
    text = re.sub(r"\s+", "", text)
    text = text.strip("。,.，、:：")

    # 训练集中已知存在“1分”这类带单位答案；这里仅处理明确观察到的单位，避免扩大规则范围。
    unit_match = re.fullmatch(r"(-?\d+(?:\.\d+)?)分", text)
    if unit_match:
        text = unit_match.group(1)

    if not text:
        raise ValueError("答案为空字符串")
    return text


def classify_answer(value: Any) -> str:
    text = normalize_answer(value)
    if ";" in text:
        parts = text.split(";")
        if all(parts):
            return "multi_answer"
        return "unknown"
    if text in YES_NO_ANSWERS:
        return "yes_no"
    if _PERCENT_RE.fullmatch(text):
        return "percent"
    if _MIXED_FRACTION_RE.fullmatch(text):
        return "mixed_fraction"
    if _FRACTION_RE.fullmatch(text):
        return "fraction"
    if _DECIMAL_RE.fullmatch(text):
        return "decimal"
    if _INTEGER_RE.fullmatch(text):
        return "integer"
    return "unknown"


def _normalize_model_output_text(raw_output: Any) -> str:
    if raw_output is None:
        raise ValueError("模型输出为空：None")
    if isinstance(raw_output, (list, dict, tuple, set)):
        raise TypeError(f"模型输出类型不支持：{type(raw_output).__name__}")

    text = str(raw_output).replace("\n", " ").replace("\r", " ").strip()
    text = _strip_outer_quotes(text)
    text = text.replace("％", "%").replace("﹪", "%").replace("；", ";").replace("／", "/")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _cleanup_model_output(raw_output: Any) -> str:
    text = _normalize_model_output_text(raw_output)
    # 只删除开头的固定答案前缀；不删除正文里的词，避免把解释文本误改成答案。
    prefixes = (
        r"^最终答案(?:是|为)?\s*[:：]?\s*",
        r"^答案(?:是|为)?\s*[:：]?\s*",
        r"^答\s*[:：]?\s*",
        r"^所以\s*[:：]?\s*",
        r"^因此\s*[:：]?\s*",
    )
    changed = True
    while changed:
        changed = False
        for pattern in prefixes:
            new_text = re.sub(pattern, "", text).strip()
            if new_text != text:
                text = new_text
                changed = True
    return text


def _last_match(pattern: re.Pattern[str], text: str) -> str | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    return matches[-1]


def _extract_from_text(text: str, allow_unknown_short: bool) -> ExtractionResult | None:
    if not text:
        return None

    # 多答案必须先抽取，否则“180;4”会被普通数字规则截成“180”。
    for pattern in (
        _MULTI_EXTRACT_RE,
        _PERCENT_EXTRACT_RE,
        _MIXED_FRACTION_EXTRACT_RE,
        _FRACTION_EXTRACT_RE,
        _NUMBER_EXTRACT_RE,
    ):
        candidate = _last_match(pattern, text)
        if candidate is not None:
            answer = normalize_answer(candidate)
            return ExtractionResult(answer=answer, answer_type=classify_answer(answer), status="matched")

    normalized = normalize_answer(text)
    # 判断题只接受完整短答案，避免把解释句中的“是”“能”等字误抽成最终答案。
    if normalized in YES_NO_ANSWERS:
        return ExtractionResult(answer=normalized, answer_type=classify_answer(normalized), status="matched")
    if allow_unknown_short and len(normalized) <= SHORT_UNKNOWN_LIMIT:
        return ExtractionResult(answer=normalized, answer_type=classify_answer(normalized), status="unknown_short")
    return None


def _extract_after_final_marker(text: str) -> ExtractionResult | None:
    matches = list(_FINAL_MARKER_RE.finditer(text))
    if not matches:
        return None

    marker = matches[-1]
    tail = text[marker.end() :].strip()
    try:
        result = _extract_from_text(tail, allow_unknown_short=True)
    except ValueError as exc:
        raise ValueError(f"最终答案标记后无法抽取答案：{tail[:80]}") from exc
    if result is None:
        raise ValueError(f"最终答案标记后无法抽取答案：{tail[:80]}")
    return result


def extract_final_answer(raw_output: Any) -> ExtractionResult:
    marker_text = _normalize_model_output_text(raw_output)
    marked_result = _extract_after_final_marker(marker_text)
    if marked_result is not None:
        return marked_result

    text = _cleanup_model_output(raw_output)
    if not text:
        return ExtractionResult(answer="", answer_type="unknown", status="empty")

    result = _extract_from_text(text, allow_unknown_short=True)
    if result is not None:
        return result
    raise ValueError(f"无法从模型输出中抽取答案：{text[:80]}")


def compare_answers(prediction: Any, reference: Any) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)
