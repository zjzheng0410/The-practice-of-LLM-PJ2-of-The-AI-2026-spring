from data_preprocess.feature_tags import (
    GEOMETRY_KEYWORDS,
    MULTI_STEP_KEYWORDS,
    RATE_RATIO_KEYWORDS,
    UNIT_SCALE_KEYWORDS,
)

SOURCE_CLEAN007 = "clean007"
SOURCE_CLEAN008 = "clean008"
SOURCE_CLEAN010 = "clean010"
SOURCES = (SOURCE_CLEAN007, SOURCE_CLEAN008, SOURCE_CLEAN010)

TAG_RATE_RATIO = "rate_ratio"
TAG_UNIT_SCALE = "unit_scale"
TAG_GEOMETRY = "geometry"
TAG_MULTI_STEP = "multi_step"
TAG_NO_TAG = "no_tag"
QUESTION_TAGS = (TAG_RATE_RATIO, TAG_UNIT_SCALE, TAG_GEOMETRY, TAG_MULTI_STEP, TAG_NO_TAG)

BUCKET_MULTI_STEP = "multi_step"
BUCKET_UNIT_SCALE = "unit_scale"
BUCKET_DEFAULT = "default"
SELECTOR_BUCKETS = (BUCKET_MULTI_STEP, BUCKET_UNIT_SCALE, BUCKET_DEFAULT)

SELECTION_RULE = "multi_step -> clean010; else unit_scale -> clean008; else clean007"


def _has_any_keyword(question: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in question for keyword in keywords)


def _require_question(question: str) -> str:
    if not isinstance(question, str):
        raise TypeError("question 必须是字符串")
    text = question.strip()
    if not text:
        raise ValueError("question 必须是非空字符串")
    return text


def _require_question_tags(question_tags: list[str]) -> None:
    if not isinstance(question_tags, list):
        raise TypeError("question_tags 必须是 list[str]")
    if not question_tags:
        raise ValueError("question_tags 不能为空")
    unknown_tags = [tag for tag in question_tags if tag not in QUESTION_TAGS]
    if unknown_tags:
        raise ValueError(f"question_tags 存在未知标签：{unknown_tags}")


def build_question_tags(question: str) -> list[str]:
    text = _require_question(question)
    tags: list[str] = []
    if _has_any_keyword(text, RATE_RATIO_KEYWORDS):
        tags.append(TAG_RATE_RATIO)
    if _has_any_keyword(text, UNIT_SCALE_KEYWORDS):
        tags.append(TAG_UNIT_SCALE)
    if _has_any_keyword(text, GEOMETRY_KEYWORDS):
        tags.append(TAG_GEOMETRY)
    if _has_any_keyword(text, MULTI_STEP_KEYWORDS):
        tags.append(TAG_MULTI_STEP)
    if not tags:
        tags.append(TAG_NO_TAG)
    return tags


def select_source(question_tags: list[str]) -> str:
    _require_question_tags(question_tags)
    # 推理期 selector 只能依赖题面标签，禁止使用 valid 才有的 answer/answer_type。
    if TAG_MULTI_STEP in question_tags:
        return SOURCE_CLEAN010
    if TAG_UNIT_SCALE in question_tags:
        return SOURCE_CLEAN008
    return SOURCE_CLEAN007


def selector_bucket(question_tags: list[str]) -> str:
    _require_question_tags(question_tags)
    if TAG_MULTI_STEP in question_tags:
        return BUCKET_MULTI_STEP
    if TAG_UNIT_SCALE in question_tags:
        return BUCKET_UNIT_SCALE
    return BUCKET_DEFAULT

