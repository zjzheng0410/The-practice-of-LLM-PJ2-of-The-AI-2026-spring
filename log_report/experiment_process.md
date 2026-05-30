# 方案 B 工具链改造记录（2026-05-30）

## 改造目标

本次只完成代码层面的方案 B 闭环：数据清洗、答案后处理、本地验证、训练入口参数化、推理入口参数化和轻量测试。未启动完整训练，未生成正式 `data/*.json`，未运行 `clean-001`。

## 核心实现

- 新增 `answer_utils.py`，统一处理答案规范化、答案类型识别、模型输出抽取和本地答案比较。规则只覆盖明确格式：整数、小数、分数、混合分数、百分数、判断短答案和 `;` 分隔多答案；不做小数/分数互转，也不猜测语义等价。
- 新增 `prepare_data.py`，默认读取 `train.json`，可生成 `data/train_clean_v1.json`、`data/train_sft_v1.json`、`data/valid_sft_v1.json` 和 `data/clean_report_v1.json`。`question` 为 list 时清理引号并用中文逗号拼接；空答案丢弃；同一题目和 instruction 但答案不同的冲突样本整组丢弃。
- 改造 `infer.py`，保留 `python infer.py --output submit.csv` 的 baseline 用法，同时支持 `--checkpoint`、`--test-file`、`--raw-output`、`--max-new-tokens`、`--experiment-id`、`--no-postprocess` 和 `--base-model`。推理显式传入 `attention_mask` 与 `pad_token_id`，最终 CSV 仍为无表头 `id,answer`。
- 新增 `eval.py`，用于指定 checkpoint 在固定验证集上计算整体准确率、分类型准确率、后处理失败率、长题准确率、分数题和百分数题准确率，并保存 raw 与 wrong JSONL。
- 改造 `qwen_ft.py` 为 `main()` + CLI。默认仍使用 `train.json`、`./output/Qwen` 和 baseline 参数；传入 `--valid-file` 后启用 eval loss 和 best checkpoint 保存。

## 风险控制

- 未修改 `.gitignore`，避免覆盖当前工作区已有改动。
- 未新增三方依赖，测试仅使用标准库 `unittest`、`tempfile`、`json`。
- 输出文件默认不覆盖；`prepare_data.py` 需要显式 `--overwrite` 才能覆盖已有清洗数据。
- 训练预处理只做 token 拼接与截断检查，不包含数据清洗逻辑；若截断后没有答案 token，直接报错。

## 已完成验证

- `test/test_answer_utils.py` 覆盖答案类型识别、`87.5﹪ -> 87.5%`、`1分 -> 1`、常见输出抽取、多答案不被截断、长文本无法抽取时报错。
- `test/test_prepare_data.py` 覆盖 list 题目拼接、空答案丢弃、冲突样本整组丢弃、固定 seed 切分和输出覆盖保护。
- 已运行 `conda run -n ai_2026spring_semester python -m py_compile answer_utils.py prepare_data.py eval.py infer.py qwen_ft.py`，语法编译通过。
- 已运行 `conda run -n ai_2026spring_semester python -m unittest discover -s test`，9 个轻量测试通过。
- 已用真实 `train.json` 在 `/tmp/math_solver_prepare_data_check_20260530` 做清洗演练，不生成项目内正式数据。结果：原始 11999 条，空答案丢弃 1 条，list 题目修复 649 条，空 list 片段跳过 1 个，冲突组 48 组，冲突丢弃 96 条，清洗后 11902 条，切分为训练 10902 条、验证 1000 条。
