# 实验记录表

| ID | 类型 | 日期 | 模型 | 训练数据 | 方法 | Checkpoint | Epochs | Learning Rate | LoRA r | Max Length | 推理脚本 | 提交文件 | DF Score | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base-001 | baseline | 2026-05-30 | Qwen2.5-0.5B-Instruct | train.json | 原始 baseline：LoRA SFT，直接生成答案，无额外数据清洗和后处理 | ./output/Qwen/checkpoint-3750/ | 5 | 1e-4 | 8 | 384 | infer.py | submit/submit.csv | 0.14012500000 | 首次 baseline 提交分数，用作后续实验对照。 |
| clean-001-final | data-cleaning | 2026-05-31 | Qwen2.5-0.5B-Instruct | data/clean_data/train_sft_v1.json | 清洗数据 baseline：使用 clean_data v1 训练，加载 clean-001 最后 checkpoint 生成提交并保留答案后处理 | output/clean-001/checkpoint-3410 | 5 | 1e-4 | 8 | 384 | infer.py | submit/clean_baseline.csv | 0.14025000000 | 本地 clean-001-final 验证集准确率 0.348；DF 比 base-001 提升 0.000125，截图提交时间 2026/05/31 00:16。 |

## 填写说明

| 字段 | 含义 |
| --- | --- |
| ID | 实验编号，例如 `base-001`、`clean-001`、`cot-001`。 |
| 类型 | 实验类别，例如 `baseline`、`data-cleaning`、`prompt`、`cot`、`postprocess`。 |
| 日期 | 本次实验生成提交文件或在 DataFountain 评分的日期。 |
| 模型 | 推理使用的基础模型，例如 `Qwen2.5-0.5B-Instruct`。 |
| 训练数据 | 训练时使用的数据版本，例如 `train.json` 或 `train_clean_v1.json`。 |
| 方法 | 本次实验的核心改动，只记录和上一次实验相比最关键的区别。 |
| Checkpoint | 推理时加载的 checkpoint 路径。 |
| Epochs | 训练轮数。 |
| Learning Rate | 训练学习率。 |
| LoRA r | LoRA 的秩。 |
| Max Length | 训练时的最大序列长度，超过该长度会被截断。 |
| 推理脚本 | 用来生成提交文件的脚本，例如 `infer.py`。 |
| 提交文件 | 提交到 DataFountain 的 CSV 文件名。 |
| DF Score | DataFountain 返回的分数。 |
| 备注 | 记录有用现象，例如输出格式错误、过拟合、平台反馈、是否做了后处理等。 |
