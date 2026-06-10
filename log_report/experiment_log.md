# 实验记录表

| ID | 类型 | 日期 | 模型 | 训练数据 | 方法 | Checkpoint | Epochs | Learning Rate | LoRA r | Max Length | 推理脚本 | 提交文件 | DF Score | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base-001 | baseline | 2026-05-30 | Qwen2.5-0.5B-Instruct | train.json | 原始 baseline：LoRA SFT，直接生成答案，无额外数据清洗和后处理 | ./output/Qwen/checkpoint-3750/ | 5 | 1e-4 | 8 | 384 | infer.py | submit/submit.csv | 0.14012500000 | 首次 baseline 提交分数，用作后续实验对照。 |
| clean-001-final | data-cleaning | 2026-05-31 | Qwen2.5-0.5B-Instruct | data/clean_data/train_sft_v1.json | 清洗数据 baseline：使用 clean_data v1 训练，加载 clean-001 最后 checkpoint 生成提交并保留答案后处理 | output/clean-001/checkpoint-3410 | 5 | 1e-4 | 8 | 384 | infer.py | submit/clean_baseline.csv | 0.14025000000 | 本地 clean-001-final 验证集准确率 0.348；DF 比 base-001 提升 0.000125，截图提交时间 2026/05/31 00:16。 |
| clean-007 | data-cleaning | 2026-06-01 | Qwen2.5-0.5B-Instruct | data/clean_data/train_sft_v2.json | valid_v2 无泄漏切分 baseline 上调 max_length：相对 clean-006 仅将 max_length 从 384 改为 512 | output/clean-007/checkpoint-3410 | 5 | 1e-4 | 8 | 512 | infer.py | submit/clean-007.csv | 0.14087500000 | 本地 valid_v2 生成准确率 0.358；最佳 checkpoint 为 checkpoint-3410；截图提交时间 2026/06/01 22:35。 |
| ensemble-007008010 | ensemble | 2026-06-02 | Qwen2.5-0.5B-Instruct | train_sft_v2 + train_sft_clean008 + train_sft_clean010 | 推理期三源组合：在 clean-007、clean-008、clean-010 三个 test raw 结果间按题面 selector 选择答案 | clean-007: checkpoint-3410; clean-008: checkpoint-4280; clean-010: checkpoint-3800 | 5 | 1e-4 | 8 | 512 | infer.py + ensemble_infer.py | submit/ensemble-007008010.csv | 0.14075000000 | 本地 valid_v2 组合准确率 0.374，374/1000；单源 best 为 clean007 0.358、clean008 0.344、clean010 0.348；DF 比 clean-007 下降 0.000125，截图提交时间 2026/06/02 11:26，无备注信息。 |
| cot-001 | cot | 2026-06-03 | Qwen2.5-0.5B-Instruct | data/clean_data/train_sft_cot_v1.json | CoT SFT 单模型：复用 DeepSeek API 生成并筛选后的 accepted CoT 数据训练，推理使用 cot prompt profile 和最终答案标记抽取 | output/cot-001/checkpoint-2000 | 5 | 1e-4 | 8 | 768 | infer.py | submit/cot-001-best.csv | 0.22837500000 | 本地 valid_v2 CoT 生成准确率 0.544，544/1000；最佳 checkpoint 为 checkpoint-2000；DF 比此前最好 clean-007 提升 0.08750000000；截图提交时间 2026/06/03 17:04，无备注信息。 |
| cot-001-vote-top4 | cot-vote | 2026-06-03 | Qwen2.5-0.5B-Instruct | data/clean_data/train_sft_cot_v1.json | CoT 多 checkpoint 投票：读取 cot-001 的 checkpoint-2000/3310/3000/1000 四路 raw 结果，按规范化答案 majority 投票，平票按 source 顺序回退 | output/cot-001/checkpoint-2000; output/cot-001/checkpoint-3310; output/cot-001/checkpoint-3000; output/cot-001/checkpoint-1000 | 5 | 1e-4 | 8 | 768 | infer.py + cot_checkpoint_vote_infer.py | submit/cot-001-vote-top4.csv | 0.24437500000 | 本地 valid_v2 top4 投票准确率 0.577，577/1000；相比 primary cot2000 0.544 增加 33 题；DF 比 cot-001 单模型提升 0.01600000000；截图提交时间 2026/06/03 23:22，所在赛程 经典赛，无备注信息。 |
| cot4-clean007008010-vote | heterogeneous-vote | 2026-06-04 | Qwen2.5-0.5B-Instruct | data/clean_data/train_sft_cot_v1.json + data/clean_data/train_sft_v2.json + data/clean_data/train_sft_clean008.json + data/clean_data/train_sft_clean010.json | CoT top4 + clean 三源异质投票：直接读取 cot-001 checkpoint-2000/3310/3000/1000 四路 raw 和 clean-007/008/010 三路 raw，按规范化答案 majority 投票，平票按 source 顺序回退 | cot: output/cot-001/checkpoint-2000/3310/3000/1000; clean-007: output/clean-007/checkpoint-3410; clean-008: output/clean-008/checkpoint-4280; clean-010: output/clean-010/checkpoint-3800 | 5 | 1e-4 | 8 | CoT 768; clean 512 | infer.py + cot_checkpoint_vote_infer.py | submit/cot4-clean007008010-vote.csv | 0.25212500000 | 本地 valid_v2 七源投票准确率 0.612，612/1000；source 顺序为 cot2000/cot3310/cot3000/cot1000/clean007/clean008/clean010；相比 cot-001-vote-top4 valid 提升 35 题，DF 提升 0.00775000000；截图提交时间 2026/06/04 10:48，所在赛程 经典赛，无备注信息。 |
| dpo-cot001-best | dpo | 2026-06-08 | Qwen2.5-0.5B-Instruct | data/clean_data/train_dpo_cot001_v1.json | DPO preference fine-tuning：使用 cot-001 checkpoint-2000 作为 policy/reference 起点，以 accepted CoT 作为 chosen、cot-001 训练集错误生成作为 rejected，训练 2 epochs 后选择 valid_v2 最优 checkpoint-66 推理 | output/dpo-cot001/checkpoint-66 | 2 | 1e-6 | 8 | 384 | qwen_dpo.py + infer.py | submit/dpo-cot001-best.csv | 0.23025000000 | 本地 valid_v2 最优为 checkpoint-66，准确率 0.559，559/1000；checkpoint-132 为 0.556；DF 比 cot-001 单模型提升 0.00187500000，但低于 cot-001-vote-top4 和七源投票；test raw 中 DPO 与 cot-001 单模型 5906/8000 相同，2094 题发生改动，说明 DPO 有增益但泛化不稳定，截图提交时间 2026/06/08 22:51，所在赛程 经典赛，无备注信息。 |
| cot4-clean007008010-dpo66-vote | heterogeneous-vote + dpo | 2026-06-09 | Qwen2.5-0.5B-Instruct | data/clean_data/train_sft_cot_v1.json + data/clean_data/train_sft_v2.json + data/clean_data/train_sft_clean008.json + data/clean_data/train_sft_clean010.json + data/clean_data/train_dpo_cot001_v1.json | CoT top4 + clean 三源 + DPO 八源投票：在七源异质投票基础上追加 dpo-cot001 checkpoint-66 的 test raw 作为第八个 majority source，平票仍按 source 顺序回退 | cot: output/cot-001/checkpoint-2000/3310/3000/1000; clean-007: output/clean-007/checkpoint-3410; clean-008: output/clean-008/checkpoint-4280; clean-010: output/clean-010/checkpoint-3800; dpo: output/dpo-cot001/checkpoint-66 | 5; DPO 2 | 1e-4; DPO 1e-6 | 8 | CoT 768; clean 512; DPO 384 | infer.py + cot_checkpoint_vote_infer.py | submit/cot4-clean007008010-dpo66-vote.csv | 0.25237500000 | 本地 valid_v2 八源投票准确率 0.607，607/1000，低于七源投票 0.612；但 DataFountain 分数比七源投票 0.25212500000 提升 0.00025000000；截图提交时间 2026/06/09 23:09，所在赛程 Antique，无备注信息。 |

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
