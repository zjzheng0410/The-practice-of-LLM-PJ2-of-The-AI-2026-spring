# 实验过程记录

## 2026-05-30 方案 B 工具链改造

- 新增答案工具层，统一答案规范化、类型识别、模型输出抽取和本地答案比较。
- 新增 v1 数据清洗入口，完成题面清理、空答案丢弃、冲突样本剔除和固定 seed 切分。
- 改造推理入口，支持 checkpoint、测试集、raw 输出、最大生成长度和实验编号参数。
- 新增本地验证入口，保存整体指标、分类型指标、后处理失败率、raw 结果和 wrong 结果。
- 改造训练入口，支持 CLI 参数化和轻量测试，完成基础工具链闭环。

## 2026-05-31 valid_v2 与生成式评估重构

- 新增 `data_preprocess` 包，数据清洗入口迁入包内，按清洗后的 `question` 分组切分 train/valid。
- 生成 `valid_v2` 数据文件，新增 `split_report_v2.json`，强制检查 train/valid 同题泄漏。
- 新增 `generation.py`，沉淀模型加载、消息构造、确定性生成和 JSON 数据读取能力。
- 新增 `evaluation` 包，拆分单 checkpoint 评估、多 checkpoint 排名选择、指标统计和错题归因。
- 将训练默认数据切换到 v2，取消基于 `eval_loss` 的 best checkpoint 语义。
- 将历史 `eval` 目录迁移为 `eval_result`，同步修正历史 metrics 内的 raw/wrong 输出路径。
- 更新 README 与命令模板，统一使用 `python -m data_preprocess.prepare_data` 和 `python -m evaluation.*` 入口。

## 2026-06-01 clean-008 训练数据重采样

- 新增题型标签模块，统一识别弱答案类型、比例速度工程、单位换算、几何和多步题标签。
- 新增 clean-008 增强入口，从 `train_sft_v2.json` 派生 `train_sft_clean008.json`，验证集继续固定为 `valid_sft_v2.json`。
- 新增增强报告结构，记录增强前后分布、标签命中、复制原因、同题泄漏检查和策略参数。
- 对训练集重复原始 id 做确定性重编号，并用 `original_id` 保留原始 id，保证 clean-008 输出 id 全局唯一。
- 新增增强逻辑测试，覆盖复制规则、唯一 id、泄漏检查、输出保护和重复 id 重编号。
- 生成 `train_sft_clean008.json` 与 `augment_report_clean008.json`，原始 10902 条，新增 2794 条，增强后 13696 条。
- 增强比例为 1.2563，train/valid 同题 overlap 为 0，单源样本最多复制 1 次。

## 2026-06-01 clean-010 几何增强策略

- 新增增强 policy registry，将 clean-008 与 clean-010 策略定义从增强主流程解耦。
- 将增强主流程改为读取 policy 对象，保留 clean-008 默认 alias、policy 名、复制规则和后缀兼容。
- 新增 clean-010 几何样本增强，只复制 `geometry` 标签样本，增强原因统一为 `geometry_focus`。
- 扩展增强单测，覆盖 registry、clean-008 回归、clean-010 触发规则、专属后缀和 run 级报告。
- 生成 `train_sft_clean010.json` 与 `augment_report_clean010.json`，原始 10902 条，新增 1248 条，增强后 12150 条。
- train/valid 同题 overlap 为 0，单源样本最多复制 1 次，输出重复 id 为 0，重复源 id 重编号 18 条。

## 2026-06-02 推理期三源组合

- 新增 `ensemble` 包，拆分题面 selector、raw 读取校验和组合核心逻辑。
- 重构 valid 组合评估入口，复用同一套 selector、prediction IO 与 combiner。
- 新增 test 组合提交入口，只读取三源 raw，输出 ensemble CSV 与 provenance JSONL。
- 新增组合单测，覆盖字段校验、id 对齐、选择规则、答案字段显式指定和输出覆盖保护。
- valid_v2 组合复现 `374/1000 = 0.374`，单模型基准为 clean007 `358/1000`、clean008 `344/1000`、clean010 `348/1000`。

## 2026-06-03 CoT SFT 单模型接入

- 新增 `prompting` 包，统一管理 direct/cot profile、训练目标字段、最终答案标记和默认生成长度。
- 改造答案抽取逻辑，优先读取 `最终答案：` 后的最终答案，避免 CoT 正文中间数字干扰。
- 改造训练入口，新增 `--target-field`，CoT SFT 固定读取 `cot_response`，支持 `--valid-file ""` 关闭 Trainer eval dataset。
- 改造推理和生成式评估入口，新增 `--prompt-profile`，raw、metrics 与 ranking 记录 profile provenance。
- 新增 CoT 数据审计入口，只审计现有 accepted CoT 训练集，不生成新数据，不补齐 reject 样本。
- 本次 CoT 数据微调/标注数据由调用 DeepSeek API 完成，本阶段直接复用现有 CoT accepted 数据进行单模型验证。
- 本阶段不修改 ensemble、selector 和组合推理流程。

## 2026-06-03 单进程 Batch 推理改造

- 新增批量生成封装，统一处理 chat template、left padding、attention mask、generate 和批量 decode。
- 保留单条生成接口，并改为复用批量接口，避免单条与批量生成逻辑分叉。
- 改造推理入口，新增 `--batch-size`，按连续 batch 调度样本，默认 batch size 为 8。
- 保持逐条答案抽取、CSV 输出、raw JSONL 字段和样本顺序不变，每个 batch 写完后刷新输出。
- 保持 `infer.py` 默认 checkpoint 和默认 `direct` profile 不变，CoT checkpoint2 仅写入运行命令记录。

## 2026-06-03 CoT 多 checkpoint 投票

- 新增 CoT checkpoint 投票核心，按 source 顺序读取候选，基于规范化答案做多数投票和平票选择。
- 扩展 raw source 参数解析，支持可变数量 `name=raw.jsonl` 输入，并保持 source 顺序作为投票优先级。
- 新增 valid 投票评估入口，输出 metrics、raw provenance 和 wrong 结果。
- 新增 test 投票提交入口，只读取 checkpoint raw，输出无表头 CSV 与 provenance JSONL，并保护已有输出。
- 新增投票、source 解析和投票评估统计测试，覆盖无效候选跳过、全部无效 primary 回退和 primary 对比统计。
- valid_v2 top4 投票复现 `577/1000 = 0.577`，相比 primary `cot2000` 的 `544/1000 = 0.544` 增加 33 题。
