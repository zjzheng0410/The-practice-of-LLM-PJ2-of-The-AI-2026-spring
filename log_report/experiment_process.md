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

