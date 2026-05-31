# Math Solver

本项目用于小学数学题的 LoRA SFT 训练、生成式验证和测试集提交。当前默认流程使用 `valid_v2`，验证集按清洗后的题目分组切分，避免同一题目同时进入训练集和验证集。

## 环境

在 modelscope 环境中通常无需额外安装；其他环境至少需要：

```bash
pip install transformers modelscope peft swanlab tqdm
```

所有命令默认在 `Math_Solver` 目录下执行。

## 数据准备

```bash
python -m data_preprocess.prepare_data \
  --input-file data/raw_data/train.json \
  --output-dir data/clean_data \
  --version v2 \
  --valid-size 1000 \
  --seed 20260530
```

输出文件：

- `data/clean_data/train_clean_v2.json`
- `data/clean_data/train_sft_v2.json`
- `data/clean_data/valid_sft_v2.json`
- `data/clean_data/clean_report_v2.json`
- `data/clean_data/split_report_v2.json`

`split_report_v2.json` 中 `train_valid_overlap_group_count` 必须为 `0`。

## 训练

```bash
python -u qwen_ft.py \
  --train-file data/clean_data/train_sft_v2.json \
  --valid-file data/clean_data/valid_sft_v2.json \
  --output-dir output/clean-006 \
  --experiment-id clean-006 \
  --epochs 5 \
  --learning-rate 1e-4 \
  --max-length 384 \
  --lora-r 8 \
  --save-steps 1000
```

训练脚本只负责产出候选 checkpoint。`eval_loss` 仅用于观察训练状态，最终 checkpoint 由生成式评估准确率选择。

## 单 Checkpoint 评估

```bash
python -m evaluation.evaluate_checkpoint \
  --checkpoint output/clean-001/checkpoint-3410 \
  --valid-file data/clean_data/valid_sft_v2.json \
  --experiment-id clean-001-final \
  --max-new-tokens 32 \
  --output-dir eval_result \
  --split-name valid_v2
```

输出目录为 `eval_result/valid_v2/<experiment-id>/`，包含 `metrics.json`、`raw.jsonl`、`wrong.jsonl`、`wrong_analysis.json` 和 `wrong_analysis.jsonl`。

## 多 Checkpoint 评估与选择

```bash
python -m evaluation.evaluate_checkpoints \
  --experiment-dir output/clean-005 \
  --experiment-id clean-005 \
  --valid-file data/clean_data/valid_sft_v2.json \
  --max-new-tokens 32 \
  --output-dir eval_result \
  --split-name valid_v2
```

选择规则固定为：生成准确率更高优先；准确率相同时后处理失败率更低优先；仍相同时 step 更早优先。汇总文件写入 `ranking.csv`、`ranking.json` 和 `best_checkpoint.json`。

## 测试集推理

```bash
python -u infer.py \
  --checkpoint output/clean-005/checkpoint-4774 \
  --test-file data/raw_data/test.json \
  --experiment-id clean-005-final \
  --max-new-tokens 32
```

默认输出 `submit/<experiment-id>.csv` 和 `submit/raw/<experiment-id>.jsonl`。

