# TP-BERTa 使用指南

## 快速开始

### 1. 设置环境变量（可选，推荐）

如果你想把 tp-berta 放在其他位置，可以设置环境变量：

```bash
# 设置 tp-berta 源代码路径
export TPBERTA_ROOT=/path/to/tp-berta

# 设置预训练模型路径（可选）
export TPBERTA_PRETRAIN_DIR=/path/to/tp-berta/checkpoints/tp-joint

# 设置基础 RoBERTa 模型路径（可选）
export TPBERTA_BASE_MODEL_DIR=/path/to/roberta-base
```

**如果不设置**，代码会使用默认路径：`../tp-berta/`（相对于 cmds 目录）

### 2. 下载预训练模型（只需一次）

```bash
cd aida
./tpberta_download.sh
```

这会下载模型到 `../tp-berta/checkpoints/tp-joint/`（约 938MB）

### 3. 运行训练

**统一脚本**：`tpberta_train.py`（合并了原来的两个脚本）

```bash
# 方式 1：完整微调（训练整个模型，encoder + head）
python cmds/tpberta_train.py \
    --data_dir data/dfs-flatten-table/avito-ad-ctr \
    --result_dir ./tpberta_outputs

# 方式 2：冻结 encoder（只训练 head，推荐小数据集）
python cmds/tpberta_train.py \
    --data_dir data/dfs-flatten-table/avito-ad-ctr \
    --result_dir ./tpberta_outputs \
    --freeze_encoder
```

## 代码功能

`tpberta_train.py` 会自动完成：

1. **数据转换**：将你的 TableData（train.csv, val.csv, test.csv）转换为 TP-BERTa 格式
2. **特征映射**：自动生成 `feature_names.json`（特征名标准化，如 "EducationField" → "education field"）
3. **模型加载**：从 `tp-berta/checkpoints/tp-joint/` 加载预训练模型
4. **微调训练**：在你的数据上微调（200 epochs，early stop 50）
5. **评估**：在测试集上评估并保存结果

## 数据格式要求

你的数据目录应包含：
```
data/dfs-flatten-table/avito-ad-ctr/
├── train.csv          # 训练集
├── val.csv            # 验证集
├── test.csv           # 测试集
└── target_col.txt     # 第一行：目标列名，第二行：任务类型（BINARY_CLASSIFICATION/REGRESSION）
```

## 主要参数

- `--data_dir`: 数据目录（必需）
- `--result_dir`: 结果输出目录（默认：`./tpberta_outputs`）
- `--max_epochs`: 最大训练轮数（默认：200）
- `--early_stop`: 早停耐心值（默认：50）
- `--batch_size`: 批次大小（默认：64）
- `--freeze_encoder`: 冻结 encoder，只训练 head（默认：False，训练整个模型）
- `--lr`: 学习率（默认：自动设置，1e-3 如果冻结，1e-5 如果完整微调）
- `--pretrain_dir`: 预训练模型路径（可选，默认自动查找）

## 输出结果

训练完成后，结果保存在 `result_dir/{dataset_name}/results.json`：
- `best_val_metric`: 最佳验证集指标
- `final_test_metric`: 最终测试集指标
- `metric_key`: 使用的指标（roc_auc 或 rmse）
- `train_losses`, `val_metrics`, `test_metrics`: 训练历史

## 注意事项

- **模型大小**：预训练模型约 938MB，需要足够磁盘空间
- **GPU 内存**：建议至少 16GB GPU 内存
- **训练时间**：比小模型（MLP/ResNet）需要更长时间
- **自动生成**：`feature_names.json` 会自动生成，无需手动创建

## TP-BERTa 模型结构

**TP-BERTa 本质是 embedding 生成器**，prediction head 可以替换！

### 模型组成

1. **TP-BERTa 核心（固定，预训练）**：
   - `TPBertaEmbeddings`: RMT tokenization（数值特征编码）
   - `IntraFeatureAttention`: IFA（特征内注意力）
   - `RobertaEncoder`: Transformer encoder
   - **输出**：`[batch, seq_len, hidden_size]` embeddings

2. **Prediction Head（可替换）**：
   - **当前用的**：`TPBertaHead`（简单 MLP）
     - 取 `[CLS]` token → Linear → Tanh → Dropout → Linear
   - **你可以替换**：在 `tpberta_train.py` 中修改 `model.classifier`

### 训练策略

**重要概念澄清**：

1. **预训练（Pre-training）**：在大量数据上训练 TP-BERTa，得到预训练权重（你下载的 `tp-joint.tar.gz`）
2. **Fine-tuning（微调）**：在预训练权重的基础上，在你的数据集上继续训练

**两种 fine-tuning 策略**（通过 `--freeze_encoder` 参数控制）：

1. **完整微调**（默认，`--freeze_encoder False`）：
   - 📍 **来源**：TP-BERTa 官方代码库的 fine-tuning 脚本
     - 文件：`tp-berta/scripts/finetune/default/run_default_config_tpberta.py`
     - 优化器：`make_tpberta_optimizer` 优化**所有模型参数**（包括 encoder 和 head）
   - ✅ 使用预训练权重作为**初始化**
   - ✅ 在 fine-tuning 时**继续训练整个模型**（encoder + head）
   - ✅ 使用**很小的学习率** `1e-5`（自动设置）来微调 encoder
   - 📝 这是 TP-BERTa 论文和官方代码库的方式
   - **推荐**：大数据集（> 100K）或复现论文

2. **冻结 encoder**（`--freeze_encoder`）：
   - 📍 **来源**：我们新增的方式（不在原始代码库中）
   - ✅ 使用预训练权重，但**完全冻结 encoder**（`param.requires_grad = False`）
   - ✅ 只训练 prediction head
   - ✅ 使用**较大的学习率** `1e-3`（自动设置，因为只训练 head）
   - **推荐**：小数据集（< 10K）或快速实验

**使用示例**：

```bash
# 完整微调（训练整个模型）
python cmds/tpberta_train.py \
    --data_dir data/dfs-flatten-table/avito-ad-ctr \
    --result_dir ./tpberta_outputs

# 冻结 encoder（只训练 head）
python cmds/tpberta_train.py \
    --data_dir data/dfs-flatten-table/avito-ad-ctr \
    --result_dir ./tpberta_outputs \
    --freeze_encoder
```

**工作原理**：
1. 加载你下载的预训练权重（`tp-joint.tar.gz`）作为初始化
2. **如果 `--freeze_encoder`**：冻结 encoder，只训练 head（更快，更稳定）
3. **如果未设置 `--freeze_encoder`**：继续训练整个模型，但用很小学习率 `1e-5`（可能获得更好性能）

## 与现有 baseline 对比

| 方法 | 脚本 | 说明 |
|------|------|------|
| LightGBM/CatBoost | `ml_baseline.py` | 传统 ML |
| MLP/ResNet/FT-Transformer | `aida_optuna.py` | 深度学习 + 超参数搜索 |
| **TP-BERTa (full fine-tuning)** | `tpberta_train.py` | **训练整个模型（encoder + head），lr=1e-5** |
| **TP-BERTa (frozen encoder)** | `tpberta_train.py --freeze_encoder` | **冻结 encoder，只训练 head，lr=1e-3（推荐小数据集）** |
