# finetune_csv/ — CSV 驱动微调流水线

基于 **原始 CSV 文件** 的 Kronos 简洁微调流水线。与 `finetune/`（Qlib 驱动）不同，本流水线通过 **YAML 配置文件** 统一管理参数，使用 **单机顺序训练**（可选 DDP 支持），数据来源于单只股票的 CSV K 线文件，无需 Qlib 环境。

---

## 文件清单与职责

### `config_loader.py` — YAML 配置加载器

提供两层级配置类，将 `config.yaml` 文件结构化为 Python 对象。

#### `ConfigLoader` — 底层 YAML 读取器

| 方法 | 职责 |
|------|------|
| `_load_config()` | 读取 YAML 文件，调用 `_resolve_dynamic_paths` 解析路径模板 |
| `_resolve_dynamic_paths(config)` | 将 `{exp_name}` 占位符、空字符串路径自动展开为完整路径 |
| `get(key)` | 点分键访问（如 `'model_paths.base_save_path'`） |
| `get_data_config()` / `get_training_config()` / `get_model_paths()` | 按配置域分类获取 |
| `update_config(updates)` / `save_config(path)` | 运行时修改并保存配置 |

- **设计模式**：配置对象模式（Configuration Object）——YAML → 字典 → 按域解析 → 扁平化为属性。
- **路径解析流程**：
  ```
  exp_name="HK_ali_09988"
  base_path="/xxx/finetune_csv/finetuned/"
  ↓
  base_save_path = "/xxx/finetune_csv/finetuned/HK_ali_09988"
  finetuned_tokenizer = "/xxx/finetune_csv/finetuned/HK_ali_09988/tokenizer/best_model"
  ```

#### `CustomFinetuneConfig` — 高层配置外观

从 `ConfigLoader` 读取所有域后，将参数展开为 `self.xxx` 属性，并计算完整保存路径。

| 配置域 | 关键属性 |
|--------|----------|
| 数据 | `data_path`, `lookback_window`, `predict_window`, `clip`, `train/val/test_ratio` |
| 训练 | `tokenizer_epochs`, `basemodel_epochs`, `batch_size`, `tokenizer/predictor_learning_rate` |
| 优化器 | `adam_beta1`, `adam_beta2`, `adam_weight_decay`, `accumulation_steps` |
| 模型路径 | `pretrained_tokenizer_path`, `pretrained_predictor_path`, `finetuned_tokenizer_path` |
| 实验 | `experiment_name`, `train_tokenizer`, `train_basemodel`, `skip_existing`, `use_comet` |
| 设备 | `use_cuda`, `device_id`, `use_ddp`, `ddp_backend` |

- **设计模式**：外观模式（Facade）——将 YAML 四层级结构统一为扁平的配置对象。
- **提供方法**：`get_tokenizer_config()` / `get_basemodel_config()` — 分别为 tokenizer 和 predictor 训练脚本提供定制化配置字典。

---

### `finetune_base_model.py` — 底层模型训练 + 数据集

#### `CustomKlineDataset` — 单股票 CSV 滑动窗口数据集

- **数据源**：单只股票的 CSV 文件（`data_path`），必须包含 `timestamps` 列和 `open/high/low/close/volume/amount` 列。
- **时间划分**：按 `train_ratio : val_ratio : test_ratio` 比例顺序切分数据集（非 Qlib 的日期区间方式）。
- **采样逻辑**：
  ```
  读取 CSV → 解析 timestamps → 排序 → 生成时间特征
      ↓
  按比例切分为 train/val/test
      ↓
  训练集: 使用确定性伪随机采样 start_idx = (idx * 9973 + epoch * 104729) % max_start
  验证集: 顺序索引（不随机）
  ```
- **设计模式**：滑动窗口（Sliding Window）+ 确定性索引（Deterministic Indexing）——训练集使用固定模运算采样，避免 `random.randint` 的开销，且保证可复现。
- **归一化**：对窗口整体做 Z-score（均值/标准差从窗口内计算），clip 到 `[-clip, clip]`。

#### `train_model()` — Predictor 训练流程

**入口**：`finetune_base_model.py` 可直接运行（`python finetune_base_model.py --config config.yaml`），或由 `train_sequential.py` 调用。

**Flow**：
```
1. 加载 tokenizer（微调后）+ predictor（预训练）
2. create_dataloaders() → train/val DataLoader
3. AdamW + OneCycleLR 调度器
4. For epoch in range(basemodel_epochs):
    ├── for batch_x, batch_x_stamp in train_loader:
    │     tokenizer.encode(batch_x, half=True) → [s1, s2]
    │     shift-by-1: token_in = [:, :-1], token_out = [:, 1:]
    │     model.forward() → logits
    │     head.compute_loss() → loss + s1_loss + s2_loss
    │     optimizer.step() → scheduler.step()
    └── 验证 + 保存 best_model
```

- **DDP 支持**：若 `dist.is_initialized()`，自动包装模型为 `DDP` 并在 all_reduce 后聚合损失。
- **日志**：`RotatingFileHandler` 滚动日志 + Rank 0 控制台输出。

#### `setup_logging()` / `create_dataloaders()`

- `setup_logging(exp_name, log_dir, rank)` — 创建滚动日志（`RotatingFileHandler`，10MB 切割，保留 5 个备份）。
- `create_dataloaders(config)` — 基于 `DistributedSampler`（若有 DDP）或普通 `DataLoader` 创建训练/验证数据加载器。

---

### `finetune_tokenizer.py` — Tokenizer 训练模块

**结构**与 `finetune_base_model.py` 对称，但负责 Tokenizer 而非 Predictor 训练。

**Key Functions**：
| 函数 | 职责 |
|------|------|
| `set_seed(seed, rank)` | 全库随机种子 |
| `get_model_size(model)` | 参数量统计 |
| `format_time(seconds)` | 时间格式化 |
| `setup_logging(exp_name, log_dir, rank)` | 日志配置 |
| `create_dataloaders(config)` | 训练/验证 DataLoader |
| `train_tokenizer(model, device, config, save_dir, logger)` | 主训练循环 |

**训练循环 Flow**：
```
for epoch in range(tokenizer_epochs):
    for batch_x, _ in train_loader:
        for accum_step in range(accumulation_steps):
            model(batch_x) → z, bsq_loss
            recon_loss = MSE(z_pre, x) + MSE(z, x)
            loss = (recon_loss + bsq_loss) / 2
            loss_scaled = loss / accumulation_steps
            loss_scaled.backward()
        ↓
        clip_grad_norm_(2.0) → optimizer.step() → scheduler.step()
    ↓
    验证循环 → all_reduce → 保存 best_model
```

**两种初始化方式**（`pre_trained_tokenizer` 控制）：
- `True`（默认）：`KronosTokenizer.from_pretrained(pretrained_tokenizer_path)`
- `False`：从 `config.json` 读取架构参数，随机初始化 Tokenizer 权重

---

### `train_sequential.py` — 顺序训练编排器

**入口**：`python finetune_csv/train_sequential.py --config configs/config_ali09988_candle-5min.yaml`

#### `SequentialTrainer` 类 — 训练流程编排

**职责**：按固定顺序执行两阶段训练，处理断点续训、目录创建、分布式初始化等编排逻辑。

**`run_training()` Flow**：
```
1. _setup_distributed()    — 按需初始化 DDP
2. _create_directories()   — 创建 tokenizer/basemodel 保存目录
3. _check_existing_models()— 检测已有模型
4.
   train_tokenizer=True?
   ├── Yes → train_tokenizer_phase()
   │     加载预训练/随机 Tokenizer → train_tokenizer() → 保存 best_model
   └── No  → 跳过
5.
   train_basemodel=True?
   ├── Yes → train_basemodel_phase()
   │     加载微调 Tokenizer + 预训练/随机 Predictor → train_model() → 保存 best_model
   └── No  → 跳过
6. 输出训练时间 + 模型路径
```

**命令行参数**：
| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径（默认 `config.yaml`） |
| `--skip-tokenizer` | 跳过 tokenizer 训练阶段 |
| `--skip-basemodel` | 跳过 basemodel 训练阶段 |
| `--skip-existing` | 若已有模型则跳过训练（断点续训） |

**设计模式**：
- **模板方法模式（Template Method）**：`run_training()` 定义两阶段骨架，`train_tokenizer_phase()` / `train_basemodel_phase()` 是具体步骤。
- **命令模式（Command）**：CLI 参数控制跳过阶段。
- **沙盒模式（Sandbox）**：`skip_existing` 检测磁盘文件，避免重复训练。

---

### `configs/config_ali09988_candle-5min.yaml` — 配置模板

**结构**：
```yaml
data:         # 数据参数
training:     # 训练参数（epochs/batch_size/LR/优化器）
model_paths:  # 模型路径（支持占位符展开）
experiment:   # 实验控制（是否训练各阶段/是否跳过已存在/随机初始化）
device:       # CUDA 设备与 DDP 配置
```

---

## 设计模式总结

| 模式 | 位置 | 说明 |
|------|------|------|
| **配置对象模式（Configuration Object）** | `config_loader.py` | YAML → 层级解析 → 扁平属性 |
| **外观模式（Facade）** | `CustomFinetuneConfig` | 将四层 YAML 结构统一暴露为单一对象 |
| **路径占位符解析（Path Template Resolution）** | `ConfigLoader._resolve_dynamic_paths` | `{exp_name}` 自动展开 |
| **模板方法模式（Template Method）** | `SequentialTrainer.run_training()` | 两阶段顺序骨架 |
| **命令模式（Command）** | `train_sequential.py` CLI | `--skip-*` 参数控制阶段启停 |
| **滑动窗口（Sliding Window）** | `CustomKlineDataset` | 固定长度窗口从上到下遍历 |
| **确定性索引（Deterministic Indexing）** | `CustomKlineDataset.__getitem__` | 训练集使用模运算代替随机采样 |
| **滚动日志（Rotating File Handler）** | `setup_logging()` | 10MB 切割 + 5 备份 |
| **断点续训（Checkpoint Resume）** | `SequentialTrainer._check_existing_models` | 检测磁盘已有模型自动跳过 |
| **随机初始化模式（Random Init Switch）** | 两个训练模块 | `pre_trained_tokenizer/predictor` 控制 |

---

## 关键集成点

| 模块 | 输入 | 输出 | 下游 |
|------|------|------|------|
| `config.yaml` | 用户编辑 | `CustomFinetuneConfig` | 所有训练模块 |
| `finetune_tokenizer.py` | 预训练 Tokenizer + CSV | 微调 Tokenizer | `finetune_base_model.py` |
| `finetune_base_model.py` | 微调 Tokenizer + 预训练 Predictor + CSV | 微调 Predictor | `examples/` 推理 |
| `train_sequential.py` | YAML 配置 | 两端微调权重 | 用户输出路径 |

### 与 `finetune/`（Qlib 版）的关键区别

| 维度 | `finetune/` | `finetune_csv/` |
|------|------------|-----------------|
| 数据源 | Qlib 金融数据库（多股票） | 单股票 CSV 文件 |
| 数据切分 | 按日期区间（train/val/test_time_range） | 按比例（train/val/test_ratio） |
| 配置方式 | Python `Config` 类硬编码 | YAML 文件动态加载 |
| 采样方式 | 随机采样（`self.indices` 池） | 确定性索引（模运算） |
| 训练模式 | `torchrun` DDP 强制 | 单机优先，可选 DDP |
| 回测 | `qlib_test.py`（Qlib 框架） | 无内置回测（需自行接入 `examples/`） |
| 日志 | Comet ML | 本地滚动文件日志 |
| 适用场景 | A 股全市场多股票微调 + 回测 | 单只股票（含港股/美股）快速微调 |

### 数据流约束

1. **两阶段顺序严格**：Tokenizer 必须先于 Predictor 训练。
2. **CSV 必须包含**：`timestamps`（时间戳）、`open`、`high`、`low`、`close`、`volume`、`amount` 列。
3. **YAML 路径支持两种写法**：
   - 直接指定绝对路径
   - 留空字符串/使用 `{exp_name}` 占位符（自动展开为 `base_path/exp_name/...`）
4. **随机初始化**：设置 `pre_trained_tokenizer: false` 或 `pre_trained_predictor: false` 可从零训练（不加载 HuggingFace 权重）。
