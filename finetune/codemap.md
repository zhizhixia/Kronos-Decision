# finetune/ — Qlib 微调流水线

基于 Qlib 数据源（A 股市场）的 Kronos 微调流水线。采用 **两阶段 DDP 分布式训练**：先微调 Tokenizer（VQ-VAE），再微调 Predictor（自回归 Transformer），最终通过 Qlib 回测框架验证效果。

---

## 文件清单与职责

### `config.py` — 全局配置中心

| 配置域 | 关键参数 | 说明 |
|--------|----------|------|
| 数据路径 | `qlib_data_path`, `instrument` | Qlib 数据目录与标的池（csi300/csi800/csi1000） |
| 时间范围 | `dataset_begin/end_time`, `train/val/test/backtest_time_range` | 数据集时间窗口，验证/测试集起始早于训练集结束以容纳 lookback_window |
| 窗口参数 | `lookback_window=90`, `predict_window=10`, `max_context=512` | 滑动窗口尺寸 |
| 特征列 | `feature_list`, `time_feature_list` | OHLCV + 成交量额 + 时间特征（分钟/小时/星期/日/月） |
| 训练超参 | `epochs=30`, `batch_size=50`, 双学习率 | tokenizer LR 2e-4, predictor LR 4e-5 |
| 优化器 | `adam_beta1=0.9`, `adam_beta2=0.95`, `weight_decay=0.1` | AdamW 配置 |
| 日志 | `use_comet`, `comet_config` | Comet ML 实验跟踪 |
| 模型路径 | `pretrained_tokenizer/predictor_path`, `finetuned_*_path` | 预训练权重 → 微调后权重 |
| 回测参数 | `backtest_n_symbol_hold=50`, `inference_T=0.6`, `top_p=0.9`, `sample_count=5` | TopkDropout 策略 + 推理采样 |
| 标准化 | `clip=5.0` | 归一化后的裁剪阈值 |

- **设计模式**：纯配置对象（Plain Config Object）——所有参数通过 `Config` 类的属性暴露，`__dict__` 转字典后传入各脚本。
- **集成点**：被 `dataset.py`、`qlib_data_preprocess.py`、`train_tokenizer.py`、`train_predictor.py`、`qlib_test.py` 全部引用。
- **注意**：使用前必须更新 `qlib_data_path`、`pretrained_tokenizer_path`、`pretrained_predictor_path`。

---

### `qlib_data_preprocess.py` — Qlib 数据预处理

- **类**：`QlibDataPreprocessor`
- **流程**：初始化 Qlib → 从 Qlib 按 `instrument` + 时间范围加载原始数据 → 按 symbol 逐只处理（透视、计算 vol/amt）→ 按 `train/val/test_time_range` 划分为三份 → Pickle 序列化到 `dataset_path`。

**Flow**：
```
qlib.init(provider_uri, REG_CN) → D.calendar()
    ↓ 计算 adjusted_start/end_index（包含 lookback/predict 缓冲区）
    ↓
QlibDataLoader 加载多只股票的 OHLCV + vwap
    ↓ stack + unstack → 按 symbol 列表迭代
    ↓ pivot → pivot_table 重塑 → 过滤 NaN → 筛选足够长度
    ↓
┌─ train_data.pkl (train_time_range)
├─ val_data.pkl   (val_time_range)
└─ test_data.pkl  (test_time_range)
```

- **设计要点**：
  - `adjusted_start_index` 减去 `lookback_window` 缓冲，确保训练样本有足够历史上下文。
  - `amt` 用 `(O+H+L+C)/4 × vol` 估算（Qlib 原始数据无成交额）。
  - 只保留长度 ≥ `lookback + predict + 1` 的股票。

---

### `dataset.py` — Qlib 滑动窗口数据集

- **类**：`QlibDataset(torch.utils.data.Dataset)`
- **职责**：从预处理好的 pickle 文件中加载数据，为训练/验证提供 **随机采样** 的滑动窗口样本。

**设计模式**：
- **预计算索引（Pre-computed Index Pool）**：在 `__init__` 中遍历所有 symbol 的 `(symbol, start_idx)` 组合，存入 `self.indices` 列表。
- **按 epoch 设置种子（Per-epoch Seeding）**：`set_epoch_seed(epoch)` 使用 `seed + epoch` 重新种子化随机数生成器，确保 DDP 各 epoch 采样不同但可复现。
- **无数据泄露**：归一化的均值/标准差仅从 `lookback_window`（过去数据）计算，不引用未来。

**`__getitem__` Flow**：
```
self.py_rng.randint(0, len(indices)-1) → 随机选 (symbol, start_idx)
    ↓
df.iloc[start_idx : start_idx + lookback + predict + 1]
    ↓
分离 feature 和 time_feature
    ↓
x[:lookback_window] 计算 mean, std → (x - mean) / (std+1e-5) → clip(-5, 5)
    ↓
返回 (x_tensor, x_stamp_tensor)
```

---

### `train_tokenizer.py` — Tokenizer 微调（DDP）

- **入口**：`torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py`
- **主函数** `main(config)`：
  1. `setup_ddp()` — 初始 `nccl` 后端，获取 `rank/world_size/local_rank`
  2. Rank 0 创建保存目录 + 初始化 Comet ML logger
  3. `KronosTokenizer.from_pretrained(pretrained_tokenizer_path)` 加载预训练权重
  4. `DDP(model, device_ids=[local_rank])` 包装为分布式模型
  5. 调用 `train_model()` 执行训练

**训练循环 Flow**：
```
for epoch in range(epochs):
    model.train()
    train_loader.sampler.set_epoch(epoch)
    ↓
    for batch_x, _ in train_loader:
        for accum_step in range(accumulation_steps):
            loss = recon_loss + bsq_loss
            loss_scaled = loss / accumulation_steps
            loss_scaled.backward()
        ↓
        clip_grad_norm_(2.0) → optimizer.step() → scheduler.step() → zero_grad()
    ↓
    model.eval()
    for batch_x, _ in val_loader:
        计算 val_loss (F.mse_loss)
    ↓
    dist.all_reduce 聚合验证损失
    ↓
    Rank 0: 若 val_loss 改善则 save_pretrained('best_model')
```

**损失函数**：
- `recon_loss = MSE(z_pre, x) + MSE(z, x)` — 重构损失
- `bsq_loss` — VQ-VAE 承诺损失 + 熵正则化
- `loss = (recon_loss + bsq_loss) / 2`

**设计模式**：
- **分布式数据并行（DDP）**：`DistributedSampler` + `DDP`，适应多 GPU。
- **梯度累积（Gradient Accumulation）**：将大 batch 拆分为 micro-batch，模拟更大 batch size。
- **OneCycleLR 调度**：先升后降的学习率策略（`pct_start=0.03`）。
- **Comet ML 日志**：Rank 0 记录训练/验证损失和学习率曲线。

---

### `train_predictor.py` — Predictor 微调（DDP）

- **入口**：`torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py`
- **关键区别**：加载 **微调后的 tokenizer**（`finetuned_tokenizer_path`）而非预训练版。Tokenizer 在训练期间处于 `eval()` 模式，`torch.no_grad()` 下编码。

**训练循环 Flow**：
```
for batch_x, batch_x_stamp in train_loader:
    with torch.no_grad():
        token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
    ↓
    token_in  = [tokens[:, :-1]]    (输入)
    token_out = [tokens[:, 1:]]     (目标，shift-by-1)
    ↓
    logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
    loss, s1_loss, s2_loss = model.module.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
    ↓
    optimizer.zero_grad() → loss.backward() → clip_grad_norm_(3.0) → optimizer.step() → scheduler.step()
```

**设计要点**：
- **Teacher Forcing**：使用真实 token 作为 decoder 输入，而非自回归生成。
- **层级损失**：s1（粗粒度）+ s2（细粒度）分别计算交叉熵损失，通过 `DualHead.compute_loss` 合并。
- Predictor 不使用梯度累积（通常 batch size 较小即可）。

---

### `qlib_test.py` — Qlib 回测

**三大组件**：

#### 1. `QlibTestDataset` — 按序滑动窗口测试数据集
- 与 `QlibDataset` 类似，但 **顺序遍历**（非随机采样），且 `__getitem__` 返回元数据 `(symbol, timestamp)` 用于映射预测结果到原时间序列。

#### 2. `generate_predictions(config, test_data)` — 推理引擎
```
QlibTestDataset → DataLoader (batch_size, collate_fn)
    ↓
auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, ...)
    ↓
preds[:, -pred_len:, :]   (取预测段)
    ↓
计算信号: last/mean/max/min (收盘价变化)
    ↓
pivot_table → 多信号 DataFrame {datetime × instrument}
```

**信号类型**：
| 信号 | 计算方式 |
|------|----------|
| `last` | `pred_close[-1] - last_history_close` |
| `mean` | `mean(pred_close) - last_history_close` |
| `max` | `max(pred_close) - last_history_close` |
| `min` | `min(pred_close) - last_history_close` |

#### 3. `QlibBacktest` — Qlib 回测包装器
- **策略**：`TopkDropoutStrategy(topk=50, n_drop=5, hold_thresh=5)`
- **执行器**：`SimulatorExecutor (day频次, delay_execution)`
- **账户**：1 亿元初始资金，沪深 300 基准
- **指标**：超额收益（含/不含成本）、风险分析（夏普、波动率、最大回撤等）
- **输出**：累计收益曲线图 + 多信号对比

**集成 Flow**：
```
qlib_data_preprocess.py → pickled datasets
    ↓
train_tokenizer.py     → 微调后 tokenizer 权重
    ↓
train_predictor.py     → 微调后 predictor 权重
    ↓
qlib_test.py ─── generate_predictions → 信号 DataFrame
    └── QlibBacktest.run_and_plot_results → 回测报告 + 图表
```

---

### `utils/training_utils.py` — DDP 工具函数

| 函数 | 职责 |
|------|------|
| `setup_ddp()` | 初始化 `nccl` 进程组，返回 `(rank, world_size, local_rank)` |
| `cleanup_ddp()` | 销毁进程组 |
| `set_seed(seed, rank)` | 设置 Python/NumPy/Torch/CUDA 随机种子 |
| `get_model_size(model)` | 计算可训练参数量（返回 "175.0M" 格式） |
| `reduce_tensor(tensor, world_size)` | 跨进程张量规约（SUM/AVG） |
| `format_time(seconds)` | 秒 → "H:M:S" 格式 |

---

## 设计模式总结

| 模式 | 位置 | 说明 |
|------|------|------|
| **纯配置对象（Plain Config Object）** | `config.py` | 所有参数集中在 `Config` 类的属性中，`__dict__` 透传 |
| **分布式数据并行（DDP）** | `train_tokenizer.py`, `train_predictor.py` | `DistributedSampler` + `DDP` 多 GPU 训练 |
| **梯度累积（Gradient Accumulation）** | `train_tokenizer.py` | micro-batch 累加梯度，模拟大 batch |
| **OneCycle 学习率调度** | 两个训练脚本 | 先升后降的 LR 策略 |
| **Teacher Forcing** | `train_predictor.py` | 训练时用真实 token 移位作为输入 |
| **预计算索引池（Pre-computed Index Pool）** | `dataset.py` | `__init__` 中枚举所有有效采样点 |
| **逐 epoch 种子（Per-epoch Seeding）** | `dataset.py` | `epoch_seed = seed + epoch` 确保可复现性 |
| **无数据泄露（No Look-ahead Bias）** | `dataset.py` | 归一化统计量仅从 lookback 窗口计算 |
| **滑动窗口（Sliding Window）** | `QlibTestDataset` | 遍历测试集生成连续推理样本 |
| **多信号聚合（Multi-signal Aggregation）** | `qlib_test.py` | last/mean/max/min 四种信号对比 |
| **TopkDropout 策略** | `qlib_test.py` | Qlib 内置的回测交易策略 |

---

## 关键集成点

| 模块 | 输入 | 输出 | 下游 |
|------|------|------|------|
| `qlib_data_preprocess.py` | Qlib 金融数据库 | `train/val/test_data.pkl` | `dataset.py`, `qlib_test.py` |
| `train_tokenizer.py` | 预训练 `KronosTokenizer` | 微调后 Tokenizer 权重 | `train_predictor.py` |
| `train_predictor.py` | 微调 Tokenizer + 预训练 `Kronos` | 微调后 Predictor 权重 | `qlib_test.py` |
| `qlib_test.py` | 微调后 Tokenizer + Predictor | 回测报告 + 收益曲线 | 实验决策 |

### 数据流约束
1. **两阶段顺序严格**：Tokenizer 必须先于 Predictor 训练。Predictor 的训练依赖 Tokenizer 的 encode 输出。
2. **微调前必须更新 config.py**：`qlib_data_path`、`pretrained_tokenizer_path`、`pretrained_predictor_path` 三个 TODO 必须填写。
3. **时间窗口重叠设计**：验证集起始时间早于训练集结束时间，以确保验证样本有足够的 lookback 历史。
