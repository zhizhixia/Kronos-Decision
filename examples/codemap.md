# examples/ — 预测示例

Kronos 模型的一站式预测示例集合。涵盖从单序列预测到批量推理、从置信区间估计到回测验证、从 A 股数据获取到可视化 GUI 的全场景用例。

---

## 文件清单与职责

### 一、基础推理示例

#### `prediction_example.py` — 单序列预测（最简示范）

**流程**：
```
1. 加载 Tokenizer (Kronos-Tokenizer-base) + Model (Kronos-small) → KronosPredictor
2. 从 CSV 读取 5min K 线数据（600977 中国电影）
3. 截取前 400 条作为 lookback，后 120 条作为预测目标
4. predictor.predict(df, x_timestamp, y_timestamp, pred_len=120) → pred_df
5. 绘图对比 Ground Truth vs Prediction（收盘价 + 成交量）
```

- **设计模式**：简化接口模式（Facade）——`KronosPredictor.predict()` 封装了校验→标准化→推理→反标准化的完整流程。
- **输出**：`pred_df`（DataFrame）+ 可视化图表。

---

#### `prediction_wo_vol_example.py` — 无 volume/amount 列的单序列预测

与 `prediction_example.py` 相同，但输入数据仅包含 `['open', 'high', 'low', 'close']` 四列。

- **设计要点**：`KronosPredictor.predict()` 自动检测缺失列并填充为 0，因此缺少 `volume` 和 `amount` 时仍可正常运行。
- **输出**：单 panel 收盘价对比图（无成交量子图）。

---

#### `prediction_batch_example.py` — 批量序列预测

**流程**：
```
1. 加载 Tokenizer + Model（使用 Kronos-base）
2. 从同一 CSV 文件构建 5 个不重叠的窗口（每个 400 lookback + 120 predict）
3. predictor.predict_batch(
       df_list=[df1..df5],
       x_timestamp_list=[...],
       y_timestamp_list=[...],
       pred_len=120
   ) → [pred_df_1, ..., pred_df_5]
```

- **设计模式**：批量处理（Batch Processing）——`predict_batch` 内部将多个序列堆叠为 `(batch, seq, feat)` 四维张量，一次前向传播完成所有预测。
- **适用场景**：对同一股票的不同时间段或不同股票同时推理。

---

### 二、置信区间与概率预测

#### `predict_with_confidence.py` — 分布式采样与置信区间

**核心功能**：通过 **多次自回归采样**（SAMPLE_COUNT=50）生成预测分布，计算置信区间和交易信号。

**流程**：
```
[1/3] 加载 Kronos-base → GPU
[2/3] 对 STOCKS 列表中的每只股票：
    ├── fetch_stock_data(code) — 腾讯财经 API 获取日线复权数据（800 天）
    ├── prepare inputs (LOOKBACK=400, PRED_LEN=60)
    ├── generate_distribution(tokenizer, model, ...)
    │    ├── 分块推理 (paths_per_batch=10, sample_count=50)
    │    ├── 逐 token 自回归解码（s1→s2 双重采样）
    │    └── 解码回 OHLCV → (sample_count, pred_len, 6)
    ├── analyze_distribution(paths, current_price)
    │    ├── 期末价格统计（均值/中位数/标准差）
    │    ├── 涨跌概率
    │    ├── VaR 95%/99%
    │    ├── 四因子综合评分（方向 35% + 收益 25% + 稳定性 20% + 尾部风险 20%）
    │    └── 信号判断 (BUY/HOLD/SELL/WAIT)
    └── plot_distribution(res, future_dates) → 保存置信区间图表
[3/3] 输出 CSV → 含 mean/p10/p50/p90/std 路径
```

**`generate_distribution()` — 分布式推理引擎**：
```
输入 x (1, LOOKBACK, 6)
    ↓ repeat(1, chunk_size, 1, 1) → (chunk_size, LOOKBACK, 6)
    ↓
tokenizer.encode(x, half=True) → [s1_tokens, s2_tokens]
    ↓
循环 pred_len 步：
    ├── 滑动窗口缓冲区 (pre_buffer, post_buffer)
    ├── model.decode_s1() → s1_logits → sample_from_logits() → next_s1
    ├── model.decode_s2() → s2_logits → sample_from_logits() → next_s2
    ├── 写入 generated_pre/post
    └── 更新滑动缓冲区 (roll)
    ↓
tokenizer.decode() → (chunk_size, total_seq_len, 6)
    ↓ 收集全部分块 → concatenate → 返回预测段
```

**`analyze_distribution()` — 统计与信号**：

| 统计指标 | 计算方式 | 用途 |
|----------|----------|------|
| 涨跌概率 | `P(final_price > current_price)` | 方向判断 |
| VaR 95/99 | 期末收益率 5%/1% 分位数对应价格 | 尾部风险衡量 |
| 期望收益 | `mean(final)/current - 1` | 预期回报 |
| 最大回撤 | 每条路径的 `max_drawdown` 均值/最小值 | 下行保护 |
| 四因子评分 | 方向×0.35 + 收益×0.25 + 稳定性×0.20 + 尾部风险×0.20 | 综合交易信号 |

**设计模式**：
- **分块推理（Chunked Inference）**：按 `paths_per_batch` 分块执行，避免 GPU OOM。
- **多候选采样（Multi-sample Monte Carlo）**：多次自回归采样构建预测分布。
- **滑动缓冲区（Sliding Window Buffer）**：固定长度 `max_context` 的循环 buffer，模拟 Transformer 有限上下文。

---

### 三、A 股数据获取与预测

#### `prediction_cn_markets_day.py` — A 股日 K 线预测（AKShare）

**特点**：
- 通过 **AKShare** 实时获取 A 股日线数据（支持重试机制）
- 使用 `KronosPredictor.predict()` 单序列预测
- 保存预测结果 CSV + 收盘价对比图

**用法**：`python prediction_cn_markets_day.py --symbol 000001`

#### `prediction_akshare_2024-2025.py` — 批量 A 股预测

- 支持中文列名自动映射（日期→timestamps, 开盘价→open...）
- 自适应参数计算：`calculate_prediction_parameters()` 根据目标天数自动调整 lookback/predict 长度
- 完整输出：预测 CSV + 可视化图表

#### `prediction_new.py` — AKShare 多股票多场景预测

**综合性预测脚本**，功能覆盖：
- **数据获取**：`akshare.stock_zh_a_hist()` 实时获取 A 股数据
- **多周期**：支持日线/周线/月线
- **复权选项**：前复权/后复权/不复权
- **指标计算**：涨跌幅、振幅、换手率自动映射
- **预测界面**：完整的 CLI 交互体验

#### `prediction_new_GUI.py` — Tkinter 图形界面（1624 行）

**全功能桌面应用**：
- **技术**：Tkinter + matplotlib.backends.backend_tkagg
- **功能**：股票代码输入、数据获取、模型预测、结果图表展示
- **辅助**：多数据源支持（AKShare/腾讯/EastMoney/BaoStock）

---

### 四、多数据源适配

#### `predict_my_stocks.py` — 腾讯 API 多股票预测

- 使用 **腾讯财经 API**（`web.ifzq.gtimg.cn`）获取前复权日线数据
- 自动判断交易所（sh/sz）
- 支持自定义股票池并发预测
- 保存结果到 `predictions/` 目录

#### `fetch_tencent.py` — 腾讯 API 数据获取测试

- 最小化测试脚本：验证腾讯 API 数据接口可用性
- 返回 OHLCV DataFrame

#### `get_date_new.py` — 多源数据获取工具（660 行）

**三数据源容错机制**：
```
AKShare → 失败 → BaoStock → 失败 → 东方财富 → 全部失败 → 生成模拟数据
```

- 自动市场代码映射（6/9 开头→上交所，0/2/3→深交所）
- EastMoney JSONP 响应解析
- 模拟数据生成（带随机游走的 `create_sample_data()`）
- 按年份范围筛选

#### `get_akshare_date_2024-2025_x.py` — AKShare 年份范围数据获取

- 基于 AKShare 的简化版数据获取器
- 支持指定年份范围 `start_year ~ end_year`

---

### 五、回测验证

#### `run_backtest_kronos.py` — 模型回测系统（454 行）

**完整回测框架**：

```
KronosBacktester:
    ├── load_historical_data(stock_code)    — 加载历史 CSV
    ├── load_predictions(stock_code)        — 加载模型预测 CSV
    ├── calculate_trading_signals()         — 计算交易信号（阈值 2%）
    │       combined['pred_return'] > threshold → 买入信号
    │       combined['pred_return'] < -threshold → 卖出信号
    ├── run_backtest(combined_df)           — 模拟交易
    │       └── 全仓交易、signal→position、记录日收益率
    ├── calculate_metrics(results,trades)   — 绩效指标计算
    │       └── 总收益率、年化收益率、波动率、夏普比率、最大回撤、胜率
    └── plot_backtest_results()             — 三面板图表
            ├── 资金曲线（含初始资金线）
            ├── 累计收益 vs 基准（买入持有）
            └── 回撤曲线
```

**设计模式**：
- **策略-回测分离**：`calculate_trading_signals()` 负责信号生成，`run_backtest()` 负责执行——关注点分离。
- **指标计算**：年化收益率（252 交易日）、夏普比率（无风险利率 3%）、滚动最大回撤。

#### `yuce/historical_backtest.py` — 历史回测验证

- `HistoricalBacktester` 类，结构类似 `run_backtest_kronos.py`
- 从 `data/` 和 `yuce/` 子目录加载数据
- 完整的回测报告输出

---

## 设计模式总结

| 模式 | 位置 | 说明 |
|------|------|------|
| **简化接口（Facade）** | `KronosPredictor.predict()` | 封装校验→标准化→推理→反标准化 |
| **批量处理（Batch Processing）** | `predict_batch()` | 多序列堆叠并行推理 |
| **分块推理（Chunked Inference）** | `generate_distribution()` | 按 `paths_per_batch` 分块，防 OOM |
| **多候选采样（Multi-sample MC）** | `predict_with_confidence.py` | 50 次采样构建预测分布 |
| **滑动窗口缓冲区（Sliding Window Buffer）** | `generate_distribution()` | 固定长度循环缓冲区模拟自回归上下文 |
| **四因子综合评分（Composite Score）** | `analyze_distribution()` | 方向+收益+稳定性+尾部风险的加权评分 |
| **策略-回测分离（Separation of Concerns）** | `run_backtest_kronos.py` | 信号生成与交易执行独立 |
| **多数据源容错（Multi-source Failover）** | `get_date_new.py` | AKShare→BaoStock→EastMoney→Simulated |
| **自适应参数（Adaptive Parameters）** | `prediction_akshare_2024-2025.py` | 根据目标预测天数自动计算窗口参数 |
| **外观模式（Facade）** | `prediction_example.py` | 一条 `predict()` 完成全流程 |
| **GUI 模式（MVVM-ish）** | `prediction_new_GUI.py` | Tkinter 分离界面逻辑与推理逻辑 |

---

## 关键集成点

| 示例脚本 | 使用的模型 | 数据源 | 输出 |
|----------|-----------|--------|------|
| `prediction_example.py` | Kronos-small + Tokenizer-base | 本地 CSV（5min K 线） | 可视化图表 |
| `prediction_batch_example.py` | Kronos-base + Tokenizer-base | 本地 CSV（5min K 线） | 多序列 DataFrame |
| `predict_with_confidence.py` | Kronos-base + Tokenizer-base | 腾讯 API（日线） | 置信区间图表 + CSV + 信号 |
| `prediction_cn_markets_day.py` | Kronos-base + Tokenizer-base | AKShare（日线） | CSV + 图表 |
| `prediction_new.py` | Kronos (any) | AKShare（日/周/月） | CSV + 图表 |
| `prediction_new_GUI.py` | Kronos (any) | 多数据源 | Tkinter GUI |
| `predict_my_stocks.py` | Kronos-base + Tokenizer-base | 腾讯 API（日线） | 多股票预测 |
| `run_backtest_kronos.py` | 预计算预测 CSV | 本地 CSV + 预测 CSV | 回测报告 + 图表 |
| `fetch_tencent.py` | — | 腾讯 API | DataFrame |
| `get_date_new.py` | — | AKShare/BaoStock/东方财富 | DataFrame |

### 数据契约

所有示例遵循 Kronos 数据约定：
- **必需列**：`['open', 'high', 'low', 'close']`（price_cols）
- **可选列**：`['volume', 'amount']`（缺失自动填 0）
- **时间戳列**：`timestamps` 或 `date`（自动转换）
- **标准化**：窗口内 Z-score → clip[-5, 5]
- **模型配对**：Kronos-small/base ↔ Tokenizer-base，Kronos-mini ↔ Tokenizer-2k
