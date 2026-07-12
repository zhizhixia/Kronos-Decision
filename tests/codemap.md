# tests/

Kronos 项目测试套件。覆盖模型回归、决策系统、数据/模型管理及端到端集成。

---

## Responsibility（职责）

| 文件 | 职责 | 被测模块 |
|------|------|----------|
| `test_kronos_regression.py` | **原始回归测试**：在固定模型/分词器版本下验证预测输出的数值确定性。包含两项参数化子测试——精确输出对比（`assert_allclose`）和 MSE 统计对比 | `model.Kronos`, `model.KronosPredictor`, `model.KronosTokenizer` |
| `test_errors.py` | **错误类型体系**：验证 `KronosError` 异常层次结构、raise/catch 语义和消息完整性 | `decision.errors` |
| `test_config.py` | **配置加载与持久化**：验证 `Config` dataclass 的默认值、子配置结构、save/reload 热更新一致性 | `decision.config` |
| `test_analyzer.py` | **信号分析**：用合成的 OHLCV 路径模拟不同趋势/噪声场景，验证 `SignalAnalyzer` 的 BUY/HOLD/SELL 输出和 `SignalResult` 字段完整 | `decision.analyzer` |
| `test_engine.py` | **决策引擎**：验证 `DecisionEngine` 的容错行为（无效代码返回 error 报告而非抛异常）、报告结构及自定义参数透传 | `decision.engine` |
| `test_fetcher.py` | **数据获取**：验证真实股票数据拉取的列格式、无效代码错误处理、缓存命中性能 | `data.fetcher` |
| `test_pool.py` | **股票池**：验证沪深 300 成分股池的格式（dict、6 位代码、非空名称）和刷新行为 | `data.pool` |
| `test_model_manager.py` | **模型管理器**：验证单例模式、就绪状态查询和初始未加载行为 | `decision.model_manager` |
| `test_integration.py` | **端到端集成**：验证全部新模块可导入、配置全局可访问、最小化流水线（不依赖 GPU/网络）可运行 | 跨模块 |
| `data/`（目录） | **回归测试静态数据**：存放 `regression_input.csv`、两个上下文长度的预期输出 CSV，以及用于重新生成 fixtures 的 `generate_regression_output.py` | —— |

---

## Design Patterns（设计模式）

### 1. 回归测试 —— 黄金数据对比（Golden Data / Snapshot Testing）
```python
np.testing.assert_allclose(obtained, expected, rtol=1e-5)
```
`test_kronos_regression.py` 将模型在某次已知良好 commit 上的输出固化为 CSV（`regression_output_*.csv`），后续修改后断言偏差不超过 `1e-5`。`generate_regression_output.py` 是 fixture 重新生成的入口，构成**可复现回归基线**。

### 2. 工厂 / Fixture 模式（pytest fixtures）
```python
@pytest.fixture
def analyzer(): return SignalAnalyzer()
@pytest.fixture
def engine(): return DecisionEngine()
@pytest.fixture
def fetcher(): return DataFetcher()
```
被测对象（`SignalAnalyzer`、`DecisionEngine`、`DataFetcher`）通过 pytest fixture 创建，便于参数化替换和测试隔离。

### 3. 模拟数据生成（Synthetic Test Data）
`test_analyzer.py` 的 `make_paths()` 函数按趋势方向（up/down/flat）和噪声水平生成模拟 OHLCV 路径，使信号分析测试不受外部数据源依赖。趋势幅度随路径索引线性变化，引入可控的路径间差异。

### 4. 参数化测试（Parametrized Tests）
```python
@pytest.mark.parametrize("context_len", TEST_CTX_LEN)           # test_kronos_regression
@pytest.mark.parametrize("context_len, expected_mse", zip(...)) # test_kronos_regression
```
同一逻辑在不同参数下验证，减少重复代码。

### 5. 单例验证（Singleton Verification）
```python
mgr1 = get_model_manager(); mgr2 = get_model_manager()
assert mgr1 is mgr2
```
`test_model_manager.py` 通过 `is` 身份比较验证工厂函数返回同一实例。

### 6. 错误通道而非异常通道（Error as Return Value）
```python
report = engine.predict_and_analyze("999999")
assert report.status in ("ok", "error", "degraded")
```
`test_engine.py` 验证无效输入以决策报告（`DecisionReport`）的 `status` 字段返回，而非抛出异常——体现了**错误是值**（Errors as Values）的设计哲学。

### 7. 最小集成测试（Minimal Integration Smoke Test）
`test_integration.py` 仅验证 import 成功、"典型"配置可读、一条最短流水线可完成——不依赖 GPU/网络，适合 CI 快速反馈。

---

## Flow（数据 / 控制流）

### 回归测试（test_kronos_regression.py）

```
regression_input.csv ──→ pd.read_csv ──→ context_df[:ctx_len] ──→ KronosPredictor.predict()
                                              │                       │
                                              └── feature slice ─────┘
                                                          │
                                                          ▼
                                                    pred_df (OHLCV)
                                                          │
                                    ┌─────────────────────┤
                                    ▼                     ▼
                        np.testing.assert_allclose    MSE vs expected_mse
                        (vs regression_output_*.csv)   (statistical check)
```

### 决策系统测试（test_analyzer.py）

```
make_paths(trend, noise) ──→ np.ndarray (N×pred_len×5) ──→ SignalAnalyzer.analyze()
                                                                    │
                                                                    ▼
                                                              SignalResult
                                 ├───────────────────────────────┼──────────────────────────────┐
                                 ▼                               ▼                            ▼
                         trend.direction                risk.volatility              signal (BUY/HOLD/SELL)
                         trend.strength                 risk.reversal_risk           signal_reason
                         trend.confidence
```

### 决策引擎测试（test_engine.py）

```
engine.predict_and_analyze(stock_code[, params])
         │
         ├── 有效代码 ──→ DecisionReport(status="ok", elapsed_seconds≥0)
         │
         └── 无效代码 ──→ DecisionReport(status="error", stock_code="999999")
```

### 数据模块测试

```
fetcher.fetch_daily("600519") ──→ DataFrame (含 open/high/low/close...)
                                         │
                                         ├── 首次请求 ──→ 网络获取 + 缓存
                                         │
                                         └── 二次请求 ──→ 缓存命中 (< 1.0s)

get_hs300_pool() ──→ dict {code: name}  (6 位代码, 非空名称)
         │
         └── refresh_pool() ──→ 强制重新拉取
```

### 模型管理器测试

```
get_model_manager() ──→ ModelManager 实例 (单例)
         │
         ├── mgr1 is mgr2  (同一引用)
         │
         └── mgr.is_ready() ──→ bool (环境依赖)

ModelManager() ──→ 初始未加载 (not is_ready())
```

### 集成测试

```
test_import_all_modules → 验证模块级 import 无异常
         │
test_config_accessible → get_config().model.max_context == 512
         │
test_full_pipeline_minimal → DecisionEngine → predict_and_analyze("000001")
                                                         │
                                                         ▼
                                                   DecisionReport(status ≠ 异常)
```

### 回归测试数据目录（tests/data/）

```
generate_regression_output.py (手动运行)
         │
         ├── regression_input.csv ──→ 输入特征供 test_kronos_regression 读取
         │
         └── regression_output_256.csv     ──→ 预期输出 (context=256)
             regression_output_512.csv     ──→ 预期输出 (context=512)
                           ▲
                           │
                    test_kronos_regression.py 读取对比
```
