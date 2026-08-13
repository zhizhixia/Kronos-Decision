# Kronos 日频交易决策系统 — 设计规格

> 日期: 2026-07-10
> 状态: 已确认
> 方案: A — 扩展现有代码，最小侵入

## 1. 概述

在 Kronos 现有模型之上叠加决策信号层，为 A 股提供具备强指导意义的日频交易决策报告，通过 Web 界面交互使用。

### 1.1 目标

- 基于 Kronos 预测结果，产出多维决策信号（非单一分数）
- 专业级 Web 界面，选股 → 预测 → 信号 → 报告一键完成
- 稳定运行：完善的数据校验、错误处理、超时降级、日志体系
- 便于维护：中文注释/文档/日志，单一职责模块，类型注解

### 1.2 不做的事

- 不改动 `model/` 核心代码，决策层通过公共 API 调用
- 不动 `finetune/` 微调流水线（留待第二阶段）
- 不实现实时行情推送（次日频已足够）
- 不做交易执行（只出信号，不下单）

## 2. 架构

```
kronos/
├── model/                 ← 不改动
├── decision/              ← 🆕 决策层
│   ├── __init__.py
│   ├── analyzer.py        # 信号分析
│   ├── engine.py          # 决策引擎
│   ├── report.py          # 报告生成
│   ├── errors.py          # 错误定义
│   └── config.yaml        # 配置
├── data/                  ← 🆕 数据管道
│   ├── __init__.py
│   ├── fetcher.py         # A 股数据获取
│   ├── pool.py            # 股票池
│   └── cache/             # 数据缓存 (.parquet)
├── webui/                 ← 🔧 增强
│   ├── app.py             # 新路由
│   ├── templates/
│   │   ├── index.html     # 原有预测页面
│   │   └── report.html    # 🆕 决策报告页面
│   └── static/
│       └── style.css      # 🆕 深色主题样式
├── tests/
│   ├── test_analyzer.py   # 🆕
│   ├── test_engine.py     # 🆕
│   └── test_fetcher.py    # 🆕
├── logs/                  # 🆕 运行日志
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-07-10-kronos-decision-system-design.md  # 本文档
```

## 3. 决策信号体系

### 3.1 核心原则

不压缩成单一模糊分数，保留多维信息，通过决策矩阵做最终判定。

### 3.2 三层信号结构

#### Layer 1: 趋势判断

| 指标 | 计算方式 | 输出 |
|------|---------|------|
| 趋势方向 | 预测期末价格 vs 当前价格中位数 | ↑ / ↓ / → |
| 趋势强度 | `(预测量价涨跌幅度 × 方向一致性) / 归一化` | 0.0 ~ 1.0 |
| 走势一致性 | 50 条采样路径在终点的标准差 | 高 / 中 / 低 |

#### Layer 2: 风险评估

| 指标 | 计算方式 | 输出 |
|------|---------|------|
| VaR 95% | 50 条路径中第 5 分位最大回撤 | 百分比（如 -3.1%） |
| 波动率 | 预测路径高低价区间宽度 | 高 / 中 / 低 |
| 反转风险 | 顶/底背离检测 + 超买超卖判定 | 高 / 中 / 低 |

#### Layer 3: 决策矩阵

| 趋势强度 | 一致性 | 风险 | → 最终信号 |
|:--:|:--:|:--:|------|
| > 0.7 | 高 | 低 | 🟢 买入（强信号） |
| > 0.7 | 高 | 高 | 🟡 偏多持有 |
| > 0.7 | 低 | - | 🟡 持有（模型不确定） |
| 0.4~0.7 | - | - | 🟡 持有 |
| < 0.4 | - | - | 🔴 卖出/回避 |
| 任意 | - | 极高（VaR < -5%） | 🔴 回避（风险过大） |

阈值 0.7 / 0.4 / -5% 为初始默认值，通过 `config.yaml` 可调。

### 3.3 信号输出示例

```json
{
  "signal": "BUY",
  "signal_reason": "趋势明确上行，50条路径高度一致，风险可控",
  "trend": { "direction": "↑", "strength": 0.82, "confidence": "高" },
  "risk": { "var_95": -0.031, "volatility": "中", "reversal_risk": "低" },
  "prediction": { "chart_json": "{...}", "path_count": 50 },
  "analysis": { "up_probability": 0.87, "details": "..." }
}
```

## 4. Web 界面

### 4.1 视觉规范

| 项目 | 规格 |
|------|------|
| 配色 | `#0F1117` 底色, `#1A1D28` 卡片, `#00D4AA` 涨, `#FF4757` 跌 |
| 字体 | 系统中文 + Tabular Nums 等宽数字 |
| 图标 | Feather Icons (内嵌 SVG) |
| 布局 | 左侧 60px 导航栏 + 弹性内容区 |
| 动效 | 200ms ease 过渡 |

### 4.2 页面结构

| 导航项 | 页面 | 功能 |
|--------|------|------|
| 📊 | 实时预测 | 选股 → 预测 → 决策报告 |
| 💹 | 投资组合 | 批量监控自选股信号 |
| 📈 | 历史回顾 | 历史预测 vs 实际走势 |
| 📋 | 信号日志 | 历史决策信号列表 |
| ⚙️ | 设置 | 模型参数、股票池、数据源 |

### 4.3 实时预测页面布局

- **顶部仪表盘**: 上证/深证指数、模型状态、数据更新时间
- **左侧选股区**: 股票搜索、参数调节（预测天数/上下文/温度/Top-P）
- **进度条**: SSE 推送四阶段进度
- **右侧报告区**:
  - Plotly 交互 K 线图（历史 + 预测 + 置信区间）
  - 决策信号卡片
  - 50 条路径叠加图 + 末端分布
  - 六维评分雷达图
  - 历史信号表格

### 4.4 后端 API

| 方法 | 路由 | 说明 |
|------|------|------|
| GET | `/` | 原有预测页面 |
| GET | `/report` | 🆕 决策报告页面 |
| POST | `/api/decision-report` | 🆕 返回报告 JSON + Plotly 图表 |
| POST | `/api/batch-report` | 🆕 批量股票报告 |
| GET | `/api/stock-list` | 🆕 股票池列表 |
| GET | `/api/signal-history` | 🆕 历史信号 |
| GET | `/api/model-status` | 现有 |

### 4.5 SSE 进度推送

```
POST /api/decision-report
  → SSE: "fetching"   — 正在获取数据
  → SSE: "predicting" — 模型推理中 (进度%)
  → SSE: "analyzing"  — 分析决策信号
  → SSE: "complete"   — 返回完整 JSON
```

## 5. 数据管道

### 5.1 数据获取

- 数据源: `akshare.stock_zh_a_hist()`
- 缓存: Parquet 格式，`data/cache/{code}.parquet`
- TTL: 当日 16:00 后缓存失效
- 保留: 365 天历史，自动清理

### 5.2 数据校验

| 检查 | 规则 | 失败处理 |
|------|------|---------|
| 列完整性 | 必须有 open/high/low/close | 抛异常 |
| 非空 | 无空值行 | 删行，少于 50 行报错 |
| 日期连续 | 跳跃 ≤ 3 天 | 插值填补，> 10 天警告 |
| 高低价合理性 | high ≥ low ≥ 0 | > 5% 异常报错 |
| 复权一致性 | 自动识别前/后复权 | 标准化为前复权 |

### 5.3 重试策略

- 次数: 3 次
- 退避: 2s → 4s → 8s
- 降级: 三次全失败用缓存 + ⚠️ 标记

## 6. 错误处理与稳定性

### 6.1 异常体系

```
KronosError (基类)
├── DataSourceError      # 数据源不可用
├── DataValidationError  # 数据校验失败
├── ModelNotReadyError   # 模型未加载
├── PredictionTimeoutError # 预测超时
└── ConfigError          # 配置错误
```

### 6.2 降级策略

| 场景 | 处理 |
|------|------|
| 数据源不可用 | 缓存数据 + 🟡 数据延迟标签 |
| 模型加载失败 | 🟡 模型未就绪 + 重试按钮 |
| 预测超时 | sample_count 降为 10 重试，仍失败则部分输出 |
| 股票代码不存在 | 明确错误 + 建议相似代码 |

### 6.3 日志规范

```
[时间] [级别] [模块:行号] 消息
[2026-07-10 15:32:01] [ERROR] [fetcher.py:45] 获取 600519 数据失败 (重试 2/3): 连接超时
```

- 日志轮转: 50MB/文件, 保留 5 个备份
- 级别: DEBUG/INFO/WARNING/ERROR

### 6.4 配置集中管理

所有可调参数统一在 `decision/config.yaml`，不散落在代码中。

## 7. 核心模块说明

### `decision/analyzer.py` — 信号分析器

- `SignalAnalyzer.analyze(prediction_result) → SignalResult`
- 计算趋势方向/强度、一致性、VaR、波动率、反转风险
- 应用决策矩阵输出 BUY/HOLD/SELL
- 纯函数，无副作用

### `decision/engine.py` — 决策引擎

- `DecisionEngine.predict_and_analyze(stock_code, params) → DecisionReport`
- 编排: 拉数据 → 预测 → 信号分析 → 返回报告
- 含超时控制、降级逻辑

### `decision/report.py` — 报告生成

- `ReportBuilder.build(report_data) → dict`
- 生成 Plotly 图表 JSON + 结构化报告数据
- HTML 模板渲染在 `webui/templates/report.html`

### `data/fetcher.py` — 数据获取

- `DataFetcher.fetch_daily(stock_code) → pd.DataFrame`
- 含缓存、重试、校验

### `data/pool.py` — 股票池

- 沪深 300 成分股字典：`{code: name}`
- 支持从配置文件加载自定义股票池

## 8. 测试

| 文件 | 测试内容 |
|------|---------|
| `test_fetcher.py` | 数据获取 + 缓存 + 校验 + 重试降级 |
| `test_analyzer.py` | 趋势计算 / 风险评估 / 决策矩阵边界值 |
| `test_engine.py` | 端到端流程 / 超时降级 / 异常处理 |

## 9. 编码规范

- 注释/文档/日志/错误信息: 简体中文
- 变量/函数/类/文件名: 英文
- 类型注解: 所有公开函数必须标注
- 函数长度: 不超过 60 行
- 每个 `.py` 文件: 单一职责，不超过 300 行
- 配置与逻辑分离: 数值不在代码中硬编码

## 10. 不做的事

- 不实现实时行情推送
- 不做交易执行（下单）
- 不引入新的外部依赖（仅 akshare + baostock 新增）
- 不改动 model/ 和 finetune/ 目录

## 11. 架构细节澄清（grill-me 审查补充）

### 11.1 未来时间戳生成

`DecisionEngine` 负责根据历史数据最后日期 + `pred_len` 自动生成 A 股未来交易日历（跳过周末，已知节假日）。复用 Kronos 现有的 `calc_time_stamps()` 函数提取时间特征。

### 11.2 模型生命周期

独立 `ModelManager` 模块管理模型实例：

```
ModelManager
├── 懒加载：首次请求时加载模型到 GPU
├── 超时释放：N 分钟无请求后 model.cpu() 释放显存
├── 健康检查：模型对象存在且可推理才返回"就绪"
└── 复用：所有请求共享同一个模型实例
```

默认超时：10 分钟无请求释放显存（通过 `config.yaml` 可调）。

### 11.3 错误传播路径

`DecisionEngine` 内部 try/catch 所有异常，始终返回 `DecisionReport` 数据类：

```python
@dataclass
class DecisionReport:
    status: Literal["ok", "error", "degraded"]
    error_message: str = ""
    signal: SignalResult | None = None
    prediction: dict | None = None
```

Flask 路由只需根据 `report.status` 分支渲染，不需要 import 任何异常类型。

### 11.4 配置热更新

Web 设置页面的修改通过 `PUT /api/config` 接口：
- 立刻更新内存中的值（本次生效）
- 同步写回 `config.yaml`（持久化，下次启动自动加载）

### 11.5 多数据源容错

数据获取优先级：
```
akshare (主) → baostock (备) → 本地缓存 (最后降级)
```

降级规则：
- akshare 不可用 → 自动切换 baostock
- 两个都不可用 + 缓存存在 → 用缓存 + 🟡 "⚠️ 数据源不可用，显示为昨日数据，预测可能不准"
- 两个都不可用 + 无缓存 → 返回 DecisionReport(status="error")

### 11.6 股票池动态更新

沪深 300 成分股通过 `akshare.index_stock_cons()` 动态获取，缓存到 `data/cache/pool.parquet`：
- 每周自动刷新一次（与 akshare 重试策略一致，失败不影响服务）
- 拉取失败用上次缓存（成分股半年调一次，几天的差异不影响使用）
- 用户可在设置页面手动触发刷新
