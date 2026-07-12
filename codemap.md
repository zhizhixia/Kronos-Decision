# Repository Atlas: Kronos（增强版）

## 项目职责

Kronos 是全球首个开源的金融 K 线（OHLCV）时序基础模型，基于 **Transformer 两阶段架构（Tokenizer + Predictor）**，已被 AAAI 2026 接收。本增强版在原项目基础上扩展了面向 A 股市场的完整决策工具链。

> 本仓库基于 [shiyn-coder/Kronos](https://github.com/shiyu-coder/Kronos) 增强
> 完整项目文档见 `PROJECT_DOCS.md`

## 系统入口点

| 入口 | 类型 | 用途 |
|------|------|------|
| `webui/run.py` | 启动脚本 | 启动 Flask Web 服务（端口 7070） |
| `webui/app.py` | Flask 后端 | HTTP API（预测 + 决策报告 + 配置 + 股票列表） |
| `start.bat` | 一键启动 | Windows 快捷启动脚本 |

## 目录地图

| 目录 | 职责摘要 | 详细地图 |
|------|---------|----------|
| `model/` | **核心模型层** — VQ-VAE Tokenizer + 自回归 Transformer Predictor，两阶段架构将 OHLCV 量化为离散 token 后建模预测 | [查看](model/codemap.md) |
| `decision/` | **决策信号层** 🆕 — 异常体系、配置管理、模型生命周期管理、SignalAnalyzer（趋势/风险/决策矩阵）、DecisionEngine（端到端编排） | [查看](decision/codemap.md) |
| `data/` | **数据管道** 🆕 — 三级降级数据获取（akshare→baostock→缓存）、沪深 300 动态股票池 | [查看](data/codemap.md) |
| `webui/` | **Web 可视化界面** — Flask 后端（11 条 API 路由）+ 原预测页（浅色/Plotly）+ 决策报告页（深色终端/Canvas K 线/全中文） | [查看](webui/codemap.md) |
| `tests/` | **测试套件** — 模型回归测试 + 决策系统测试 + 数据/模型测试 + 端到端集成测试 | [查看](tests/codemap.md) |
| `examples/` | **预测示例** — 单序列/批量/置信区间/回测/A 股数据获取等 14 个脚本 | [查看](examples/codemap.md) |
| `finetune/` | **Qlib 微调流水线** — 基于 Qlib 的 A 股 DDP 分布式训练（Tokenizer → Predictor → 回测） | [查看](finetune/codemap.md) |
| `finetune_csv/` | **CSV 微调流水线** — YAML 配置驱动的单机顺序训练，更简洁 | [查看](finetune_csv/codemap.md) |

## 数据流概要

```
外部数据源 (akshare/baostock)
  │
  ▼
data/fetcher.py ──→ data/pool.py
  │                    │
  │                    ▼
  │              沪深 300 股票池
  ▼
decision/engine.py
  │
  ├─ 1. fetch_daily()     → DataFrame (OHLCV + date)
  ├─ 2. prepare_inputs()  → x_df, x_timestamp, y_timestamp
  ├─ 3. model.predict()   → pred_df (未来 120 步 OHLCV)
  ├─ 4. simulate_paths()  → 20 条噪声路径
  └─ 5. analyzer.analyze() → SignalResult
       │
       ▼
  DecisionReport (JSON)
       │
       ▼
  webui/app.py → renderReport() → Canvas K 线图 + 信号卡片
```

## 核心架构决策

| 决策 | 选型 | 理由 |
|------|------|------|
| 错误处理 | Engine 始终返回 DecisionReport（永不抛异常） | 统一 API 契约 |
| 模型管理 | 独立 ModelManager + 懒加载 + 超时释放 GPU | 避免空占显存 |
| 数据降级 | akshare → baostock → 过期缓存 | 保障服务可用性 |
| 缓存格式 | CSV（无 pyarrow 依赖） | 避免额外依赖 |
| 前端信号 | Canvas 原生 K 线 + 深色终端主题 | 性能 + 专业感 |

> 本文档由 codemap 自动生成，最后更新 2026-07-12
