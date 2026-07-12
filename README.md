<div align="center">
  <h2><b>Kronos: A Foundation Model for the Language of Financial Markets </b></h2>
</div>


<div align="center">

<a href="https://huggingface.co/NeoQuasar"> 
<img src="https://img.shields.io/badge/🤗-Hugging_Face-yellow" alt="Hugging Face"> 
</a> 
<a href="https://github.com/shiyu-coder/Kronos/graphs/commit-activity"> 
<img src="https://img.shields.io/github/last-commit/shiyu-coder/Kronos?color=blue" alt="Last Commit"> 
</a> 
<a href="https://github.com/shiyu-coder/Kronos/stargazers"> 
<img src="https://img.shields.io/github/stars/shiyu-coder/Kronos?color=lightblue" alt="GitHub Stars"> 
</a> 
<a href="https://github.com/shiyu-coder/Kronos/network/members"> 
<img src="https://img.shields.io/github/forks/shiyu-coder/Kronos?color=yellow" alt="GitHub Forks"> 
</a> 
<a href="./LICENSE"> 
<img src="https://img.shields.io/github/license/shiyu-coder/Kronos?color=green" alt="License"> 
</a>

</div>

<p align="center">

<img src="./figures/logo.png" width="100">

</p>

> Kronos is the **first open-source foundation model** for financial candlesticks (K-lines), 
> trained on data from over **45 global exchanges**.

---

## 🔥 增强版本声明

> **本项目是基于 [原 Kronos 项目](https://github.com/shiyu-coder/Kronos) 的增强版本**，在保留原 Kronos 全部核心能力的基础上，扩展了面向 A 股市场实战的完整工具链。

**原论文**：[Kronos: A Foundation Model for the Language of Financial Markets](https://arxiv.org/abs/2508.02739)（AAAI 2026）

### 增强内容概要

| 模块 | 说明 |
|------|------|
| **决策信号系统** (`decision/`) | 多路径预测 → 趋势/风险分析 → BUY/HOLD/SELL 决策矩阵，支持 A 股日 K 实时决策 |
| **数据管道** (`data/`) | A 股数据获取，三级降级策略（akshare → baostock → 缓存），沪深 300 成分股管理 |
| **增强 Web 界面** (`webui/`) | 深色交易终端主题，决策报告页，模型动态切换，在线配置管理，沪深 300 股票搜索 |
| **设计文档** (`docs/`) | 完整系统设计规格文档，含架构图、API 定义、视觉规范 |

### 目录结构（新增模块以 ⭐ 标注，增强模块以 ✨ 标注）

```
kronos/
├── model/                 # 核心库（原 Kronos）：Tokenizer + Predictor
├── decision/         ⭐   # 决策信号系统
│   ├── engine.py          # DecisionEngine 核心编排
│   ├── analyzer.py        # 信号分析器（趋势/风险/决策矩阵）
│   ├── model_manager.py   # 模型懒加载与显存管理
│   ├── config.py/.yaml    # 线程安全配置
│   └── errors.py          # 层次化异常体系
├── data/             ⭐   # 数据管道
│   ├── fetcher.py         # A 股数据获取（akshare/baostock/缓存三级降级）
│   ├── pool.py            # 沪深300成分股管理
│   └── cache/             # 本地数据缓存
├── webui/            ✨   # Web 可视化界面（增强版）
│   ├── app.py             # Flask 路由（+7 个新 API）
│   ├── templates/         # 预测页 + 决策报告页
│   └── run.py             # 启动入口
├── finetune/              # Qlib 微调流水线
├── finetune_csv/          # CSV 微调流水线
├── examples/              # 预测示例脚本
├── tests/                 # 回归测试
├── docs/            ⭐   # 设计文档
│   └── superpowers/specs/ # 系统设计规格
├── start.bat         ⭐   # Windows 一键启动脚本
└── PROJECT_DOCS.md   ⭐   # 完整技术文档
```

> 📖 **完整技术文档**：[PROJECT_DOCS.md](./PROJECT_DOCS.md)

---

## 🧠 决策系统快速开始

增强版内置了一个面向 A 股市场的智能决策系统，基于 Kronos 多路径预测生成趋势方向、风险评估和买卖信号。

### 一键启动

```shell
# Windows 用户双击运行
start.bat

# 或手动启动
cd webui && python run.py
```

启动后自动打开浏览器访问：
- **预测页面**：[http://localhost:7070](http://localhost:7070)
- **决策报告页**：[http://localhost:7070/report](http://localhost:7070/report)

### 决策系统工作流

```
A股数据获取 → 数据清洗 → Kronos 模型推理（多路径采样）→
信号分析（趋势方向/风险 VaR/RSI）→ 决策矩阵判定 → DecisionReport
```

### 决策信号说明

| 信号维度 | 分析内容 | 产出 |
|---------|---------|------|
| 趋势分析 | 预测路径方向一致性、强度 | `trend`: BULLISH / BEARISH / NEUTRAL |
| 风险评估 | VaR 95%、路径波动幅度 | `risk`: LOW / MEDIUM / HIGH |
| 决策矩阵 | 综合趋势+风险→交易信号 | `decision`: BUY / HOLD / SELL |

### API 增强

| 新增 API | 功能 |
|---------|------|
| `POST /api/decision-report` | 生成决策报告（集成 DecisionEngine） |
| `GET /api/stock-list` | 获取沪深 300 成分股列表 |
| `GET /api/available-models` | 查询可用模型（mini/small/base） |
| `POST /api/load-model` | 动态切换模型和设备 |
| `GET/PUT /api/config` | 在线查看/修改配置 |

### 配置说明

编辑 `decision/config.yaml` 可调整预测和决策参数：

```yaml
prediction:
  pred_len: 120        # 预测长度
  sample_count: 50     # 采样路径数
  temperature: 1.0     # 采样温度
  top_p: 0.9          # 核采样概率

signal:
  trend_buy_threshold: 0.7    # 趋势买入阈值
  trend_sell_threshold: 0.4   # 趋势卖出阈值
  var_high_risk: 0.05         # VaR 高风险阈值
  rsi_overbought: 70          # RSI 超买线
  rsi_oversold: 30            # RSI 超卖线
```

### 注意事项

- **首次启动**会自动从 Hugging Face 下载模型权重（约需数分钟）
- 决策系统依赖 akshare 库获取 A 股数据，若网络不稳定会自动降级到 baostock 或本地缓存
- 建议使用 **GPU（CUDA）** 运行以获得实时决策体验，CPU 模式下单次预测可能需要较长时间
- 本系统仅供研究参考，不构成投资建议

---

## 📖 Citation

If you use Kronos in your research, we would appreciate a citation to our [paper](https://arxiv.org/abs/2508.02739):

```
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets}, 
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025},
      eprint={2508.02739},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2508.02739}, 
}
```

## 📜 License 
This project is licensed under the [MIT License](./LICENSE).