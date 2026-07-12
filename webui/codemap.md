# `webui/` — Kronos 金融预测 Web 界面

## 总体职责

为 Kronos 金融时序预测模型提供 **B/S 架构图形化操作界面**，包含两套独立前端：
1. **原预测页（`index.html`）**：数据加载 → 模型选择 → K 线预测 → 对比分析的传统工作流
2. **决策报告页（`report.html`）**：股票搜索 → 一键生成带买卖信号、趋势分析、风险评估的决策报告

后端统一由 `app.py`（Flask）提供 API。

---

## 文件职责

### `app.py` — Flask 后端（775 行）

**Responsibility（职责）**
- 提供 8 条 RESTful API 路由，支撑两套前端的所有交互
- 封装 Kronos 模型加载、数据文件解析、预测执行、结果持久化
- 集成 Plotly 图表生成与 Canvas 图表数据接口

**API 路由一览**

| 路由 | 方法 | 功能 | 所属前端 |
|---|---|---|---|
| `/` | GET | 根路由，302 重定向到 `/report`（决策报告页） | — |
| `/report` | GET | 渲染 `report.html` 模板 | 决策报告 |
| `/api/data-files` | GET | 扫描 `data/` 目录，返回可用 CSV/Feather 文件列表 | 原预测页 |
| `/api/load-data` | POST | 加载指定文件，校验 OHLCV 列，返回行数/时间范围/频率等信息 | 原预测页 |
| `/api/predict` | POST | 执行预测（lookback=400, pred_len=120），返回 Chart JSON + 预测结果 + 实际对比数据 | 原预测页 |
| `/api/load-model` | POST | 从 Hugging Face 加载 Kronos-mini/small/base 模型到指定设备 | 原预测页 |
| `/api/available-models` | GET | 返回可用模型配置列表 | 原预测页 |
| `/api/model-status` | GET | 返回模型可用性 + 加载状态 | 原预测页 |
| `/api/decision-report` | POST | 调用 `DecisionEngine` 生成完整决策报告（含信号、趋势、风险） | 决策报告 |
| `/api/stock-list` | GET | 从 `data.pool.get_hs300_pool()` 获取沪深 300 股票池 | 决策报告 |
| `/api/config` | GET/PUT | 读取/保存决策引擎 YAML 配置（嵌套 dataclass 树） | 决策报告 |

> 注：`/api/decision-report`、`/api/stock-list`、`/api/config` 三条路由定义在文件末尾 `# === 决策报告 API ===` 段落中，依赖 `decision/` 模块。

**Design Patterns（设计模式）**
- **单例式全局状态**：`tokenizer`、`model`、`predictor` 三个模块级变量持有模型实例，通过 `global` 关键字在各路由间共享
- **哨兵对象**：`MODEL_AVAILABLE` 布尔值作为功能降级开关；导入失败时走模拟路径
- **辅助函数封装**：`load_data_file()` / `create_prediction_chart()` / `save_prediction_results()` 将 IO、图表、持久化逻辑与路由解耦
- **策略配置**：`AVAILABLE_MODELS` 字典以模型名为 key，配置参数化，路由通过 key 索引
- **条件数据解析**：`load_data_file()` 探测 timestamp 列名（`timestamps` / `timestamp` / `date`），丢失时自动生成

**Data Flow（数据流）**
```
用户操作 → Flask request → 路由处理 → 数据加载/模型预测 → JSON 响应 → 前端渲染
                                             ↓
                                   save_prediction_results()
                                             ↓
                                   prediction_results/*.json
```

**Integration Points（集成点）**
- `from model import Kronos, KronosTokenizer, KronosPredictor` — 核心模型库
- `from decision.engine import DecisionEngine` — 决策引擎（生成报告）
- `from data.pool import get_hs300_pool` — 股票池数据
- `from decision.config import get_config, reload_config, save_config` — 配置管理
- `sys.path.append(project_root)` 动态添加项目根路径到 Python 模块搜索路径

---

### `run.py` — 启动脚本（83 行）

**Responsibility（职责）**
- 启动入口：依赖检查 → 模型可用性探测 → 启动 Flask 开发服务器
- 提供交互式依赖安装流程（检测到缺失时询问用户是否自动 `pip install`）

**Design Patterns（设计模式）**
- **引导（Bootstrap）脚本**：封装启动前的环境准备和健康检查，而非直接调用 `app.run()`
- **防御式编程**：`try/except ImportError` 包裹模型导入，缺失仅警告不崩溃

**Data Flow**
```
run.py → check_dependencies() → main() → 打印启动信息 → app.run(port=7070)
```

**Integration Points**
- `from app import app` — 引入 Flask 应用实例
- `from model import Kronos, KronosPredictor` — 仅作可用性检测
- 硬编码端口 `7070`，监听 `0.0.0.0`（局域网可访问）

---

### `static/style.css` — 深色主题（1506 行）

**Responsibility（职责）**
- 为 `report.html` 提供 **Bloomberg Terminal / TradingView 风格** 的深色量化交易终端主题
- 完整的设计令牌系统 + 全组件样式

**Design Patterns（设计模式）**
- **CSS 自定义属性（Design Tokens）**：68 个 `:root` 变量统一定义颜色、字体、阴影、圆角、过渡，实现单一变更源
- **BEM 近似命名**：`.signal-card` / `.signal-card-header` / `.signal-card-recommendation` 等层级嵌套命名
- **CSS 层叠纹理**：`body::before` 伪元素叠加 32px 网格纹理营造终端气氛
- **信号色语义系统**：`--signal-buy`(绿) / `--signal-sell`(红) / `--signal-hold`(黄)，配合 `--dim` 透明度变体用于背景
- **动画微交互**：入场 `fadeUp` 错落动画（每子元素 80ms 延迟）、加载态 indeterminate 进度条滑动动画、shimmer 辉光

**Integration Points**
- 被 `report.html` 通过 `{{ url_for('static', filename='style.css') }}` 引用
- 字体依赖 Google Fonts（Space Grotesk / JetBrains Mono）

---

### `templates/index.html` — 原预测页（1238 行）

**Responsibility（职责）**
- 传统金融数据预测工作流：数据文件选择 → 模型加载 → 参数调节 → 时间窗口滑块 → K 线预测 → 对比分析
- 前端逻辑由内联 `<style>` + 内联 `<script>` 承载（Axios + Plotly.js CDN）

**Design Patterns（设计模式）**
- **浅色渐变主题**：紫蓝渐变背景 `linear-gradient(135deg, #667eea, #764ba2)`，白色卡片面板
- **双列布局**：左侧控制面板（模型/数据/参数）→ 右侧 Plotly 图表 + 对比分析
- **复杂状态管理**：通过 DOM 元素的 `display` / `classList` 切换控制 UI 状态（加载、错误、数据就绪、对比分析）
- **自定义时间窗口滑块**：纯粹用 JavaScript 实现的 Range Slider 组件（`slider-handle` 拖拽 + 选区高亮），用于选择 400+120 数据窗口

**Data Flow**
```
用户选择文件 → GET /api/data-files → 填充下拉列表
用户点击「加载数据」→ POST /api/load-data → 显示行数/时间范围/频率
用户选择模型 → POST /api/load-model → 模型就绪
用户拖动时间窗口 → 更新起止日期显示
用户点击「开始预测」→ POST /api/predict → Plotly JSON → Plotly.newPlot()
用户点击「比较分析」→ 基于 actual_data 渲染差值表 + 误差统计卡片
```

**Integration Points**
- 所有 API 路由指向 `app.py` 的对应端点
- Plotly.js CDN (https://cdn.plot.ly/plotly-latest.min.js)
- Axios CDN (https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js)

---

### `templates/report.html` — 决策报告页（1267 行）

**Responsibility（职责）**
- **一站式决策报告生成**：股票搜索 → 一键生成含买卖信号、趋势分析、风险评估、评分表格、Canvas 蜡烛图的完整报告
- 内嵌完整前端 SPA 逻辑（IIFE 模式，~960 行 JS）

**Design Patterns（设计模式）**
- **IIFE 模块模式**：整个前端逻辑包裹在 `(function() { 'use strict'; ... })()` 中，避免全局污染
- **Autocomplete 自定义组件**：键盘导航（↑↓→Enter/Esc）+ 模糊搜索 + 防抖加载 + ARIA 无障碍属性
- **配置扁平化（flattenConfig/unflattenConfig）**：将嵌套的 Python dataclass 配置拍平为 `a.b.c` 键值对用于 HTML 表单渲染，提交时还原
- **Canvas 原生蜡烛图**：完全用 Canvas 2D API 手绘 OHLC 蜡烛图（含影线、实体、价格刻度、X 轴标签、hover tooltip），不依赖任何图表库
- **信号色响应系统**：`SIGNAL_MAP` / `RISK_LEVEL_MAP` 统一管理 BUY/SELL/HOLD 的中文名、CSS 类名、方向箭头
- **降级策略**：报告 `status === 'degraded'` 时显示黄色警告但不阻断渲染
- **错落入场动画**：report-section 的子元素按顺序 `fadeUp` 动画（CSS 配合 `nth-child`）

**Data Flow**
```
页面加载 → init()
  ├─ updateClock() — 每秒更新右上角时钟
  ├─ loadStockList() → GET /api/stock-list → 填充 autocomplete 数据源
  ├─ loadConfig() → GET /api/config → flatten → 渲染设置面板字段
  └─ bindEvents() / bindAutocomplete() / bindChartHover()

用户输入股票代码 → filterStocks() → renderSuggest() → 选择股票 → selectStock()
用户点击「生成报告」→ generateReport()
  ├─ 参数校验 (validateAllParams)
  ├─ POST /api/decision-report { stock_code, pred_len, temperature, top_p }
  ├─ 返回 DecisionReport JSON
  ├─ renderReport(report)
  │   ├─ signalCard — 大号信号文字 + 理由
  │   ├─ renderTrend — 方向箭头 + 强度进度条 + 上涨概率
  │   ├─ renderRisk — VaR 95% / 波动率 / 反转风险
  │   ├─ renderScoreDetail — 评分表格
  │   └─ renderChart — Canvas 蜡烛图
  └─ footer 填充生成时间/耗时

用户修改配置 → 保存 → PUT /api/config → flatten → unflatten → 提交
```

**Integration Points**
- `GET /api/stock-list` — 股票池数据（来自 `data.pool.get_hs300_pool()`）
- `POST /api/decision-report` — 决策引擎（来自 `decision.engine.DecisionEngine`）
- `GET/PUT /api/config` — 配置管理（来自 `decision.config`）
- Chart 数据格式：`prediction.pred_df` 支持列式（`{columns: [], data: []}`）、行式（`Array<Object>`）两种结构
