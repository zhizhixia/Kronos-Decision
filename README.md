<div align="center">
  <h2><b>Kronos-Decision</b></h2>
  <p>面向 A 股研究用途的 Kronos 增强决策工具链</p>
</div>

<div align="center">

<a href="https://arxiv.org/abs/2508.02739">
<img src="https://img.shields.io/badge/arXiv-2508.02739-brightgreen" alt="Paper">
</a>
<a href="https://github.com/shiyu-coder/Kronos">
<img src="https://img.shields.io/badge/upstream-Kronos-blue" alt="Upstream">
</a>
<a href="./LICENSE">
<img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</a>

</div>

<p align="center">
<img src="./figures/logo.png" width="100">
</p>

---

## 1. 项目定位

Kronos-Decision 是在 [Kronos](https://github.com/shiyu-coder/Kronos)（AAAI 2026，论文
[Kronos: A Foundation Model for the Language of Financial Markets](https://arxiv.org/abs/2508.02739)）
之上扩展的本地研究工具，提供 A 股日 K 数据获取、Kronos 多路径预测、可信研究评估与
本地组合模拟。核心模型与 Tokenizer 来自上游仓库，本项目仅在其上构建决策、评估与界面层。

> 上游作者：Yu Shi、Zongliang Fu、Shuo Chen、Bohan Zhao、Wei Xu、Changshui Zhang、Jian Li。
> 本仓库保留上游 MIT 许可证，并致谢原项目作者公开发布模型与代码。

## 2. 风险边界（务必先读）

- 本系统是**本地研究工具**，不构成任何投资建议；
- **不连接券商**，**不自动下单**，**不保存任何交易凭据或密钥**（见
  `portfolio/store.py` 与 `webui/app.py` 的写策略）；
- 证据门禁失败时，报告固定返回 `INSUFFICIENT_EVIDENCE`（旧接口映射为 `HOLD`），
  只提示“证据不足”，不产生模拟买单或调仓（见 `decision/v2.py`、
  `portfolio/service.py`）；
- 决策动作只由结构化规则决定，市场状态、行业强弱、新闻公告等展示信息不进入量化决策
  （见 [docs/usage.md](docs/usage.md) 信任边界章节）。

## 3. 当前功能

| 能力 | 说明 | 代码位置 |
|------|------|----------|
| 真实多路径预测 | Kronos 模型真实采样，CPU 自动降至 10 条并标记 `reduced_for_cpu` | `decision/engine.py`、`decision/model_manager.py` |
| 数据三级降级 | 新鲜 CSV 缓存优先命中，否则 AkShare → BaoStock → 可选 Tushare，最后过期 CSV 缓存；成功的任一网络源都会原子写 CSV 与 meta | `data/fetcher.py` |
| 点时股票池 | 沪深 300 每月最后一个交易日检查点成分枚举，重建有效区间 | `data/universe.py`、`data/universe_import.py` |
| Qlib 评估 | 点时数据回测、回填、指标、校准与门禁工件 | `evaluation/run.py` |
| 校准与门禁 | 5/20/60 日独立校准档案，20 日主门禁；13 项硬门禁（见下） | `evaluation/calibration.py`、`evaluation/gates.py` |
| 五态动作 | `ADD` / `HOLD` / `REDUCE` / `AVOID` / `INSUFFICIENT_EVIDENCE`，20 日主、5/60 否决 | `decision/v2.py` |
| 组合模拟 | PyPortfolioOpt + HRP 回退的硬约束优化与 A 股模拟订单取整 | `portfolio/optimizer.py`、`portfolio/service.py` |
| v2 WebUI 四页 | 决策报告 / 组合 / 研究 / 系统诊断（外加 v1 兼容页） | `webui/app.py`、`webui/templates/` |

门禁检查项（`evaluation/gates.py`）：`DATA_COMPLETE`、`NO_QUALITY_ERROR`、
`INTERVAL_COVERAGE`（覆盖率落入 0.75–0.85）、`PROBABILITY_ECE`（≤0.08）、`RANK_IC`（>0）、
`BOOTSTRAP_RANK_IC`（≥0.80）、`NET_EXCESS_RETURN`（>0）、`INFORMATION_RATIO`（≥0.5）、
`WINDOW_STABILITY`（≥0.60）、`DRAWDOWN`（≤0.05）、`VERSION_MATCH`、`FORMAL_PROTOCOL`、
`FRESH_ARTIFACT`（30 日内）。任一失败即 `INSUFFICIENT_EVIDENCE`。

## 4. 实现状态

| 状态 | 项目 |
|------|------|
| 已落地（工程功能） | 决策引擎、五态动作、数据三级降级、点时成分、Qlib 评估管线、校准、门禁、组合优化、v2 WebUI 四页、SQLite 本地持仓、写接口本机来源校验 |
| 已验证（小规模） | 本机 Windows + CUDA GPU 完成小规模全链路验证；离线回归套件已验证通过，网络/模型/GPU 相关测试按 `network`/`model`/`gpu` 标记单独分开运行（见 [docs/usage.md](docs/usage.md)） |
| 尚未完成 | 正式完整沪深 300 回测（周锚点 2017-01-01 起、100 条真实路径、全收益基准、未限制 `--stocks`）尚未跑通；20 日以上前瞻验证尚未完成 |
| 未宣称 | 不宣称投资有效；不宣称全量研究已通过；门禁未通过即不得产生交易动作 |

> `--collect-only`、`--max-new-anchors`、`monthly` 锚点或单一 `--stocks` 均强制
> `FORMAL_PROTOCOL=false`，结果只作研究展示。

## 5. 安装与启动（Windows / Python 3.10/3.11）

一键启动（`start.bat` 会自动定位 Python 3.10/3.11、检查依赖、启动 WebUI）：

```powershell
.\start.bat
```

手动启动：

```powershell
pip install -r requirements.txt
python webui/run.py
```

默认地址：<http://127.0.0.1:7070/report>（根路径 `/` 自动重定向到 `/report`）。

> `start.bat` 仅接受 Python 3.10/3.11，否则报错退出。脚本按以下顺序定位解释器并逐个
> 实际校验版本：① 项目 `.venv`；② `py -3.11` / `py -3.10`；③
> `%LOCALAPPDATA%\Programs\Python\Python311` / `Python310`；④ PATH 中的 `python`。
> 仍未找到时，若本机已装 [uv](https://docs.astral.sh/uv/)，脚本会用
> `uv python find ... --no-python-downloads` 确认本机已有解释器，再用
> `uv venv --offline --python 3.11 --no-python-downloads --seed` 在项目根离线创建隔离
> `.venv`（`--offline` 保证连 seed 包也不联网，全程不下载 Python，不使用
> `--break-system-packages`，也不直接选用
> AppData 下 uv 的裸 Python）。若本机连一个 3.10/3.11 解释器都没有，先手工
> `uv python install 3.11` 再重跑 `start.bat`。
> 设置环境变量 `KRONOS_STARTUP_CHECK_ONLY=1` 可只做启动自检不拉起服务；此模式下若
> 缺依赖只给出安装命令并以非零码退出，不联网安装、不启动服务。

## 6. 可选研究依赖与环境变量

`requirements.txt` 包含主程序与正式研究底座（pyqlib、PyPortfolioOpt、scikit-learn、scipy；
cvxpy 由组合依赖安装），但在 Windows 上 pyqlib 等重依赖安装可能失败。此时用仓库脚本安装到 `vendor/`：

```powershell
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts\install_research_deps.ps1
```

可选付费数据 Tushare Pro（`requirements-optional.txt` 仅含 Tushare，基础运行不安装；
作为 AkShare/BaoStock 之后的可选降级源）：

```powershell
pip install -r requirements-optional.txt
$env:TUSHARE_TOKEN = '你的令牌'
```

随后在 `decision/config.yaml` 的 `data.optional_sources` 中加入 `tushare`；令牌缺失时
Tushare 不会进入实际 fallback chain，也不写入缓存或评估工件（见
`data/fetcher.py`）。

常用环境变量：

| 变量 | 作用 |
|------|------|
| `KRONOS_QLIB_DATA` | 指定 Qlib 中国数据目录（`evaluation/run.py`） |
| `TUSHARE_TOKEN` | 可选 Tushare Pro 令牌 |
| `HF_HUB_OFFLINE` | 设为 `1` 使用本地 HuggingFace 缓存 |
| `PYTHONPATH` | 沙箱运行需加入 `vendor` 与仓库根（见 [docs/usage.md](docs/usage.md)） |
| `KRONOS_STARTUP_CHECK_ONLY` | 设为 `1` 时 `start.bat` 只做自检，缺依赖则给命令并非零退出，不联网安装、不启动服务 |

> 不要把根 `requirements.txt` 解读为“一定能在所有 Windows 环境装好研究重依赖”——
> pyqlib/cvxpy 在部分 Windows 环境需走 `vendor/` 旁路安装。

## 7. 四个页面使用说明

| 页面 | 路径 | 用途 |
|------|------|------|
| 决策报告 | `/report` | v2 五态决策主入口；采样参数固定，网页不可覆写；含数据/模型/证据来源与门禁结果 |
| v1 兼容 | `/report/legacy` | 允许修改采样参数，仅供旧行为对照（已挂迁移提示横幅） |
| 组合 | `/portfolio` | 本地 SQLite 持仓与模拟调仓；不提供下单控件 |
| 研究 | `/research` | 正式研究评估清单、回测、校准与门禁结果 |
| 系统诊断 | `/settings` | 只读展示配置路径、模型/数据/证据快照；外部改配置需重启服务，且已有证据可能因哈希变化失效 |

## 8. Smoke 与 Full 评估及工件目录

```powershell
# 真实数据冒烟：最新锚点、30 只代表股、真实采样
python -m evaluation.run --stage smoke --as-of YYYY-MM-DD

# 完整正式评估：周锚点、100 条路径、回测、校准、门禁
python -m evaluation.run --stage full --as-of YYYY-MM-DD

# 分片收集预测（断点续跑，强制 INSUFFICIENT_EVIDENCE，不生成回测/动作）
python -m evaluation.run --stage full --as-of YYYY-MM-DD --collect-only --max-new-anchors N --run-id formal-hs300

# 全部预测收集完成后做一次最终回测与门禁计算
python -m evaluation.run --stage full --as-of YYYY-MM-DD --sample-count 100 --run-id formal-hs300
```

可选参数：`--stocks`（限定股票，与点时成分取交集）、`--sample-count`、
`--anchors-start`（默认 `2017-01-01`）、`--anchor-frequency weekly|monthly`、
`--run-id`。CPU 设备且请求数超过 10 条时自动降至 10 并标记 `reduced_for_cpu`。

工件位于 `artifacts/evaluations/<run_id>/`：`manifest.json`、`predictions.csv`
（5/20/60 三期限）、`horizon_metrics.csv`、`calibration.json`、
`backtest_daily.csv`、`positions.csv`、`trades.csv`、`gate_result.json`。
断点续跑按 `prediction_key` 跳过已完成推理。详见 [docs/usage.md](docs/usage.md)。

## 9. v2 API 简表

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/v2/decision-report` | 固定采样的五态报告；同时写入建议历史 |
| GET | `/api/v2/evaluation/latest` | 最近一次评估清单与门禁结果 |
| GET | `/api/v2/forward-ledger/latest` | 最新每日建议前瞻台账 |
| GET/PUT | `/api/v2/portfolio` | 读取或更新本地 SQLite 持仓 |
| POST | `/api/v2/portfolio/rebalance` | 门禁通过时生成模拟目标权重与订单；永不执行真实交易 |
| GET | `/api/v2/recommendations` | 本地建议历史 |
| GET/PUT | `/api/config` | 读取或更新 `decision/config.yaml` |
| POST | `/api/decision-report` | v1 兼容报告 |
| GET | `/api/stock-list` | 沪深 300 成分股列表 |
| GET | `/api/available-models` | 可用模型（mini/small/base） |
| POST | `/api/load-model` | 动态切换模型与设备 |

写接口（POST/PUT/PATCH/DELETE）会拒绝非本机来源（`webui/app.py`）。

## 10. 目录结构

```
kronos/
├── model/                # 核心：KronosTokenizer / Kronos / KronosPredictor（上游）
├── decision/             # 决策信号层（engine / analyzer / v2 / config / model_manager / versioning / errors）
├── data/                 # 数据管道（fetcher / pool / universe / eligibility / calendar / industry）
├── evaluation/           # Qlib 正式研究（run / qlib_data / qlib_adapter / calibration / gates / metrics / binding / forward_ledger）
├── portfolio/            # 本地组合（store SQLite / optimizer HRP / service 模拟调仓）
├── webui/                # Flask 界面（app.py / run.py / templates / static）
├── finetune/             # Qlib 微调流水线（上游）
├── finetune_csv/         # CSV 微调流水线（上游）
├── examples/             # 预测示例脚本（上游）
├── tests/                # pytest 回归与决策系统测试
├── scripts/              # 工具脚本（mini qlib / 基准导入 / 每日建议 / 研究依赖安装）
├── docs/                 # 设计文档与使用说明
├── figures/              # 上游 logo 与示例图
├── start.bat             # Windows 一键启动脚本
└── requirements*.txt     # 依赖清单
```

## 11. 已验证 / 部分验证 / 未验证

- **已验证**：小规模全链路（本机 Windows + CUDA GPU）；离线回归套件已验证通过，
  网络/模型/GPU 相关测试按 `network`/`model`/`gpu` 标记单独分开运行，未一并执行；
  数据三级降级与缓存元数据来源记录；五态动作规则与门禁逻辑（单元测试覆盖）。
- **部分验证**：Qlib 点时成分导入与回测在小型数据集上跑通；全收益基准导入脚本可工作
  但需可用的 `H00300` 全收益序列。
- **未验证**：正式完整沪深 300 全收益回测；20 日以上前瞻验证；多边界 GPU canary 仍属诊断
  不可作为投资证据。

## 12. 常见问题

- **Python not found**：`start.bat` 按以下顺序定位 Python 3.10/3.11，并对每个候选
  实际校验版本（错误版本不会阻断后续候选）：
  ① 项目 `.venv`（仅当为 3.10/3.11）；
  ② `py -3.11` / `py -3.10`（仅当存在 py launcher）；
  ③ `%LOCALAPPDATA%\Programs\Python\Python311` / `Python310`；
  ④ PATH 中的 `python`；
  均未命中时，若本机已装 [uv](https://docs.astral.sh/uv/)，脚本用
  `uv python find 3.11 --no-python-downloads`（失败再 3.10）确认本机已有解释器，
  随后用 `uv venv --offline --python 3.11 --no-python-downloads --seed` 在项目根离线创建 `.venv`
  并复用（失败再 3.10）。`--offline` 保证创建 `.venv` 时 seed 包也不联网。
  仍然失败则报错退出。修复方式：
  (a) 从 [python.org](https://www.python.org) 安装 Python 3.11，并在安装界面勾选
      “Add Python to PATH”；
  (b) 若已装 uv 但本机还没有 3.10/3.11 解释器，先手工运行
      `uv python install 3.11`，再重跑 `start.bat`。
  脚本绝不直接选用 AppData 下 uv 的裸 Python（否则会触发
  `externally-managed-environment`），也不用 `--break-system-packages`，也不自动下载 Python。
- **Python version must be 3.10 or 3.11**：`start.bat` 严格校验版本，其他版本会拒绝。
- **CPU 推理很慢**：单次多路径采样在 CPU 上耗时较长；配置自动把 `sample_count` 降至 10 并
  标记 `reduced_for_cpu`，且该次运行不满足 `FORMAL_PROTOCOL`。
- **首次模型下载**：首次运行会从 HuggingFace 下载 `NeoQuasar/Kronos-small` 与
  `Kronos-Tokenizer-base` 权重；可设 `HF_HUB_OFFLINE=1` 使用本地缓存。
- **数据源降级**：新鲜 CSV 缓存优先命中；否则 AkShare 失败自动降级 BaoStock（可选
  Tushare 经配置加入），再失败用过期缓存并标记 `STALE_CACHE`；缓存元数据保留来源与
  fallback chain，损坏缓存不会静默进入报告。
- **证据不足**：报告显示 `INSUFFICIENT_EVIDENCE` 表示任一硬门禁未通过，属设计行为；
  请检查 `artifacts/evaluations/<run_id>/gate_result.json` 的失败项。
- **pyqlib 安装失败**：Windows 沙箱拒绝临时目录权限，走 `scripts/install_research_deps.ps1`
  安装到 `vendor/` 并设 `PYTHONPATH`（见 [docs/usage.md](docs/usage.md)）。
- **端口被占用**：默认 7070（`decision/config.yaml`），改 `webui.port` 后重启服务。

## 13. 路线图、文档、许可证与引用

短期：补齐完整沪深 300 全收益回测与 20 日以上前瞻验证；在线档案滚动窗口自动化研究。
长期：把校准档案与门禁阈值变动纳入版本化证据契约。

文档：

- 设计文档：[docs/kronos-decision-trustworthy-research-design.md](docs/kronos-decision-trustworthy-research-design.md)
- 使用说明：[docs/usage.md](docs/usage.md)
- 验收清单：[docs/acceptance-checklist.md](docs/acceptance-checklist.md)
- 各子目录 `codemap.md`

许可证：MIT（见 [LICENSE](LICENSE)，继承上游）。

引用 Kronos 原论文：

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