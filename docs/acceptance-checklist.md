# Kronos-Decision 验收清单

> 对照已批准实施计划逐条验收。状态标记：
> **已验证** = 本沙箱已有可复现证据（测试/运行产物/真实运行记录）；
> **部分验证** = 代码与测试就位但受数据/环境限制，仅在小样本或模拟上验证；
> **需真实环境** = 依赖完整 Qlib 数据、GPU 或浏览器人工操作。

## 阶段0：基线与规格固化

| 门禁 | 状态 | 证据 |
|---|:---:|---|
| CPU 测试可在新环境运行 | 已验证 | 默认离线套件 190 项通过、12 项 `network/model/gpu` 不选中；不会下载模型或访问网络 |
| v1 接口契约快照固定 | 已验证 | `tests/test_v1_compat.py`：DecisionReport 字段集合与 fetch_daily 契约断言；`/api/config` 对无效类型、范围、跨字段和未知字段在副本上拒绝，不污染内存配置 |
| WebUI 和配置可启动 | 已验证 | 真实 `python webui/run.py` 启动，/portfolio、/research、/api/v2/portfolio 200；`start.bat` 存在 |
| UTF-8 无 BOM 检查通过 | 已验证 | 全部新增/修改文件字节级检查无 BOM |

## 阶段1：可信预测与数据合同

| 门禁 | 状态 | 证据 |
|---|:---:|---|
| 相同输入和种子逐元素一致 | 已验证 | `tests/test_predictor.py::test_research_predictor_is_deterministic_across_runs`、`test_research_predictor_derives_seed_per_stock_and_cutoff`；真实 GPU `gpu-seed-collection-verify` 3 股对应 3 个派生采样种子 |
| 路径均值与旧预测在容差内一致 | 已验证 | `tests/test_kronos_regression.py::test_predict_paths_mean_matches_legacy_predict`（真实模型 1e-5 容差） |
| 周末、节假日、停牌和盘中数据测试通过 | 已验证 | `tests/test_qlib_data.py`（春节周锚点）、`tests/test_calendar.py`（Qlib 日历优先、节假日、盘中、过期降级、请求日会话解析：历史请求不继承当前日历过期、周末紧邻已知周五不误降级）、`tests/test_data_freshness.py`（完整日线、缺失完整会话硬失败、显式 as_of 不得保留最新完整会话之后的K线）、`tests/test_session_gaps.py`（缺失交易日硬失败、零成交不冒充确认停牌、备源恢复、历史校验忽略未来缺口与跳价） |
| 过期缓存返回 degraded 且动作强制证据不足 | 已验证 | `tests/test_v2_contract.py::test_stale_cache_or_path_flags_force_degraded_hold`（is_stale/路径质量标记 → HOLD + degraded）；`test_fetcher_cache_provenance.py` 验证新鲜缓存重校验、来源链保留与损坏缓存回退 |
| 真实单股 GPU 冒烟成功 | 已验证 | `gpu-full-data-verify`：Windows CUDA GPU、完整 Qlib 格式数据、100 条真实路径，预测/回填/TopkDropout/基线/前瞻全链路完成 |

## 阶段2：Qlib 最小纵向闭环

| 门禁 | 状态 | 证据 |
|---|:---:|---|
| 从命令启动到报告产物全链路可重放 | 已验证 | `full-mini-verify`：预测24/回填24/回测73交易日/门禁/随机游走/20日动量/锚点点时等权基线/前瞻/positions/trades/horizon_metrics 全落盘 |
| 任一锚点无法访问未来数据 | 已验证 | `QlibPointData.bars_at` 截至锚点 + `tests/test_evaluation_research.py` 未来数据注入必失败 |
| 两次运行哈希和指标一致 | 已验证 | `full-mid-verify` 重跑：metrics 逐位一致（仅 `evaluated_at` 更新），data_hash 相同 |
| 故意注入未来数据时测试必须失败 | 已验证 | `tests/test_evaluation_research.py::test_prediction_contract_rejects_future_data` |
| Qlib 成本前后结果有明确差异 | 已验证 | `tests/test_evaluation_research.py::test_cost_before_and_after_metrics_differ`；回测 report 含 cost 列且净额差异 |

## 阶段3：完整校准与证据门禁

| 门禁 | 状态 | 证据 |
|---|:---:|---|
| 覆盖率/ECE/RankIC/扣费超额/IR/回撤/稳定性可自动计算 | 已验证 | `full-mid-verify` manifest.metrics 全部真实计算（Bootstrap 0.6515 等）；5/20/60 独立 Conformal+Isotonic 档案由 `apply_multi_horizon_calibration_metrics` 计算，覆盖率/ECE 仅统计逐锚点历史拟合产生的严格样本外观测，20日 RankIC 不混入其他期限 |
| 沪深300全收益基准可审计接入 | 部分验证 | `scripts/import_total_return_benchmark.py` 将连续 CSV 原子注册为 `CSIH00300`，记录来源、内容哈希、来源 URL 与检索时间；只有受信 HTTPS 且 URL 含 `H00300` 的来源（`provenance_verified=true`）才可放行正式协议，任意来源标签只能作诊断数据。`test_total_return_benchmark.py`（含未核验来源失败闭合）与真实 Qlib 集成测试验证回测读取。尚未导入并核验覆盖正式区间的真实 H00300 数据 |
| 任意哈希不匹配时门禁失败 | 已验证 | `tests/test_binding.py::test_calibration_hash_mismatch_blocks_action`、`test_online_report_must_match_evaluation_versions`、`test_gate_fails_on_version_mismatch`；运行内 `VERSION_MATCH` 逐行校验模型/配置/数据并校验规则哈希 |
| 非正式 canary 永不放行交易动作 | 已验证 | `FORMAL_PROTOCOL` 硬门禁：还要求完整股票池、2017-01-01 起点、覆盖 Qlib 数据末端、沪深300全收益基准且非收集模式；价格指数只能用于研究展示。`test_nonformal_protocol_cannot_pass_evidence_gate`、`test_formal_protocol_rejects_partial_universe_history_and_price_benchmark` 覆盖 |
| 任一门禁失败均不输出增持/减持/回避 | 已验证 | `tests/test_binding.py`（门禁失败、5/60校准缺失或过期→INSUFFICIENT_EVIDENCE）、`tests/test_v2_contract.py`（永不映射 SELL） |
| 旧接口兼容映射符合约定 | 已验证 | `decision/v2.py::legacy_signal`（ADD→BUY、HOLD→HOLD、REDUCE/AVOID→SELL、不足→HOLD）+ 契约测试 |
| 标签重叠采用最长 60 日 purge/embargo | 已验证（设计说明） | 仅评估预训练模型、无训练/验证切分，标准 purge 无适用对象；点时协议 + 锚点聚合落实，见设计文档"标签重叠与 purge/embargo 的说明"；引入微调时必须独立窗口 + 60 日 purge/embargo |

## 阶段4：组合、持仓与 v2 WebUI

| 门禁 | 状态 | 证据 |
|---|:---:|---|
| 所有权重/行业/现金/换手约束满足误差容限 | 已验证 | `tests/test_portfolio_optimizer.py`（含池外 10% 上限、行业上限、现金下限、换手上限、佣金/印花税后现金约束） |
| 股数符合 A 股整数单位规则 | 已验证 | `build_simulated_orders` 100 股取整/零股整卖 + 测试 |
| 证据不足时不产生模拟买单 | 已验证 | 真实端点 POST /api/v2/portfolio/rebalance → 409 INSUFFICIENT_EVIDENCE |
| Flask API、页面渲染、Canvas/图表和错误状态 | 部分验证 | 本机 Flask 冒烟：报告/研究/评估/配置端点 200，无效配置 400 且内存保持原值；研究页展示基准来源，报告页展示逐项恢复条件。浏览器人工验收仍需真实环境 |
| 启动 WebUI 后完成一次真实股票报告和一次模拟调仓 | 部分验证 | 真实报告完成（`decision_report_v2("600519")` 437s，INSUFFICIENT_EVIDENCE 正确）；模拟调仓需门禁通过的正式评估（需真实环境） |

## 阶段5：研究扩展与前瞻验证

| 门禁 | 状态 | 证据 |
|---|:---:|---|
| 前瞻结果与历史回测分开报告 | 已验证 | `full-mid-verify` manifest.forward_validation 与 metrics 分开；`forward_report` disclaimer |
| 数据不可用时降级路径明确 | 已验证 | 三级降级 fetcher（真实验证：AkShare→BaoStock）；缺失应有完整会话时以实际 `as_of` 回退并标记 `ERROR_INCOMPLETE_LATEST_SESSION`；可选 Tushare 仅在配置与环境令牌同时存在时追加，离线验证令牌不回显；展示信息 20s 预算超时 |
| 新证据模块关闭后核心量化结果不变 | 已验证 | 市场/基本面/事件仅展示不进门禁（`data/market_context.py` 等）+ 测试 |
| 不因 LLM 或事件解释改变锁定动作 | 已验证 | 动作仅由 `decide_action` 结构化规则决定，无 LLM 调用 |

## 测试矩阵

| 类别 | 状态 | 证据 |
|---|:---:|---|
| 单元测试（路径/日历/期限/五态/门禁/约束/存储/配置） | 已验证 | 默认离线套件 190 项通过，真实环境 12 项显式标记 |
| 集成测试（固定 Qlib 数据集端到端） | 已验证 | `tests/test_qlib_integration.py`（真实 pyqlib + 小型数据回测） |
| 动态成分加入/退出、成交约束、费用滑点 | 已验证 | `tests/test_universe_import.py`、`test_evaluation_research.py`（exchange 契约） |
| 评估产物过期、模型版本变化、配置变化 | 已验证 | `tests/test_binding.py`（过期校准、哈希不匹配） |
| 真实环境验证（多源降级/CPU-GPU/30股/全量/浏览器） | 部分验证 | 沙箱内：BaoStock 降级、CPU 推理、GPU 完整 Qlib 格式 canary（3 股 × 3 周锚点 × 30 路径，27/27 回填、73 回测日、可续跑）、GPU 分片收集续跑（3 股、两次各 1 锚点，9→18 条预测、无回测/校准/动作）、2 股 smoke；正式全量、AkShare 主源与浏览器人工验收仍待完成 |

## 研究验收声明（不可省略）

必须同时报告：

- 已验证：自动测试与固定数据结果（本清单"已验证"项）；
- 部分验证：有限股票/有限时期的真实运行（`full-mini-verify`、`full-mid-verify`、`smoke-live-verify`、`gpu-full-data-verify`、`gpu-canary-3x3-202006`、`gpu-collection-one-anchor-verify`）；
- 未验证：尚未覆盖的市场状态或未来表现（任何历史回测/前瞻模拟均不构成投资有效性声明）。

## 明确不在首版范围（确认未实现）

券商登录/下单/撤单、实时盘口/高频、多用户/云托管/公开投顾、全 A 股横截面、
强制付费数据、自动模型微调/RD-Agent、Riskfolio-Lib、AKQuant 生产接入、
LLM 直接生成或修改量化动作——均未实现且无相关代码路径。
