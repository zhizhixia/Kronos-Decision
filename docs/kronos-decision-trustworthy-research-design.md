# Kronos-Decision 可信研究设计

本设计覆盖 2026-07-10 旧设计中“不得修改 `model/`、不得增加依赖”的边界。新边界以真实采样、点时数据、Qlib 样本外研究和严格证据门禁为准。

系统只服务于单用户本地研究：不连接券商，不自动下单，不保存交易凭据。任何未通过证据门禁的报告都必须返回 `INSUFFICIENT_EVIDENCE`；旧三态接口只能映射为 `HOLD`。

所有缓存使用 CSV；评估工件位于 `artifacts/evaluations/<run_id>/`，其中清单、数据、模型、配置和随机种子必须可追溯。正式历史研究仅接受带有效区间的沪深300股票池和 Qlib 交易日历；工作日回退只能用于展示或冒烟，不能放行正式建议。

### 标签重叠与 purge/embargo 的说明

计划要求"标签重叠采用最长 60 日的 purge/embargo"。本系统的正式研究只评估
预训练 Kronos 模型、不训练不微调，因此不存在"训练/验证/测试"切分，
标准 purge/embargo（防止训练标签泄漏进验证窗口）没有适用对象。对应的
重叠处理落实为两点：

- 点时协议：每个锚点的预测与回填标签只使用锚点之后可见的价格（`bars_at`
  截至锚点、回填取锚点后第 N 个交易日收盘），不存在未来信息；
- 锚点聚合：RankIC/覆盖率等指标按锚点聚合（每组一个统计量），周锚点之间
  20 日收益窗口的重叠只影响观测独立性假设，不影响无偏性；若后续引入微调，
  必须按计划独立建立训练/验证/校准/测试窗口并实现 60 日 purge/embargo。

## 实施状态（2026-08-10）

已实现并通过 190 项离线测试（默认 CPU 套件；12 项真实 network/model/gpu 测试显式不选中，不下载模型或访问网络）：

- 真实采样 `predict_paths()`：100 条/20 条分批/确定性种子/OHLCVA 修复与质量标记；决策链已移除人工高斯路径。
- 数据合同：`MarketDataBundle` 来源链、完整交易日、新鲜度、内容哈希、原子 CSV；`as_of` 始终写实际最后一根可用K线，缺少应有完整会话时标记 `ERROR_INCOMPLETE_LATEST_SESSION` 并失败闭合；日历工作日回退明确标记。
- 点时股票池 `HistoricalUniverse` 与行业映射 `IndustryMap`（CSV 导入）。
- 评估工件 `EvaluationArtifacts`（manifest/预测/门禁/断点续跑）与完整硬门禁 `evaluate_evidence_gate`。
- 预测合同（未来数据注入测试必失败）、Qlib 适配层（下一交易日开盘、100股、涨跌停、费用、滑点、容量）、RankIC/Bootstrap/ECE/覆盖率/扣费组合指标。
- 滚动三年 Conformal + Isotonic 校准档案（24锚点/2000观测、30天过期、哈希绑定）。
- 五态动作与旧接口安全映射（证据不足永远 `HOLD`）。
- 组合优化：cvxpy 最大二次效用（现金/单股/行业/换手硬约束）+ HRP 回退 + L2/交易成本；100股买入/零股整卖模拟订单。
- 评估绑定：在线报告从最新评估运行读取横截面排名与校准概率；5/60 日证据缺失时保守否决增持。
- v2 API 与 WebUI 页面：五态报告页（真实路径图）、组合页（本地 SQLite）、研究页（可追溯 manifest）。
- Qlib 点时数据读取器：每周最后交易日锚点、点时成分、可见日线（`$close` 缺失即停牌剔除）。
- 基线比较模块：模型/随机游走/20日动量 RankIC、按每个锚点有效股票集合计算的20日点时等权收益、沪深300基准（价格指数仅展示不进门禁）。
- 每日建议历史：`recommendation_history` 保存与 `/api/v2/recommendations` 查询，供前瞻模拟与审计。
- 历史成员导入器 `data/universe_import.py`：从 Qlib csi300 instruments 生成带有效区间的点时成员表。
- v1 接口兼容快照：`DecisionReport` 字段集合与 `fetch_daily` DataFrame 契约固定。
- 前瞻模拟框架 `evaluation/forward_sim.py`：固定信号文件逐锚点 topk 建仓、20 日周期持有，结果与历史回测分开报告。
- 研究页展示基线比较与前瞻验证摘要（嵌套 manifest 字段渲染）。
- 市场状态与行业相对强弱 `data/market_context.py`：宽基趋势/波动状态、行业相对基准强弱；仅作展示信息，失败返回 unavailable 不阻塞报告，绝不进入量化门禁。
- 数据源超时保护：AkShare/BaoStock 单次请求 60 秒超时（独立线程），无超时的网络请求不再挂起 WebUI 或评估流程，超时自动降级到下一源。可选 Tushare Pro 仅在 `data.optional_sources` 显式启用且对应环境变量存在时追加到免费链之后；通过临时 `pro_api(token)` 调用，不持久化或回显令牌，尚未作真实令牌环境验证。
- 实际收益回填 `evaluation/backfill.py`：full 阶段用 Qlib 点时数据回填 `actual_return`（RankIC 与校准的前置），不足 20 根时返回 NaN 并计入 `backfilled_actuals` 统计。
- AKQuant 隔离对照 `evaluation/akquant_compare.py`：同一固定信号文件两引擎对比；未安装或差异无法解释时保持 Qlib，不讨论替换。
- 使用说明 `docs/usage.md`：安装、WebUI、数据准备、评估与测试命令、信任边界。
- 基本面与事件风险展示适配器：`data/fundamental.py`（历史日期无点时数据时隐藏数值并声明不进量化门禁）、`data/events.py`（公告/新闻仅作事件风险展示，不直接决定动作）。
- Qlib 成本前后指标差异测试与动态成分加入/退出测试（阶段2/3 集成门禁）。
- 质量审计修复：移除 `ResearchPredictor` 未使用的 `as_of` 参数与导入；采样元数据改为参数传递消除多线程竞态；`evaluation/__init__` 补全导出。
- 可重放性测试：同一输入两次生成逐行一致；CPU 采样降级（>30 条自动降至 30 并标记 `reduced_for_cpu`）有专项测试。
- WebUI 全端点验证：6 个页面 200、v2 API 正常、无评估时调仓 409 与参数错误 400 正确拒绝。
- smoke 全链路已在本机真实运行：`python -m evaluation.run --stage smoke --as-of 2026-08-07 --sample-count 5 --stocks "600519,000001"` 生成真实采样预测工件（6 行×3 期限），断点续跑命中缓存 3 秒完成且不重复追加。
- v2 报告端点 CPU 端到端验证：30 条真实路径（CPU 自动降级并标记 `reduced_for_cpu`），无正式评估时输出 `INSUFFICIENT_EVIDENCE` 与旧接口 `HOLD`，含 20 条展示路径。
- pyqlib 0.9.7 已装入仓库 `vendor/`（`--target` 安装 + 清华镜像，已 gitignore）；
  `sitecustomize.py` 修复沙箱 `tempfile.mkdtemp` 0o700 权限问题，仅仓库根在
  `PYTHONPATH` 时生效。
- Qlib 真实回测已在本沙箱小型数据上全链路跑通：
  `python scripts/build_mini_qlib.py --codes "600519,000001,600036"` 构建
  `qlib_data/cn_data`（含合成 SH000300 等权基准，csi300.txt 仅含真实股票），
  `python -m evaluation.run --stage full --as-of 2024-02-29 --anchors-start 2024-02-01
  --stocks "600519,000001" --sample-count 3 --run-id full-mini-verify` 完成
  预测→点时回填（24/24）→TopkDropout 回测（73 交易日、成本生效、topk/n_drop
  自动钳制）→扣费指标与完整门禁（mini 数据观测不足，覆盖率/ECE/IR 等 6 项按
  设计失败闭合）→基线比较→前瞻模拟（4 期，期末可见序列，与历史回测分开报告）
  →positions/trades/horizon_metrics 产物落盘；断点续跑复用已有预测 4.7 秒完成。
- full 阶段修复：`--stocks` 过滤与点时成分取交集、数据不足 60 根自动跳过、
  评估链 CPU 采样降级（>30 条→30）、基准指数从股票池排除并直读 `SH000300`、
  多期限 CSV 回测信号只取 20 日、`--sample-count/--anchors-start/--run-id` 生效、
  manifest 记录真实数据哈希、`_qlib_data_hash` 内容寻址。
- 点时成员导入改为月度检查点重建有效区间（离开再回归保留多个区间），
  测试改为依赖注入 D，不再污染 `sys.modules`（修复与真实 qlib 的日志配置冲突）。
- 中等规模全链路验证：`python -m evaluation.run --stage full --as-of 2024-04-30
  --anchors-start 2024-01-01 --stocks "600519,000001,600036" --sample-count 1
  --run-id full-mid-verify` 约 11 分钟完成（CPU）：17 个周锚点 × 3 只股票 =
  153 条预测全部点时回填，TopkDropout 回测 138 个交易日、12 笔模拟成交，
  扣费年化超额 +13.7%、IR 1.67、12 个月窗口正超额比例 1.0、Bootstrap
  RankIC 为正概率 0.65（真实计算值，不再是占位 0.0），前瞻模拟 17 期
  胜率 82%；mini 数据观测不足时覆盖率/ECE 按设计失败闭合。
- 哈希绑定测试补充：预测数据哈希与校准档案不一致时校准概率不可用、
  动作强制 `INSUFFICIENT_EVIDENCE`；`VERSION_MATCH` 门禁单独失败验证。
- 可重放性实证：`full-mid-verify` 断点续跑（预测全命中缓存，5.3 秒完成）
  与首轮指标逐位一致——扣费年化超额、IR、RankIC、Bootstrap 概率、回撤、
  数据哈希、成交/回测天数全部相同，仅 `evaluated_at` 按设计更新；
  对应阶段2门禁"两次运行哈希和指标一致"。
- 阶段1门禁补测：真实模型上 `predict_paths(1条,T=1.0,top_k=1)` 的均值路径
  与旧 `predict()` 输出在 1e-5 容差内一致。
- 股票资格规则 `data/eligibility.py`：ST/退市名称、上市不足 252 个交易日、
  长期停牌（最新成交日距锚点超 14 个自然日）、日均成交额低于 2000 万元
  均判定不可增持；成交额字段缺失时不误判。v2 报告携带 `eligibility`，
  资格阻断在门禁通过时输出 `AVOID`（reason_codes 含
  `TRADE_ELIGIBILITY_OR_RISK_BLOCK`），v1 契约快照保持不变。
- 组合优化新增池外总权重 10% 硬约束：不在点时评估股票池的持仓合计
  不得超过 10%（cvxpy 约束 + HRP 投影 + 校验三重保障），对应
  "池外股票总组合权重不得超过 10%"条款。
- 每日建议批量入口 `scripts/daily_recommendations.py`：对自选股票生成
  v2 五态报告、保存建议历史并原子写出 `artifacts/daily/YYYY-MM-DD.csv`，
  供 Windows 计划任务调用；只保存建议，不产生任何真实交易。
- 展示信息总预算：市场状态/行业强弱按计划"失败返回 unavailable 不阻塞报告"
  落实为 20 秒总预算（daemon 线程，超时即返回，后台重试链不再拖住报告或
  CLI 进程退出）；新增超时预算测试。
- 过期缓存/路径质量标记强制降级测试：is_stale 或 PATH_REPAIR_EXCESSIVE
  时旧三态动作强制 HOLD 且状态 degraded，正常路径不受影响。
- 验收清单 `docs/acceptance-checklist.md`：计划六阶段门禁、测试矩阵、
  研究验收声明与"不在首版范围"逐条映射证据与状态，供真实环境对照验收。
- 超时保护统一收口 `data/timeout.py::call_with_timeout`（daemon 线程）：
  数据获取与股票池的 AkShare 调用全部走同一超时工具；修复股票池拉取
  无超时挂起（曾导致每日建议脚本卡死）与 ThreadPoolExecutor 非 daemon
  线程拖住进程退出的问题；新增超时工具与股票池超时降级测试。

### 每日建议脚本真实运行（2026-08-10）

`scripts/daily_recommendations.py --stocks "600519"` 真实运行 439.7 秒完成
（30 条采样 CPU 推理），CSV 输出 `artifacts/daily/verify-20260810.csv`
（INSUFFICIENT_EVIDENCE/HOLD/eligible），建议历史写入 SQLite
`recommendation_history`（5/20/60 期限完整 horizons 与 no-evidence 版本标记），
进程干净退出；对应阶段5"每日保存建议"条款的真实验证。

### 本沙箱真实环境验证（2026-08-10）

- 真实数据冒烟 `--stage smoke --as-of 2026-08-07 --stocks "600519,000001"`
  74 秒完成：AkShare 在本沙箱不可达，自动降级 BaoStock 成功（缓存 meta
  source=baostock），6 条真实采样预测（3 期限 × 2 股）带真实分位数落盘。
- 真实 WebUI 启动：`python webui/run.py` 后台启动后 `/portfolio`、
  `/research`、`/api/v2/portfolio` 均返回 200，进程正常停止。
- 真实 v2 股票报告端到端：`decision_report_v2("600519")` 437 秒完成
  （30 条采样 CPU 自动降级标记 `reduced_for_cpu`），资格规则在真实数据上
  返回 eligible、市场上下文 20 秒预算生效、无正式评估时输出
  `INSUFFICIENT_EVIDENCE`/`HOLD`，均符合设计。
- 一键启动脚本 `start.bat` 修复：Python 自动探测（项目 .venv → Hermes venv →
  PATH），依赖检查清单同步（flask/plotly/cvxpy/pypfopt/sklearn），浏览器延迟
  3 秒等服务器就绪；全部 ASCII 无 Unicode。
- 超时工具统一后冒烟回归：`--stage smoke --as-of 2026-08-07
  --sample-count 1 --stocks "600519,000001"` 25.6 秒完成（缓存命中），
  fetcher/pool 超时改动无回归。

### 轻量 GPU canary 与工件审计修复（2026-08-10）

- 已下载并验证 Qlib 中国日频全量格式数据（数据截止 2020-09-25），用 Windows CUDA GPU 运行 `gpu-full-data-verify`：1 股 × 2 周锚点 × 100 条真实路径，5 条三期限预测全部回填，Qlib TopkDropout 回测 59 个交易日、基线和前瞻工件完整生成。
- `gpu-canary-3x3-202006`：3 股 × 3 周锚点 × 30 条真实路径，27/27 回填、73 个交易日回测、基线和前瞻工件完整生成；二次运行命中缓存且 `predictions.csv` SHA-256 不变。此运行明确为诊断 canary，不能构成投资有效性结论。
- 正式门禁新增 `FORMAL_PROTOCOL`：除每周锚点、100 条真实路径且未发生 CPU 降级外，还必须使用完整股票池、精确的 2017-01-01 样本外起点、覆盖 Qlib 数据末端且不是收集模式；月度、低采样、限股票或历史截断 canary 强制 `INSUFFICIENT_EVIDENCE`。
- `manifest.json` 现使用严格 JSON（NaN/Infinity 转为 null），记录规范化股票池、锚点频率、采样数与可复制 PowerShell 命令；修复未加引号时股票代码前导零可能丢失的问题。
- 组合协方差改用已锁定的 scikit-learn Ledoit-Wolf（年化），保留 PyPortfolioOpt HRP 回退；解决 Qlib 初始化后 PyPortfolioOpt 软依赖元数据扫描受损环境的跨测试问题。
- 新增 `--collect-only --max-new-anchors N`：正式大规模研究可按同一 `as_of/run_id` 分批收集预测，期间只落盘 `predictions.csv` 且门禁强制失败闭合；最后一次不带收集参数的完整运行才回填、回测、校准和评估门禁。`gpu-collection-one-anchor-verify` 已在 GPU 上连续运行两次，工件从 9 条扩展到 18 条三期限预测、覆盖 2017-01-06/2017-01-13 两个锚点且无重复键，未生成回测或校准工件。
- 断点续跑合并修复：已有检查点后产生新预测时，CSV 按 `prediction_key + horizon` 合并并保留已有实际回填字段；同时在内存中缓存已有键，避免每只股票重复扫描完整 CSV。
- 采样种子改为逐点派生：默认 `sha256(model_hash|stock_code|last_visible_date|config_hash)`，每条预测写入 `sampling_seed` 和 `sampling_params_hash`。GPU `gpu-seed-collection-verify` 的 3 只股票验证为 3 个不同且可审计的种子；运行级 `manifest.seed` 继续只用于 Bootstrap 等研究统计随机性。
- 多期限校准闭环补齐：实际收益回填按每行 5/20/60 日期限执行，并按“股票×锚点”复用价格读取；`calibration.json` v2 分别持久化三期限 Conformal+Isotonic 档案。在线绑定仅使用对应期限、哈希兼容且未过期的概率和区间；5/60 档案缺失或过期会将动作强制为 `INSUFFICIENT_EVIDENCE`，不能以原始概率绕过否决。`full-mini-verify` 已在 GPU 上重建：24 条预测均有采样审计字段，5/20/60 各 8 条真实收益回填；因样本不足，v2 工件正确写入空档案并保持门禁失败闭合。
- RankIC 门禁明确只消费20日主决策信号，5/60日标签不再混入横截面相关性统计；预测合同允许同一路径的三个期限共享 `prediction_key`，但拒绝重复的 `prediction_key + horizon`。
- 版本可审计性补齐：模型与 Tokenizer 固定到配置中的 Hugging Face 提交修订，运行清单写入模型管线、配置和五态规则哈希；评估阶段逐行核验模型/配置/数据哈希，在线报告还核验模型管线、采样种子、采样参数、规则及数据截止日。旧检查点中不兼容版本的行会被过滤，不能与新路径组成重复 Qlib 信号。`full-mini-verify` 的真实 GPU 重放已验证 `VERSION_MATCH=True`，同时因非正式协议保持失败闭合。

### 可信性审计修复（2026-08-10）

- 校准诊断改为严格滚动样本外：每个锚点只能使用此前三年的预测与真实标签拟合 Conformal/Isotonic，覆盖率和 ECE 只统计该档案对该锚点的后验预测；每个期限在工件和研究页显示样本外锚点数与观测数。样本不足时保持不可用而非将同一批数据的拟合指标当作有效证据。
- CSV 缓存每次读取都会执行 OHLCV 质量校验；缓存元数据保留原始来源、质量标记和 fallback chain，损坏、截断或无法解析的新鲜缓存会被拒绝并继续尝试数据源。股票池缓存也采用临时文件原子替换。
- 在线日线完整性改为优先读取 `KRONOS_QLIB_DATA/calendars/day.txt`；节假日、盘中 15:30 前和后续预测日期均使用真实交易日。日历缺失或落后当前候选日时才生成工作日候选，并写入 `CALENDAR_WEEKDAY_FALLBACK` 或 `CALENDAR_QLIB_OUTDATED`，从而使门禁失败闭合。
- 配置加载、热重载和保存统一校验类型、范围和跨字段约束；`/api/config` 先修改深拷贝并验证，再以原子替换保存，因此无效采样、超时、缓存、端口或数据源配置不会污染正在运行的实例。
- 评估绑定与 `/api/v2/evaluation/latest` 现在优先最新正式运行；新的 smoke/canary 仅在没有正式运行时作为诊断回退，不能遮蔽可追溯的正式证据。报告页会按失败门禁输出恢复条件，研究页同时展示全收益基准的来源、覆盖和不可用原因。
- 模拟调仓订单现逐笔披露名义金额、佣金、卖出印花税、估算费用与现金影响；买单在费用后仍必须满足现金和 100 股整手约束，已有持仓缺少有效价格时失败闭合。
- `FORMAL_PROTOCOL` 现额外要求沪深300全收益基准。`scripts/import_total_return_benchmark.py` 可把经核验、交易日连续的 `H00300` CSV 原子注册为 `CSIH00300`，并保存来源、覆盖范围、内容哈希、来源 URL 与检索时间；只有来源 URL 为受信 HTTPS 主机且包含 `H00300`（`provenance_verified=true`）才可放行正式协议，任意来源标签的 CSV 只能作为诊断数据。运行清单仅在该元数据完整覆盖评估区间时标记 `benchmark_is_total_return=true`。当前未导入真实全收益数据时仍使用 `SH000300` 价格指数做展示，轻量 `full-mini-benchmark-resolver-verify`（24 条预测、73 个回测日）已验证该条件令正式协议与总门禁同时失败；真实 H00300 导入后的长周期验证仍待完成。
- 在线数据质量会用本地 Qlib 日历核对首末观测日之间的预期交易日：缺失记录标记 `ERROR_UNRESOLVED_SESSION_GAP` 并阻断正式建议；零成交记录仅标记 `POSSIBLE_SUSPENSION_ZERO_VOLUME`，绝不作为已确认停牌。备源能补齐缺口时优先返回完整来源；没有可核验停牌状态源时保留不确定性。
- 默认离线回归在本机为 190 通过、12 项真实网络/模型/GPU 测试显式不选中；研究页已展示全收益基准状态与三期限严格样本外校准质量。

尚未完成（需要长周期运行或人工环境验收）：

- 已有截止 2020-09-25 的完整 Qlib 中国日频格式数据与历史点时沪深300成员表，但全量正式运行尚未完成：每周锚点 × 数百成分在 GPU 上仍需长周期计算。现在可用 `--collect-only --max-new-anchors N` 分批续跑，待全部预测齐备后再执行一次最终正式回测。
- 校准档案的真实拟合（24 锚点/2000 观测）与完整门禁通过/失败的真实统计，需在
  全量数据上验证；mini 数据按设计失败闭合。
- 前瞻模拟的 20 日实时周期跟踪与 AKQuant 隔离对照（同一固定信号文件）未实盘运行。
- WebUI 浏览器人工验收（页面渲染、Canvas/图表交互）与"主源成功"路径
  （AkShare 在本沙箱不可达，仅验证了备源与缓存路径）。

任何未通过门禁的在线报告都返回 `INSUFFICIENT_EVIDENCE`，不会生成模拟买单；模拟调仓端点仅在评估工件门禁通过后可用。
