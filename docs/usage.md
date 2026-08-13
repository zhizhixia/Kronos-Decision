# Kronos-Decision 使用说明

## 环境要求

- Python 3.10/3.11（本机验证版本 3.11）
- 推理建议使用 GPU；CPU 会自动把采样数从 100 降至 30 并标记 `reduced_for_cpu`。默认配置固定 Kronos 模型与 Tokenizer 的 Hugging Face 修订，`device: auto` 会优先使用可用 GPU。
- 正式研究（Qlib 回测）需要可安装 pyqlib 的环境；本机 Windows + CUDA GPU 已完成小规模全链路验证

## 安装

```powershell
pip install -r requirements.txt
# 研究依赖（pyqlib 等），Windows 临时目录权限异常时使用脚本：
powershell -ExecutionPolicy Bypass -File scripts\install_research_deps.ps1
```

### 可选 Tushare Pro 数据源

基础安装不会安装或调用 Tushare。需要它作为 AkShare、BaoStock 之后的可选降级源时，单独安装并仅通过环境变量提供令牌：

```powershell
pip install -r requirements-optional.txt
$env:TUSHARE_TOKEN = '你的令牌'
```

随后在 `decision/config.yaml` 的 `data.optional_sources` 中加入 `tushare`。默认保持空列表，因此免费数据链、缓存和既有报告行为不会改变；令牌缺失时 Tushare 不会加入实际 fallback chain，也不会写入缓存、日志或评估工件。

### 沙箱/vendor 运行方式（本仓库已验证路径）

Windows 沙箱拒绝写入 `tempfile.mkdtemp` 创建的 0o700 目录，导致 pip 解包失败。
仓库内的解决方案：

- `vendor/`：pyqlib 0.9.7 及其依赖（`--target` 安装，已 gitignore）；
- `sitecustomize.py`：把 `tempfile.mkdtemp` 改为 0o777 建目录，仅当仓库根在
  `PYTHONPATH` 中才生效，正常用户运行不受影响；
- `pytest.ini` 已配置 `pythonpath = vendor`。

运行前设置：

```powershell
# 在仓库根目录运行；vendor 路径相对仓库根解析
$env:PYTHONPATH = ".\vendor;."
$env:HF_HUB_OFFLINE='1'   # 使用本地 HuggingFace 缓存
```

模型权重从 HuggingFace 自动下载（`NeoQuasar/Kronos-small` 与 Tokenizer），
也可预先设置 `HF_HUB_OFFLINE=1` 使用本地缓存。

## 启动 WebUI

```powershell
python webui/run.py
# http://127.0.0.1:7070
```

页面：

- `/report`：v2 五态决策报告（主入口；采样参数固定，普通页面不可修改）
- `/report/legacy`：v1 兼容页面（可改采样参数，仅供对照）
- `/portfolio`：本地组合（SQLite）与模拟调仓
- `/research`：正式研究评估与门禁结果

## 数据准备

1. 实时数据：AkShare/BaoStock 自动降级，缓存于 `data/cache/`（CSV，24 小时有效）。每次读取缓存都会重新校验 OHLCV；缓存元数据保留原始来源、质量标记与实际 fallback chain，损坏缓存不会静默进入报告。
2. Qlib 中国数据（正式研究必需）：

```bash
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

3. 点时成分导入（正式研究必需）：

```bash
python -c "from data.universe_import import import_from_qlib; print(import_from_qlib())"
```

   导入器按每月最后一个交易日检查点枚举 csi300 成分，重建每只股票的有效区间
   （月度点时精度，离开再回归的股票保留多个区间），自动剔除基准指数与非法代码。

4. 行业映射（模拟调仓必需）：`data/cache/industry_map.csv`，列为
   `stock_code,industry`；缺失时调仓明确返回行业映射缺失，不静默兜底。

5. 沪深300全收益基准（正式建议必需）：优先从 AkShare 下载 `csiH00300`：

```powershell
python scripts/import_total_return_benchmark.py --download-akshare --provider-uri $env:KRONOS_QLIB_DATA
```

   若网络或免费源不可用，从已核验来源取得 `H00300` 日线 CSV（至少包含
   `date,close`，可选 `open,high,low`）并导入同一份 Qlib 日历：

```powershell
python scripts/import_total_return_benchmark.py --csv C:\path\to\h00300.csv --provider-uri $env:KRONOS_QLIB_DATA --source csindex_export --source-url "https://www.csindex.com.cn/indices/H00300"
```

   导入器会校验交易日连续性、OHLC 不变量，并以原子替换写入 `CSIH00300` 特征、
   有效区间、内容哈希和来源元数据。手工 CSV 必须提供受信 HTTPS 且 URL 包含 `H00300`
   的 `--source-url`，否则只能保存诊断数据，不得放行正式协议。只有这些工件完整覆盖
   `2011-01-01` 至评估截止日且来源校验通过时，正式协议才会使用全收益基准；缺失、截断、
   来源未核验时仍使用 `SH000300` 价格指数展示，并强制 `INSUFFICIENT_EVIDENCE`。

### 小型 Qlib 数据（集成验证用）

没有完整 Qlib 中国数据时，可从本地 CSV 缓存构建小型数据目录用于全链路验证：

```powershell
python scripts/build_mini_qlib.py --codes "600519,000001,600036"
$env:KRONOS_QLIB_DATA='C:\path\to\qlib_data\cn_data'
```

该目录含合成 `SH000300` 等权基准，`csi300.txt` 只含真实股票成员；回测参数
（topk/n_drop）会按股票数自动钳制。

## 运行评估

```powershell
# 真实数据冒烟（最新锚点、30 只代表股、真实采样）
python -m evaluation.run --stage smoke --as-of YYYY-MM-DD

# 完整正式评估（Qlib 点时数据、周锚点、回测、校准、门禁）
python -m evaluation.run --stage full --as-of YYYY-MM-DD

# 轻量 GPU canary（只验证链路，不可作为正式投资证据）
python -m evaluation.run --stage full --as-of 2020-06-19 --stocks "600519,000001,601318" --anchors-start 2020-06-01 --sample-count 30 --run-id gpu-canary-3x3-202006

# 长周期正式评估的分片收集：每次至多新增 N 个周锚点，不运行回测或生成动作
# 每次必须保持相同的 as-of、run-id、模型与配置；正式运行不要传 --stocks。
python -m evaluation.run --stage full --as-of YYYY-MM-DD --sample-count 100 --collect-only --max-new-anchors N --run-id formal-hs300

# 全部预测收集完成后，以相同 as-of/run-id 运行一次最终回测与门禁计算
python -m evaluation.run --stage full --as-of YYYY-MM-DD --sample-count 100 --run-id formal-hs300
```

可选参数：

- `--stocks "600519,000001"`：限定股票范围（与点时成分取交集，数据不足 60 根的成分自动跳过）
- `--sample-count N`：采样条数；CPU 超过 30 条自动降至 30 并标记 `reduced_for_cpu`
- `--anchors-start YYYY-MM-DD`：周锚点起点（正式协议 2017-01-01）
- `--anchor-frequency weekly|monthly`：默认 `weekly`；`monthly` 仅用于诊断 canary，强制不能通过正式证据门禁
- `--run-id NAME`：指定产物目录，便于断点续跑
- `--collect-only`：只收集并原子写入预测工件，强制 `INSUFFICIENT_EVIDENCE`，不生成回测、校准或调仓建议
- `--max-new-anchors N`：单次最多推理 N 个尚未完成的锚点批次，并自动启用 `--collect-only`；N 必须大于 0

产物位于 `artifacts/evaluations/<run_id>/`：`manifest.json`、
`predictions.csv`（5/20/60 三期限）、`horizon_metrics.csv`、`calibration.json`
（观测不足时按设计不生成并让门禁失败）、`backtest_daily.csv`、`positions.csv`、
`trades.csv`（相邻持仓快照差派生的模拟成交）、`gate_result.json`。
断点续跑命中 `prediction_key` 时跳过重复推理，复用已有工件。

### 工件与正式协议

`manifest.json` 使用严格 JSON（非有限指标写为 `null`），并记录规范化股票代码、锚点频率、采样数和可复制命令。每条预测还记录由 `model_hash|stock_code|last_visible_date|config_hash` 派生的 `sampling_seed` 与采样参数哈希，便于逐点重放；旧工件缺少审计列或模型、配置、数据哈希不匹配时会自动重新推理，而不会静默混合版本。最终运行按每行 5/20/60 日期限回填真实收益，`calibration.json` 使用 v2 多期限档案；覆盖率和 ECE 只用“每个锚点之前拟合的校准档案”产生的严格样本外观测，工件会记录各期限的样本外锚点数和观测数。任一行动所需期限缺少兼容、未过期校准时强制“证据不足”。在线报告还必须匹配评估的模型管线哈希、配置哈希、规则哈希、采样种子/参数和数据截止日。`benchmark_provenance` 会记录实际使用的基准、来源、覆盖范围、内容哈希和失败原因。只有最终完整运行同时满足“每周锚点、100 条真实路径、未发生 CPU 降级、未限制 `--stocks`、锚点起点精确为 2017-01-01、`as_of` 覆盖 Qlib 数据末端、非收集模式、沪深300全收益基准可用”时，才具有 `FORMAL_PROTOCOL` 资格；价格指数只可展示研究结果，不能放行建议。它仍须通过全部其他硬门禁后才能生成交易动作。分片收集工件可以断点续跑，但其门禁始终为失败闭合。

## 每日建议（可选）

```powershell
python scripts/daily_recommendations.py --stocks "600519,000001,600036"
```

对每只股票生成 v2 五态报告、保存建议历史并写出
`artifacts/daily/YYYY-MM-DD.csv`；可用 Windows 计划任务每日收盘后调用。

## 运行测试

```powershell
python -m pytest tests -p no:cacheprovider
```

默认套件会排除标记为 `network`、`model`、`gpu` 的真实环境测试，保证不下载模型或访问网络。本机验证为 190 项通过、12 项不选中。需要显式验证真实网络或模型时，覆盖默认筛选条件：

```powershell
python -m pytest tests -o "addopts=-p no:cacheprovider" -m "network or model"
```

## 信任边界

系统只生成研究建议，不连接券商、不自动下单、不保存交易凭据。任一硬门禁
未通过时，报告固定为 `INSUFFICIENT_EVIDENCE`（旧接口映射为 `HOLD`），
不产生模拟买单。市场状态、行业强弱、新闻公告等展示信息不进入量化决策。
