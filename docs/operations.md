# Kronos Decision 操作说明

## 当前边界

本项目当前交付的是**可审计的离线研究版**，不是自动交易系统：

- 不连接券商、不下单、不采购行情、不下载模型；
- `mode=baseline` 只生成 5/10/20 日历史事实基线，`action_permission=NONE`；
- 真实行情、真实模型、真实样本外前瞻和正式策略资格仍需单独授权与验收；
- 不可变快照 ID 同时绑定内容哈希和不可变来源/质量元数据；同样内容但来源或质量状态不同，会形成不同版本，不能复用旧版本的来源说明。
- 任何缺失、过期、损坏或无法证明时点的数据都会保持失败闭合。
- 开盘执行回测必须提供 `execution_volume` 或 `open_volume`；日线收盘后的 `volume` 不用于限制开盘成交，避免把未来全天成交量倒灌到订单决策。

## 启动与检查

在 Conda 环境已具备依赖的前提下：

```bash
conda run -n kronos python webui/run.py
```

只做启动条件检查，不启动长期服务：

```bash
KRONOS_STARTUP_CHECK_ONLY=1 conda run -n kronos python webui/run.py
```

Windows 入口为项目根目录下的 `start.bat`。启动器只检查 Python 和依赖，不自动安装依赖、解释器、模型或数据。

## 模型启用边界

`decision/config.yaml` 中的 `model.allow_model_download` 默认是 `false`。因此：

- 没有本地模型权重时，调用模型管理器会失败闭合，不会隐式联网下载；
- `ModelManager.lease()` 是长任务的模型使用边界，活跃租约期间不会释放模型；
- 空闲超时释放只回收不再使用的模型，失败会保留错误状态，不把模型伪装成可用；
- 只有在获得单独授权、准备好固定版本权重并完成真实设备验收后，才允许显式启用模型下载配置；
- 模型缺失不能阻塞无模型 `baseline` 研究，也不能把基线结果表述成 Kronos 模型验证。

## 报告流程

新报告必须经过同一条链路：

```text
DataFetcher
→ 不可变 CSV 快照与 manifest
→ DecisionEngine 或无模型基线
→ SQLite 待发布报告
→ 快照引用保护
→ PUBLISHED
→ API/页面读取
```

推荐先用无模型基线验证工程链路：

```http
POST /api/v2/decision-report
Content-Type: application/json
Origin: http://127.0.0.1:7070

{"stock_code":"600519","mode":"baseline","as_of":"2024-01-02"}
```

响应中的 `report_id`、`report_version` 和 `snapshot_id` 是后续读回所需的引用。历史读回只读 SQLite 和原始快照，不重新取数、不重新运行模型：

```http
GET /api/v2/decision-report/<report_id>
```

如果报告或快照校验失败，接口返回 `REPORT_REPLAY_UNAVAILABLE`，不得把错误恢复成 BUY/HOLD/SELL。

## 持仓快照

持仓 CSV 先预览，预览不会写入正式账本：

```http
POST /api/v2/portfolio/preview
{"csv_content":"code,quantity,available_quantity,avg_cost,as_of,source\n600519,100,90,100.5,2024-01-02,manual"}
```

只有在核对 `content_hash` 后显式确认才会建立快照：

```http
POST /api/v2/portfolio/confirm
{"confirmed":true,"preview_id":"...","content_hash":"...","cash":10000,"as_of":"2024-01-02","available_at":"2024-01-02T16:00:00+08:00","source":"manual","csv_content":"..."}
```

该快照只是研究/数量建议输入，不代表真实账户已同步，也不会触发调仓或交易。

## 备份与恢复

`decision.recovery.RuntimeRecovery` 只对显式传入的脱敏临时副本操作。备份前关闭正在写入的任务，至少包含：

- `data/reports/decision.sqlite`；
- `data/snapshots/`；
- `data/user/portfolio_snapshots.json`；
- 当前配置文件和研究账本（如使用）。

备份后必须调用 `verify_backup` 校验每个文件哈希。恢复默认拒绝覆盖已有目标；先恢复到干净目录，再运行启动检查和离线回归。不要在真实用户数据上用删除、覆盖或损坏文件演练。

## 故障排查

1. 报告显示数据不可用：检查 `data/cache/` 的来源、完整交易日和质量字段；不要手工把缓存标成新鲜。
2. 历史报告读回失败：检查 SQLite 报告、manifest、CSV 内容哈希和 schema/version 是否一致；不要重新运行模型补救。
3. 持仓数量建议被拒绝：检查现金、可卖数量、记录时点、证券状态和风险约束；缺字段按未知处理。
4. 任务超时或取消：读取任务终态，旧任务晚到结果不会覆盖新版本；失败记录不得手工改成成功。
5. 启动检查失败：先修复本地 Python/依赖/路径问题，不运行自动安装命令。

## 验收口径

离线夹具通过只证明工程边界和恢复逻辑；不等于真实行情、真实模型、真实 GPU、真实前瞻或正式资格通过。任何资格开放都必须另有冻结协议、成熟样本和主控复核记录。
