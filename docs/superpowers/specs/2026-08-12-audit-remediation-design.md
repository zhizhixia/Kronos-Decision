# Kronos-Decision 审查缺陷修复设计

## 目标

在不引入新依赖、不连接券商、不运行全量沪深300评估的前提下，修复已复现的安全、可信决策、采样审计、模拟订单和 Windows 运行缺陷。修复后仍不得把自动测试或最小真实运行表述为投资有效性证明。

## 行为边界

- Web 服务默认只监听 `127.0.0.1`，关闭 Flask 调试器，不返回跨域许可；带非本机 `Origin` 的写请求直接拒绝。
- 旧数据文件页只允许读取项目 `data/` 目录中的 CSV；拒绝目录穿越、任意绝对路径和 Feather/Parquet。
- 在线 CPU 推理使用 `prediction.fallback_sample_count`；`prediction.timeout_seconds` 作为自回归推理的单调时钟截止时间。超时同步中断并返回 `PREDICTION_TIMEOUT`，不创建无法停止的后台推理线程。
- v1 的 BUY/HOLD/SELL 只能由正式五态动作映射。缺少正式评估、版本不匹配、过期证据、陈旧数据、`ERROR_*` 数据旗标、未经真实日历验证的完整交易日或路径硬错误时一律返回 HOLD/degraded。
- 普通数据 warning 保留展示但不自动视为硬错误；`ERROR_*`、`CALENDAR_QLIB_OUTDATED`、`CALENDAR_WEEKDAY_FALLBACK`、陈旧数据和 `PATH_REPAIR_EXCESSIVE` 视为硬失败。
- v2 从指定本地组合读取是否持仓；相同弱信号对持仓股为 REDUCE，对未持仓股为 AVOID。
- 采样身份绑定 `pred_len/sample_count/sample_batch_size/T/top_p/top_k`。评估预测键和断点续跑兼容检查必须包含该身份，批大小变化不得复用旧预测。
- 买入数量只能是 100 股整数倍；卖出数量只能是 100 股整数倍，或一次性包含当前持仓的完整零股余额。
- 模型通过租约管理。活跃租约大于零时不得释放；最后一个租约归还后才启动空闲计时。
- Windows 启动脚本在括号块内使用运行时错误码，正确处理含空格 Python 路径，并由根依赖文件覆盖其检查项。
- 修复直接运行的前瞻 CLI 和根目录微调文档入口；删除临时 `sitecustomize.py`，清除示例 BOM。

## 验证路径

1. 定向单元测试：安全请求、路径限制、CPU 配置、deadline、硬/软质量旗标、持仓五态、采样哈希/缓存键、整手与零股、模型租约。
2. 离线回归：默认排除 network/model/GPU 标记的完整 pytest。
3. 静态与入口检查：CLI `--help`、BAT 语法契约、Node 内联脚本语法、UTF-8 无 BOM、`git diff --check`。
4. 轻量运行检查：回环地址启动后访问页面/API，确认恶意 Origin 无法写入；不触发全量模型或历史沪深300推理。

## 明确未验证

- 全量沪深300正式评估及 H00300 全收益基准门禁。
- 不同 GPU/驱动之间逐元素一致性。
- 未来至少一个完整 20 交易日周期的投资表现。
