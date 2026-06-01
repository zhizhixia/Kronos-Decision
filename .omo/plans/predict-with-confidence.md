# Kronos 分布预测 + 交易信号脚本

## TL;DR

> **Quick Summary**: 新建 `examples/predict_with_confidence.py`，用 50 条采样路径替代单点预测，输出概率分布 + 交易信号评分，每只股票一张双面板图表。
>
> **Deliverables**:
> - `examples/predict_with_confidence.py`（唯一新文件）
> - 每只股票生成 `{code}_{name}_confidence.csv` + `{code}_{name}_confidence_chart.png`
>
> **Estimated Effort**: Short（1 文件，~350 行）
> **Parallel Execution**: NO（单文件顺序实现）
> **Critical Path**: Task 1 → Task 2 → Task 3

---

## Context

### Original Request
用户要求新建一个脚本，增加 Kronos 股票 K 线预测的真实交易指导意义——从单点预测改为分布预测，输出交易信号。

### Interview Summary
**Key Discussions**:
- 当前 `predict_my_stocks.py` 只输出单点预测，`sample_count=1` 导致每次结果不同
- `auto_regressive_inference` 在 `model/kronos.py:467` 用 `np.mean` 压平了所有采样路径
- 需要保留 50 条独立路径，计算概率分布和交易信号
- 不修改 `model/` 下的库代码，在新脚本中复制推理函数
- 数据源复用腾讯 API，交易所前缀自动判断（sh/sz）
- GPU: RTX 5060 Ti (8GB)，Kronos-base (102M)

**Research Findings**:
- `auto_regressive_inference`: L389-469，核心推理函数，需要复制并移除 mean
- `sample_from_logits`: L373-386，采样函数，可直接 import
- `calc_time_stamps`: L472-479，时间特征提取，可直接 import
- `KronosPredictor.__init__` 不调用 `.eval()`，需显式设置
- 去标准化必须在每条路径上独立进行

### Metis 审查发现
- **GPU OOM 风险**: 50 路径并行推理可能超出 8GB 显存，需分块（10 路 × 5 批）
- **random seed**: 需设置 `torch.manual_seed` 确保可复现
- **去标准化顺序**: 必须逐路径去标准化后再算统计量，不能先算统计再反标准化
- **OHLCV 约束**: 单条路径可能违反 high≥max(open,close)，统计量用 close 列
- **VaR 定义**: 对收盘价期末回报计算，不涉及全 6 维

---

## Work Objectives

### Core Objective
新建一个脚本，从 Kronos 模型生成 50 条预测路径，提取概率分布信息并输出可操作的交易信号。

### Concrete Deliverables
- `examples/predict_with_confidence.py`（唯一新文件）

### Definition of Done
- [x] 脚本运行 3 只股票无报错
- [x] 每只股票输出 CSV + 双面板 PNG 图表
- [x] 控制台输出包含 Composite Score + 交易信号 (BUY/HOLD/WAIT/SELL)
- [x] 不修改 `model/` 下任何文件
- [x] `pytest tests/ -x` 依然通过

### Must Have
- 50 条采样路径去标准化后独立保留
- 分布统计：均值、中位数、标准差、80%CI、涨跌概率、VaR 95%/99%
- 四因子综合评分：方向(35%) + 收益(25%) + 稳定性(20%) + 尾部风险(20%)
- 每只股票单独图表：上层价格+置信带，下层分布直方图
- GPU 分块推理防止 OOM

### Must NOT Have
- 不修改 `model/` 目录下任何文件
- 不添加新依赖（只用 requirements.txt 已有的库）
- 不添加 CLI 参数解析（使用文件顶部常量配置）
- 不实现中文节假日处理
- 不添加回测或历史验证模式
- 不添加 Web 服务器或 API

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES（pytest）
- **Automated tests**: None（脚本验证，非单元测试）
- **Agent-Executed QA**: YES（所有验证通过运行命令完成）

### QA Policy
- Script exit code 0
- 输出文件存在且非空
- 现有测试不破坏

---

## TODOs

---

- [x] 1. 复制推理函数并移除路径平均

  **What to do**:
  - 在 `predict_with_confidence.py` 中创建 `generate_distribution()` 函数
  - 复制 `auto_regressive_inference` (model/kronos.py L389-469) 的全部逻辑
  - 删除 L467 `preds = np.mean(preds, axis=1)`，改为返回原始 `(1, sample_count, pred_len, 6)` 形状
  - 在返回前用 `preds[0, :, -pred_len:, :]` 只取预测窗口
  - 添加 `TORCH_PATHS_PER_BATCH = 10` 分块常量，外层循环分 5 批处理 50 路径
  - 显式调用 `model.eval()` 和 `tokenizer.eval()` 在推理前

  **Must NOT do**:
  - 不修改 `model/kronos.py`
  - 不复制 `sample_from_logits` 和 `calc_time_stamps`（直接 import）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 纯函数复制+微调，不涉及新架构设计
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - 无

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 2, Task 3
  - **Blocked By**: None

  **References**:
  - `model/kronos.py:389-469` — `auto_regressive_inference` 完整函数，复制模板
  - `model/kronos.py:373-386` — `sample_from_logits`，直接 import
  - `model/kronos.py:472-479` — `calc_time_stamps`，直接 import
  - `tests/test_kronos_regression.py:64-65` — `.eval()` 调用模式
  - `model/kronos.py:544-556` — 标准化/去标准化逻辑

  **Acceptance Criteria**:
  - [ ] `generate_distribution()` 返回 `(50, 60, 6)` 形状的 numpy array
  - [ ] 调用前 `model.eval()` 和 `tokenizer.eval()` 已执行
  - [ ] 分块处理：10 路径 × 5 批，不 OOM

  **QA Scenarios**:
  ```
  Scenario: 推理函数返回正确形状
    Tool: Bash (Python)
    Preconditions: 加载 Kronos-base 模型 + tokenizer
    Steps:
      1. 准备随机输入 (400, 6) OHLCV 数据
      2. 调用 generate_distribution(tokenizer, model, x, ...)
      3. assert paths.shape == (50, 60, 6)
    Expected Result: shape 正确，无 CUDA OOM
    Evidence: .omo/evidence/task-1-shape.txt

  Scenario: repeat 验证结果稳定
    Tool: Bash (Python)
    Preconditions: torch.manual_seed(42)
    Steps:
      1. 运行推理两次
      2. 比较两次的 close 路径是否完全相同
    Expected Result: paths_close[0] 两次完全一致（seed 确定性）
    Evidence: .omo/evidence/task-1-seed.txt
  ```

  **Commit**: YES
  - Message: `feat(examples): add generate_distribution with per-path return`
  - Files: `examples/predict_with_confidence.py`

---

- [ ] 2. 实现分布分析与交易信号评分

  **What to do**:
  - 实现 `analyze_distribution(paths, current_price)` 函数
  - 逐路径去标准化：`paths * (x_std + 1e-5) + x_mean`
  - 从 close 列（index 3）提取 50 条收盘价路径
  - 计算期末价格统计：均值、中位数、标准差、min、max
  - 计算涨跌概率：`sum(final > current) / 50 * 100`
  - 计算 VaR 95%/99%（期末收盘价回报的百分位数）
  - 计算每条路径的最大回撤，取均值和最差值
  - 计算 80% 置信带：每个时间步的 10th/90th 百分位
  - 计算四因子综合评分，映射到 BUY(>0.4) / HOLD(>0.15) / WAIT(≥-0.2) / SELL(<-0.2)
  - 返回结构化 dict

  **Must NOT do**:
  - 不对全 6 维 OHLCV 算 VaR（只用 close）
  - 不对单条路径做 OHLCV 约束修正（文档注明 limitation）
  - 不使用 scipy/statsmodels（只用 numpy）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 纯 numpy 统计计算，无外部依赖
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - 无

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References**:
  - 无外部参考，统计公式为标准实现

  **Acceptance Criteria**:
  - [ ] `analyze_distribution` 返回的 dict 包含所有必需字段
  - [ ] 涨跌概率在 0-100 范围内
  - [ ] VaR 95% < VaR 99% < mean
  - [ ] 综合评分在 -1 到 1 范围内

  **QA Scenarios**:
  ```
  Scenario: 全涨路径 → up_prob = 100%
    Tool: Bash (Python)
    Steps:
      1. 构造 50 条全涨路径（all final > current）
      2. 调用 analyze_distribution
      3. assert stats['up_prob'] == 100 and 'BUY' in stats['signal']
    Expected Result: BUY 信号，up_prob=100%
    Evidence: .omo/evidence/task-2-up100.txt

  Scenario: 全跌路径 → up_prob = 0%
    Tool: Bash (Python)
    Steps:
      1. 构造 50 条全跌路径（all final < current）
      2. 调用 analyze_distribution
      3. assert stats['up_prob'] == 0 and 'SELL' in stats['signal']
    Expected Result: SELL 信号，up_prob=0%
    Evidence: .omo/evidence/task-2-down100.txt
  ```

  **Commit**: YES
  - Message: `feat(examples): add distribution analysis and trading signal scoring`
  - Files: `examples/predict_with_confidence.py`

---

- [ ] 3. 实现可视化与 CSV 输出

  **What to do**:
  - 实现 `plot_distribution(res, future_dates, name, code)` 函数
  - 上层子图：历史收盘价 + 均值预测线 + 80% 置信带填充
  - 下层子图：期末价格分布直方图，标注当前价、均值、VaR 95%
  - matplotlib "Agg" 后端，不弹出窗口
  - 保存为 `{code}_{name}_confidence_chart.png`
  - 保存 CSV：date, pred_close_mean, pred_close_median, pred_close_std, pred_close_p10, pred_close_p90, up_probability
  - 保存原始 50 条路径 CSV：`{code}_{name}_paths.csv`
  - 主流程：for 循环处理每只股票，先数据获取 → 推理 → 分析 → 绘图

  **Must NOT do**:
  - 不生成多于一个 chart 文件/股票
  - 不添加交互模式
  - 不使用 seaborn（只用 matplotlib）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: matplotlib 标准图表，无复杂可视化
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: 无 Web 前端需求

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: None (final task)
  - **Blocked By**: Task 2

  **References**:
  - `examples/predict_my_stocks.py:174-212` — 现有图表代码模式
  - `examples/predict_my_stocks.py:42-82` — 数据获取函数模式
  - `examples/predict_my_stocks.py:27-37` — 配置常量模式

  **Acceptance Criteria**:
  - [ ] 脚本运行 3 只股票无报错，exit code 0
  - [ ] `pytest tests/ -x` 依然通过
  - [ ] 每只股票生成 CSV + PNG 各一个
  - [ ] PNG 文件大小 > 10KB（非空图）
  - [ ] CSV 包含所有指定列
  - [ ] 控制台输出含 Composite Score + Signal

  **QA Scenarios**:
  ```
  Scenario: 完整运行 3 只股票
    Tool: Bash
    Preconditions: PYTHONPATH=C:\Users\xzh\Desktop\kronos, kronos conda env
    Steps:
      1. cd C:\Users\xzh\Desktop\kronos
      2. conda run -n kronos python examples/predict_with_confidence.py
      3. 检查 exit code = 0
      4. 检查输出文件存在且非空
    Expected Result: exit 0，6 个输出文件（3 CSV + 3 PNG）
    Evidence: .omo/evidence/task-3-run.txt

  Scenario: 现有测试不破坏
    Tool: Bash
    Steps:
      1. cd C:\Users\xzh\Desktop\kronos
      2. conda run -n kronos pytest tests/ -x
    Expected Result: 所有已有测试通过
    Evidence: .omo/evidence/task-3-tests.txt
  ```

  **Commit**: YES
  - Message: `feat(examples): add visualization and main pipeline for predict_with_confidence`
  - Files: `examples/predict_with_confidence.py`

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  验证：新文件只有 `examples/predict_with_confidence.py`，不修改 `model/`，不引入新依赖。
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  检查：无 `as any` 等效模式、无空 catch、无 console.log、无 hardcoded 路径。
  Output: `Build [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  运行脚本验证完整流程：exit 0 + 文件生成 + 测试不破坏。
  Output: `Execution [PASS/FAIL] | Tests [PASS/FAIL] | Files [N generated] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  逐条检查 Must Have/Must NOT Have 合规性，diff 确认未修改 model/。
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

- **Task 1**: `feat(examples): add generate_distribution with per-path return` — `examples/predict_with_confidence.py`
- **Task 2**: `feat(examples): add distribution analysis and trading signal scoring` — `examples/predict_with_confidence.py`
- **Task 3**: `feat(examples): add visualization and main pipeline` — `examples/predict_with_confidence.py`

---

## Success Criteria

### Verification Commands
```bash
# 运行新脚本
cd C:\Users\xzh\Desktop\kronos && conda run -n kronos python examples/predict_with_confidence.py
# 预期: exit 0，输出 6 个文件 + 控制台交易信号

# 已有测试不被破坏
cd C:\Users\xzh\Desktop\kronos && conda run -n kronos pytest tests/ -x
# 预期: 全部通过
```

### Final Checklist
- [ ] 新文件只有 `examples/predict_with_confidence.py`
- [ ] `model/` 下零修改
- [ ] GPU OOM 已通过分块处理
- [ ] `model.eval()` 显式调用
- [ ] 去标准化逐路径执行
- [ ] 已有测试全部通过
