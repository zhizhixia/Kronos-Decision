# AGENTS.md — Kronos-Decision

> 本文件为 Hermes + OpenCode 协作开发规范，每次会话自动加载。

---

## 项目概览

Kronos-Decision 是基于 [Kronos](https://github.com/shiyu-coder/Kronos) (AAAI 2026) 的 A 股交易决策系统增强版。

## 全局开发规则

以下规则适用于本项目的**所有编程任务**。

### §1 模型选择

| 任务类型 | 模型 | 思考强度 |
|---------|------|:--:|
| 编码实现/修复 Bug | `go/glm-5.2` | medium |
| 前端 HTML/CSS/JS | `go/kimi-k2.7-code` | medium |
| 推理规划/架构分析 | `go/deepseek-v4-pro` | high |
| 代码搜索/文件扫描 | `go/deepseek-v4-flash` | low |

- **禁止**对简单机械任务使用 high reasoning

### §2 设计文档

- 设计文档中的每个"已有/现有/原文"断言，写之前用 `grep` 或 `search_files` 确认
- 按风险等级确定审查力度，不搞一刀切：
  - **高风险**（破坏性变更、API 修改）→ 三阶段审查
  - **中风险**（新增文件、UI 修改）→ 两阶段审查
  - **低风险**（文案修正、格式调整）→ 一阶段审查
- 完成后立即执行"关键路径可启动"验证

### §3 代码编写

- 写完文件立刻检查编码为 UTF-8 无 BOM
- **便于维护而非便于编写**：独立模块、结构体返回错误、配置集中管理
- 中文注释/文档/日志/错误信息；英文标识符（变量/函数/类/文件名）
- 函数不超过 60 行；公开 API 必须有类型注解

### §4 Git 操作

- 永远显式 `git add <file>`，不用 `git add .` 或 `-A`
- 提交前 `git diff --stat --cached` 检查只含预期文件
- 发现脏提交立即 `git reset --soft HEAD~1` 重来
- 中文 commit message，格式：`类型: 描述`
- 类型：`feat` / `fix` / `docs` / `style` / `chore` / `test`

### §5 OpenCode 使用

- 首次使用 opencode 编写代码后，后续所有代码修改都用 opencode
- 一次性任务用 `opencode run`，不启动交互式 TUI
- 需要审批的任务加 `--auto` 跳过设计等待
- 每次修改后必须验证：
  - 代码修改 → 运行相关测试
  - 前端修改 → 启动 WebUI 验证渲染
  - 配置修改 → 检查服务可启动
- 子智能体选择：
  - `--agent fixer`（代码修改）
  - `--agent designer`（前端修改）
  - `--agent council`（代码审查）
  - `--agent orchestrator`（规划协调）

### §6 编排与并行

- 14+ 任务计划分批并行 dispatch，不串行到底
- 5-8 分钟能完成的机械任务合并成大任务，减少代理调度开销
- 并行任务间必须无文件冲突；有冲突的显式标注依赖串行

### §7 数据与缓存

- 缓存文件使用 **CSV** 格式，禁止依赖 `pyarrow`/`parquet`
- `data/cache/` 目录必须在 `.gitignore` 中
- 数据获取必须有三级降级：主源 → 备源 → 缓存

### §8 前端开发

- 前端修改后必须启动 WebUI 验证渲染效果
- JavaScript 变量作用域是高频 Bug 来源——跨函数依赖必须**显式传参**
- Canvas 渲染必须在容器可见后进行（先 `classList.add('active')` 再 `render`）
- Windows `.bat` 脚本避免 Unicode 字符和 emoji（中文可用，特殊符号用纯 ASCII）

---

## 目录结构

| 目录 | 用途 |
|------|------|
| `model/` | 核心库：`KronosTokenizer`, `Kronos`, `KronosPredictor` |
| `decision/` | 🆕 决策信号层（analyzer / engine / config / model_manager / errors） |
| `data/` | 🆕 数据管道（fetcher 三级降级 / pool 动态股票池） |
| `webui/` | 🔧 Flask Web 界面（决策报告 / 深色主题 / Canvas K 线） |
| `finetune/` | Qlib 微调流水线（DDP） |
| `finetune_csv/` | CSV 驱动微调流水线 |
| `examples/` | 预测示例脚本 |
| `tests/` | pytest 测试（回归 + 决策系统） |

## 启动

```shell
pip install -r requirements.txt
pip install flask flask-cors plotly akshare baostock pyyaml loguru
python webui/run.py
# http://localhost:7070/report
```

## 代码地图

完整代码地图见根目录 `codemap.md` 及各子目录 `codemap.md`。
