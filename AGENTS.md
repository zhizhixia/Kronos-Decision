# AGENTS.md — Kronos

## 项目概览

Kronos 是一个基于 Transformer 的金融 K 线（OHLCV）时序基础模型。采用两阶段架构：**Tokenizer（量化）→ 预测器（自回归 Transformer）**。所有模型权重从 Hugging Face Hub 下载，不进 Git。

## 目录结构

| 目录 | 用途 |
|------|------|
| `model/` | 核心库：`KronosTokenizer`, `Kronos`, `KronosPredictor` |
| `finetune/` | 基于 Qlib 的 A 股微调流水线（DistributedDataParallel） |
| `finetune_csv/` | 基于原始 CSV 的微调流水线（更简单，YAML 配置） |
| `webui/` | Flask Web 可视化界面（端口 7070） |
| `examples/` | 预测脚本示例 |
| `tests/` | 回归测试（pytest） |

## 关键架构约定

### Token 层级结构
Kronos 使用**层级离散 token**：s1（粗粒度） + s2（细粒度）。
- `KronosTokenizer.encode(x, half=True)` 返回 `[s1_indices, s2_indices]`
- `KronosTokenizer.decode(indices, half=True)` 从双索引重建 OHLCV
- `Kronos.forward()` 先预测 s1 logits，再基于 s1 条件预测 s2 logits（`DependencyAwareLayer`）

### 模型与 Tokenizer 配对
| 模型 | Tokenizer | 最大上下文 |
|------|-----------|-----------|
| Kronos-mini | Kronos-Tokenizer-**2k** | 2048 |
| Kronos-small | Kronos-Tokenizer-**base** | 512 |
| Kronos-base | Kronos-Tokenizer-**base** | 512 |

**不匹配会导致静默错误。** Kronos-mini 有自己专用的 2k tokenizer。

### 必需的 DataFrame 列
`['open', 'high', 'low', 'close']` —— 必须存在。`volume` 和 `amount` 可选（缺失时自动填 0）。

## 命令参考

### 安装
```shell
pip install -r requirements.txt
```

### 运行测试
```shell
pytest tests/
```
测试固定了 Hugging Face 模型版本（`MODEL_REVISION`、`TOKENIZER_REVISION`），不要随意更新。

### 预测（推理）
```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

pred_df = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=120)
```

### 微调 — Qlib 流水线（`finetune/`）
**顺序：tokenizer 先，predictor 后。** predictor 依赖 tokenizer 对数据进行量化。

```shell
# 1. 修改 finetune/config.py 中的路径
# 2. 数据预处理
python finetune/qlib_data_preprocess.py
# 3. 微调 tokenizer（多 GPU）
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
# 4. 微调 predictor（多 GPU）
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
# 5. 回测
python finetune/qlib_test.py --device cuda:0
```

### 微调 — CSV 流水线（`finetune_csv/`）
```shell
python finetune_csv/train_sequential.py --config finetune_csv/configs/your_config.yaml
```

### WebUI
```shell
cd webui && python run.py    # → http://localhost:7070
```
WebUI 有独立的 `webui/requirements.txt`（包含 Flask、plotly）。

## 陷阱与注意事项

- **sys.path hack**：`model/kronos.py` 使用 `sys.path.append("../")`。务必从项目根目录运行脚本，否则 `from model.module import *` 会失败。
- **无 lint/formatter 配置**：没有 .flake8、ruff.toml 或 pyproject.toml。修改代码时不要引入格式化工具配置，除非另行要求。
- **模型权重不进 Git**：`.pth`、`.pt`、`.ckpt`、`.bin` 均被 `.gitignore` 排除。必须从 Hugging Face Hub 下载。
- **微调时的 token order**：tokenizer 必须先于 predictor 训练。predictor 接收 tokenizer 量化后的 token 作为输入。
- **finetune/config.py 中的 TODO**：微调前必须在 `finetune/config.py` 中更新 `qlib_data_path`、`pretrained_tokenizer_path`、`pretrained_predictor_path`。
- **`finetune/` 目录中的 AI 生成注释**（README 注明）：`finetune/` 下许多注释由 Gemini 2.5 Pro 生成，可能存在不准确之处。以代码逻辑为准。
- **回归测试固定版本**：`test_kronos_regression.py` 固定了 HF 模型 revision 以确保确定性输出。升级模型依赖时需同时更新预期输出 CSV 文件。
- **没有 CI**：本项目无 GitHub Actions 或 CI 配置。所有验证需本地运行。

## 公共 API

只有这三个类从 `model/` 导出：
- `KronosTokenizer`：将 OHLCV 数据量化为离散 token
- `Kronos`：自回归 Transformer 预测器
- `KronosPredictor`：高层推理接口（处理标准化、预测、逆标准化）

## Repository Map

完整的代码地图见根目录 `codemap.md`。

在开始任何任务之前，阅读 `codemap.md` 以了解：
- 项目架构和系统入口点
- 各目录职责和设计模式
- 数据流和模块间集成点

对特定目录进行深度工作时，也请阅读该目录下的 `codemap.md`。
