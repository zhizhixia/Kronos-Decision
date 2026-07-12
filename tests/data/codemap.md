# tests/data/

Kronos 回归测试静态数据目录。

---

## Responsibility（职责）

存放模型回归测试所需的黄金数据（Golden Data）：

| 文件 | 职责 |
|------|------|
| `regression_input.csv` | 输入特征数据，包含 `timestamps` + `open/high/low/close/volume/amount` 列 |
| `regression_output_256.csv` | context\_len=256 时的预期预测输出 |
| `regression_output_512.csv` | context\_len=512 时的预期预测输出 |
| `generate_regression_output.py` | fixture 重新生成脚本。在已知良好的模型/分词器 commit 下运行，产出上述两个 CSV。不应在每次 CI 中自动运行，仅在模型版本升级后手动执行 |

---

## Design Patterns（设计模式）

### Golden Data — 可复现回归基线
- 输出 CSV 在模型某次被认可的 commit 上生成（`MODEL_REVISION=901c26c...`, `TOKENIZER_REVISION=0e01173...`）
- `test_kronos_regression.py` 以 `rtol=1e-5` 的相对容差对比当前产出与基线
- fixture 生成脚本与测试共享常量（`SEED=123`, `PRED_LEN=8`, `FEATURE_NAMES`），保证一致性

### 脚本而非自动 fixture —— 显式信任
`generate_regression_output.py` 设计为手动工具而非自动 fixture，确保基线的更新需要显式 review 而不是随代码改动隐式刷新，避免回归测试失去意义。

---

## Flow

```
generate_regression_output.py (手动运行，种子 123)
         │
         ├── 读取 regression_input.csv
         │
         ├── 加载 Kronos-small @ 901c26c + Kronos-Tokenizer-base @ 0e01173
         │
         ├── context=256: predict(前 256 行) → regression_output_256.csv
         │
         └── context=512: predict(前 512 行) → regression_output_512.csv

test_kronos_regression.py (CI 自动运行)
         │
         ├── 读取 regression_input.csv 做输入
         │
         ├── 读取 regression_output_{ctx}.csv 做预期
         │
         └── assert_allclose(obtained, expected, rtol=1e-5)
```
