# model/ — 核心模型库

Kronos 的模型层遵循**两阶段架构**：**Tokenizer（VQ-VAE 量化器）→ Predictor（自回归 Transformer）**。原始 OHLCV 数据先被量化为离散 token，再由 Transformer 自回归预测未来 token，最后解码回连续价格空间。

---

## 文件清单与职责

### `__init__.py` — 公共导出与工厂函数

| 导出项 | 类型 | 说明 |
|--------|------|------|
| `KronosTokenizer` | class | VQ-VAE tokenizer |
| `Kronos` | class | Transformer predictor |
| `KronosPredictor` | class | 高层推理接口 |
| `get_model_class(name)` | func | 按名称字符串查找模型类 |

- **设计模式**：外观模式（Facade）——将三个核心类统一暴露；简单工厂模式（`get_model_class` 将模型名称映射到类对象）。
- **集成点**：被 `finetune/`、`finetune_csv/`、`examples/`、`tests/` 等外部模块引用，是该目录的唯一公共入口。

---

### `module.py` — 底层神经网络组件

包含所有可复用的子模块，构成 tokenizer 和 predictor 的"积木块"。

#### 量化组件（Quantizers）

| 类 | 职责 | 设计模式 |
|----|------|----------|
| `DifferentiableEntropyFunction` | 自动微分的熵函数（straight-through estimator） | 自定义 autograd Function |
| `codebook_entropy()` | 便捷包装熵计算 | 函数封装 |
| `BinarySphericalQuantizer` | 核心二元球面量化器，实现 [arXiv:2406.07548](https://arxiv.org/pdf/2406.07548.pdf) | 策略模式（支持 soft/hard entropy 策略） |
| `BSQuantizer` | 将 BSQ 的输出拆分为 s1（粗粒度）+ s2（细粒度）两半 | 适配器模式（Adapter） |

**Data Flow**：`z`（连续向量）→ `BinarySphericalQuantizer.quantize()` → 二值化（{-1, +1}）→ `codes_to_indexes()` → 整数索引。`half=True` 时通过 `BSQuantizer` 拆分为 `[s1_indices, s2_indices]`。

#### Transformer 核心组件

| 类 | 职责 | 设计模式 |
|----|------|----------|
| `RMSNorm` | 均方根归一化（比 LayerNorm 更轻量） | 规范化策略 |
| `RotaryPositionalEmbedding` | RoPE 旋转位置编码 | 位置编码策略 |
| `FeedForward` | SwiGLU 前馈网络（`w1`/`w3` 门控 + `w2` 输出） | 标准 FFN |
| `MultiHeadAttentionWithRoPE` | 带 RoPE 的因果自注意力 | 多头注意力 + 位置编码组合 |
| `MultiHeadCrossAttentionWithRoPE` | 带 RoPE 的交叉注意力 | 同自注意力，但 query/key/value 可来自不同源 |
| `TransformerBlock` | Pre-norm Transformer 块：`Self-Attention → Residual → FFN → Residual` | 残差连接 + 预归一化 |
| `HierarchicalEmbedding` | 层级 Token 嵌入：分别嵌入 s1/s2 后拼接融合 | 特征融合策略 |
| `TemporalEmbedding` | 时间特征嵌入（分钟/小时/星期/日/月），支持可学习或固定正弦位置编码 | 策略模式（learnable vs fixed） |
| `DependencyAwareLayer` | 依赖感知层：s2 通过交叉注意力条件化在 s1 嵌入上 | 条件注意力 |
| `DualHead` | 双输出头：`proj_s1` + `proj_s2`，含交叉熵损失计算 | 多头输出 |
| `FixedEmbedding` | 固定正弦-余弦位置编码（不可学习） | 位置编码策略 |

**Data Flow**（Kronos predictor 内部）：

```
s1_ids, s2_ids  →  HierarchicalEmbedding
                                    ↓
                  + TemporalEmbedding(stamp)  →  token_dropout
                                    ↓
                        TransformerBlock × N 层
                                    ↓
                              RMSNorm
                                    ↓
                          DualHead.proj_s1  →  s1_logits
                                    ↓
                  DependencyAwareLayer (交叉注意力 + sibling_embed)
                                    ↓
                          DualHead.cond_forward  →  s2_logits
```

**设计要点**：
- 全部使用 **Pre-norm** 架构（先 norm 再 attention/ffn），训练更稳定。
- 使用 PyTorch 的 `scaled_dot_product_attention`（Flash Attention 后端），支持 causal mask 和 dropout。
- `HierarchicalEmbedding` 支持**复合 token** 输入：可直接传入 `(s1_ids, s2_ids)` 二元组，或传入单个整数张量自动通过位运算拆分。

---

### `kronos.py` — 核心模型实现

#### `KronosTokenizer` — VQ-VAE 量化器

```
Input (B, T, d_in)
    ↓
nn.Linear (embed)
    ↓
TransformerBlock × (n_enc_layers - 1)   [编码器]
    ↓
nn.Linear (quant_embed) → (B, T, codebook_dim)
    ↓
BSQuantizer
    ↓
┌─────────────────────────────────────────────┐
│  s1 路径: quantized[:,:,:s1_bits]            │
│   → post_quant_embed_pre → decoder × L → head│
│   → z_pre (B, T, d_in)                       │
├─────────────────────────────────────────────┤
│  s2 路径: quantized                          │
│   → post_quant_embed → decoder × L → head    │
│   → z (B, T, d_in)                           │
└─────────────────────────────────────────────┘
Output: (z_pre, z), bsq_loss, quantized, z_indices
```

- **设计模式**：编码器-解码器（Encoder-Decoder）+ 残差量化（Residual Quantization）。
- **损失函数**：`commit_loss`（β 加权） + entropy 惩罚（γ0/gamma/zeta 控制）。

#### `Kronos` — Transformer 预测器

```
s1_ids, s2_ids  ←  KronosTokenizer.encode(x) 的输出
    ↓
HierarchicalEmbedding([s1_ids, s2_ids])
    ↓
+ TemporalEmbedding(stamp)  →  token_drop
    ↓
TransformerBlock × n_layers
    ↓
RMSNorm
    ↓
DualHead.proj_s1  →  s1_logits
    ↓
(Teacher forcing: 使用 s1_targets)
(或 Sampling: softmax → multinomial → sample_s1_ids)
    ↓
emb_s1(sample_s1_ids)  →  DependencyAwareLayer(hidden, sibling_embed)
    ↓
DualHead.cond_forward  →  s2_logits
```

- **设计模式**：
  - 推理时使用**双重采样策略**：先采样 s1，再以 sampled s1 为条件采样 s2。
  - 训练时支持 **Teacher Forcing**：直接使用真实 s1 标签作为 s2 的条件输入。
- **层级预测**：s1（粗粒度趋势）→ s2（细粒度细节），降低预测难度。

#### `auto_regressive_inference` — 自回归生成引擎

**Data Flow**：

```
输入 x (B, T, d_in)              ← 已标准化 + clip
    ↓ repeat(1, sample_count, 1, 1)  →  (B×sample_count, T, d_in)
    ↓
tokenizer.encode(x, half=True)    →  [s1_tokens, s2_tokens]
    ↓
滑动窗口缓冲区 (pre_buffer, post_buffer)   ← 长度为 max_context
    ↓
for i in range(pred_len):
    ├─ decode_s1(input_tokens, stamp)   →  s1_logits + context
    ├─ sample_from_logits(s1_logits)    →  sample_pre (next s1 token)
    ├─ decode_s2(context, sample_pre)   →  s2_logits
    ├─ sample_from_logits(s2_logits)    →  sample_post (next s2 token)
    ├─ 写入 generated_pre/post
    └─ 更新滑动缓冲区 (roll or append)
    ↓
拼接 full_pre + full_post → tokenizer.decode(half=True) → (B, T, d_in)
    ↓
reshape (B, sample_count, T, d_in) → mean over sample_count → numpy
```

- **设计模式**：滑动窗口（Sliding Window） + 多候选平均（Multi-sample Averaging）。
- **采样策略**：支持 `temperature`、`top-k`、`top-p`（nucleus）过滤。

#### `KronosPredictor` — 高层推理接口

| 方法 | 职责 |
|------|------|
| `__init__` | 自动检测设备（CUDA/MPS/CPU），将模型移动到目标设备 |
| `generate` | numpy → tensor → `auto_regressive_inference` → 切片返回 `pred_len` 步 |
| `predict` | 单序列完整流程：DataFrame 校验 → 缺失列填充 → NaN 检查 → 时间戳编码 → Z-score 标准化 → clamp → generate → 反标准化 → 返回 DataFrame |
| `predict_batch` | 多序列并行预测：校验一致性 → 堆叠为 batch → generate → 逐序列反标准化 → 返回 DataFrame 列表 |

**Data Flow（predict）**：

```
DataFrame(df)
    ↓ 校验价格列 + 填充 volume/amount
    ↓ 计算时间戳特征 (minute, hour, weekday, day, month)
    ↓
x = df[price_cols + vol_cols].values
    ↓ Z-score: (x - mean) / (std + 1e-5)
    ↓ clip: [-self.clip, self.clip]
    ↓ unsqueeze(0)  →  (1, seq_len, feat)
    ↓
generate() → auto_regressive_inference()
    ↓
preds * (std + 1e-5) + mean    [反标准化]
    ↓
DataFrame(preds, index=y_timestamp)  →  返回
```

---

## 设计模式总结

| 模式 | 位置 | 说明 |
|------|------|------|
| **编码器-解码器（Encoder-Decoder）** | `KronosTokenizer` | Transformer 编码器 → 量化 → Transformer 解码器 |
| **分层预测（Hierarchical Prediction）** | `Kronos.forward` | s1 粗粒度 → s2 细粒度，通过 `DependencyAwareLayer` 条件建模 |
| **残差连接（Residual Connection）** | `TransformerBlock` | Pre-norm 残差结构 |
| **Straight-Through Estimator** | `BinarySphericalQuantizer.quantize` | 前向硬量化、反向直通梯度 |
| **策略模式（Strategy Pattern）** | `BinarySphericalQuantizer` | soft_entropy vs hard_entropy 切换 |
| **策略模式（Strategy Pattern）** | `TemporalEmbedding` | `learn_pe` 控制使用可学习/固定位置编码 |
| **滑动窗口（Sliding Window）** | `auto_regressive_inference` | 固定长度 `max_context` 的循环缓冲区 |
| **多候选平均（Multi-sample Averaging）** | `auto_regressive_inference` | sample_count 次采样后取均值作为最终预测 |
| **适配器模式（Adapter）** | `BSQuantizer` | 包装 `BinarySphericalQuantizer` 并拆分 s1/s2 输出 |
| **外观模式（Facade）** | `__init__.py` | 统一导出三个核心类 |
| **简单工厂（Simple Factory）** | `get_model_class()` | 字符串到模型类的映射查找 |
| **Hub 集成模式** | `KronosTokenizer`, `Kronos` | 继承 `PyTorchModelHubMixin` 以实现 `from_pretrained` 和 `push_to_hub` |

---

## 关键集成点

| 接口 / 契约 | 消费方 | 说明 |
|-------------|--------|------|
| `KronosTokenizer.encode(x, half=True) → [s1, s2]` | `auto_regressive_inference` | 推理时量化输入 |
| `KronosTokenizer.decode([s1, s2], half=True) → tensor` | `auto_regressive_inference` | 推理结束后解码回 OHLCV |
| `Kronos.forward(s1, s2, stamp, ...) → s1_logits, s2_logits` | 训练脚本 (`finetune/`) | 训练 predictor 时的前向传播 |
| `Kronos.decode_s1 / decode_s2` | `auto_regressive_inference` | 分步自回归生成 |
| `KronosPredictor.predict(df, ...) → DataFrame` | `examples/`, `webui/` | 终端用户推理接口 |
| `KronosPredictor.predict_batch(...) → List[DataFrame]` | `finetune_csv/`, `webui/` | 批量推理接口 |
| `BSQuantizer.forward(z, half, collect_metrics)` | `KronosTokenizer.forward` | Tokenizer 训练时量化 + 损失 |

### 外部依赖

- **Hugging Face Hub**（通过 `PyTorchModelHubMixin`）：模型权重托管于 `NeoQuasar/Kronos-*` 仓库。`from_pretrained()` 自动下载。
- **einops**：`rearrange` / `reduce` 用于张量形状变换和归约。
- **pandas / numpy**：`KronosPredictor` 的 DataFrame 输入/输出。

### 注意与约束

1. **模型与 Tokenizer 必须配对**：Kronos-mini 用 2k tokenizer，small/base 用 base tokenizer。不匹配导致静默错误。
2. **输入列必须包含**：`['open', 'high', 'low', 'close']`，`volume`/`amount` 可选。
3. **sys.path hack**：`kronos.py` 顶部有 `sys.path.append("../")`，脚本必须从项目根目录运行。
4. **权重不进 Git**：所有 `.pth`/`.pt`/`.bin` 被 `.gitignore` 排除。必须通过 `from_pretrained` 从 Hugging Face Hub 加载。
