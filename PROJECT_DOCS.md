# Kronos 项目完整技术文档

> **Kronos** — 全球首个开源金融 K 线（OHLCV）时序基础模型  
> 论文：已被 AAAI 2026 接收 | arXiv: [2508.02739](https://arxiv.org/abs/2508.02739)  
> GitHub: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)  
> HuggingFace: [NeoQuasar](https://huggingface.co/NeoQuasar)

---

## 一、项目概述

### 1.1 项目定位

Kronos 是一个基于 Transformer 的**Decoder-only 金融时序基础模型**（Foundation Model）。不同于通用的时序预测模型，Kronos 专门针对金融市场数据的高噪声、非平稳特性设计，采用创新的**两阶段架构**：

```
Tokenizer（量化） → Predictor（自回归 Transformer）
```

1. **Tokenizer 阶段**：将连续的、多维的 K 线数据（OHLCV）量化为**层级离散 Token**
2. **Predictor 阶段**：在离散 Token 上进行自回归预训练，实现统一的多任务量化预测

### 1.2 核心创新点

- **层级离散 Token**：采用 s1（粗粒度）+ s2（细粒度）双层 Token 结构
- **Binary Spherical Quantization (BSQ)**：基于球形量化的离散化方法，有效压缩连续金融数据
- **Dependency-Aware Layer**：预测 s2 时显式条件化于 s1，保证层级之间的一致性
- **大规模预训练**：在超过 45 个全球交易所的数据上进行预训练
- **决策信号层**：基于预测结果的多路径统计分析，通过趋势/风险/反转矩阵生成 BUY/HOLD/SELL 交易信号

### 1.3 模型规格

| 模型 | Tokenizer | 上下文长度 | 参数量 | 开源 |
|------|-----------|-----------|--------|------|
| Kronos-mini | Kronos-Tokenizer-**2k** | 2048 | 4.1M | ✅ |
| Kronos-small | Kronos-Tokenizer-**base** | 512 | 24.7M | ✅ |
| Kronos-base | Kronos-Tokenizer-**base** | 512 | 102.3M | ✅ |
| Kronos-large | Kronos-Tokenizer-base | 512 | 499.2M | ❌ |

---

## 二、技术架构

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────┐
│                 Kronos 两阶段架构                    │
├──────────────────────────────────────────────────┤
│                                                    │
│  原始 OHLCV 数据                                   │
│       │                                            │
│       ▼                                            │
│  ┌─────────────────────┐                          │
│  │   KronosTokenizer   │  阶段一：量化              │
│  │                     │                          │
│  │  Encoder → BSQ →    │  输出：s1_indices        │
│  │  Decoder (双路)      │       s2_indices        │
│  └────────┬────────────┘                          │
│           │                                        │
│           ▼                                        │
│  ┌─────────────────────┐                          │
│  │      Kronos         │  阶段二：预测              │
│  │                     │                          │
│  │  HierarchicalEmb    │                          │
│  │  + TemporalEmb      │                          │
│  │  + N×Transformer    │  自回归生成 s1, s2       │
│  │  + DualHead         │                          │
│  └────────┬────────────┘                          │
│           │                                        │
│           ▼                                        │
│  ┌─────────────────────┐                          │
│  │  KronosPredictor    │  高层推理封装              │
│  │  • 标准化/逆标准化    │                          │
│  │  • 批量推理          │                          │
│  │  • 采样策略          │                          │
│  └─────────────────────┘                          │
│                                                    │
└──────────────────────────────────────────────────┘
```

### 2.2 数据流详解

```
输入: DataFrame [open, high, low, close, volume, amount]
  │
  ├─ 时间特征提取: minute, hour, weekday, day, month
  ├─ Z-score 标准化: (x - mean) / (std + 1e-5)
  ├─ 裁剪: np.clip(x, -5, 5)
  │
  ▼
KronosTokenizer.encode(x) → [s1_indices, s2_indices]
  │
  ▼
Kronos.forward(s1_ids, s2_ids, stamp)
  │  ├─ HierarchicalEmbedding: s1_emb + s2_emb → fused
  │  ├─ + TemporalEmbedding(stamp)
  │  ├─ N×TransformerBlock(RMSNorm → RoPE-Attention → FFN)
  │  ├─ DualHead.forward(x) → s1_logits
  │  └─ DependencyAwareLayer(x, s1_embed) → DualHead.cond_forward → s2_logits
  │
  ▼
自回归推理: 逐步采样 s1 → 条件采样 s2 → decode → OHLCV
  │
  ▼
逆标准化: pred * (std + 1e-5) + mean
  │
  ▼
输出: DataFrame [open, high, low, close, volume, amount]
```

---

## 三、目录结构

```
kronos/
├── AGENTS.md                    # AI Agent 开发指南（架构约定、陷阱）
├── PROJECT_DOCS.md              # 项目完整技术文档（本文件）
├── README.md                    # 项目说明（英文）
├── requirements.txt             # 核心依赖
├── LICENSE                      # MIT 许可证
├── figures/                     # 图片资源（logo, 示例图）
│
├── model/                       # 🔴 核心模型库
│   ├── __init__.py              # 导出 KronosTokenizer, Kronos, KronosPredictor
│   ├── kronos.py                # Tokenizer + Kronos + Predictor + 推理函数 (~662行)
│   └── module.py                # 底层模块：BSQ, Attention, Embedding等 (~570行)
│
├── decision/                    # 🆕 决策信号层
│   ├── __init__.py              # 包声明
│   ├── errors.py                # 层次化异常体系 (KronosError 基类 + 5 子类)
│   ├── config.yaml              # YAML 配置文件（模型/预测/信号/数据/Web UI/日志）
│   ├── config.py                # 线程安全单例配置管理 + 热更新
│   ├── model_manager.py         # 模型生命周期管理（懒加载 + 超时释放）
│   ├── analyzer.py              # 信号分析器（趋势/风险/反转 → 决策矩阵）
│   └── engine.py                # 决策引擎编排（端到端 → DecisionReport）
│
├── data/                        # 🆕 数据管道（重构为正式模块）
│   ├── __init__.py              # 包声明
│   ├── fetcher.py               # A股日K数据获取（akshare + baostock 三级降级）
│   ├── pool.py                  # 沪深300动态股票池（周级刷新 + 缓存降级）
│   └── cache/                   # Parquet 缓存目录（行情数据 + 股票池）
│
├── webui/                       # 🌐 Flask Web 可视化界面
│   ├── run.py                   # 启动脚本（端口 7070）
│   ├── app.py                   # Flask 应用（API路由 + 预测逻辑 + 决策报告）
│   ├── requirements.txt         # Web UI 额外依赖（Flask, plotly）
│   ├── static/
│   │   └── style.css            # 🆕 深色交易终端主题 (1348行)
│   └── templates/
│       ├── index.html           # 原有预测页面
│       └── report.html          # 🆕 决策报告页面（全中文 UI, 972行）
│
├── finetune/                    # 🔧 基于 Qlib 的微调流水线（多GPU DDP）
│   ├── config.py                # 配置类（数据路径、超参、模型路径）
│   ├── dataset.py               # QlibDataset（滑动窗口采样）
│   ├── qlib_data_preprocess.py  # Qlib 数据预处理
│   ├── train_tokenizer.py       # Tokenizer 微调训练（DDP）
│   ├── train_predictor.py       # Predictor 微调训练（DDP）
│   ├── qlib_test.py             # 回测与推理脚本
│   └── utils/
│       ├── __init__.py
│       └── training_utils.py    # DDP 工具函数
│
├── finetune_csv/                # 🔧 基于 CSV 的简化微调流水线
│   ├── train_sequential.py      # 顺序训练编排器
│   ├── finetune_tokenizer.py    # Tokenizer 微调
│   ├── finetune_base_model.py   # Predictor 微调 + CustomKlineDataset
│   ├── config_loader.py         # YAML 配置加载器
│   └── configs/                 # YAML 配置文件目录
│
├── examples/                    # 📊 示例脚本
│   ├── prediction_example.py    # 基础预测示例
│   ├── prediction_wo_vol_example.py  # 无成交量示例
│   ├── prediction_batch_example.py   # 批量预测示例
│   ├── predict_with_confidence.py    # 置信区间预测（分布式推理）
│   ├── predict_my_stocks.py          # A股预测（腾讯API数据源）
│   ├── fetch_tencent.py              # 腾讯API数据获取测试
│   ├── prediction_new.py / prediction_new_GUI.py / etc.
│   ├── yuce/                    # 回测相关
│   └── predictions/             # 预测结果输出
│
├── tests/                       # 🧪 测试体系 (10个文件, 28个用例)
│   ├── test_kronos_regression.py      # 确定性输出回归测试
│   ├── test_errors.py                 # 🆕 异常体系测试 (3用例)
│   ├── test_config.py                 # 🆕 配置管理测试 (3用例)
│   ├── test_analyzer.py               # 🆕 信号分析器测试 (4用例)
│   ├── test_model_manager.py          # 🆕 模型管理器测试 (3用例)
│   ├── test_engine.py                 # 🆕 决策引擎测试 (3用例)
│   ├── test_fetcher.py                # 🆕 数据获取器测试 (3用例)
│   ├── test_pool.py                   # 🆕 股票池测试 (2用例)
│   ├── test_integration.py            # 🆕 集成测试 (3用例)
│   └── data/
│       ├── regression_input.csv       # 测试输入数据
│       ├── regression_output_256.csv  # 预期输出（context=256）
│       ├── regression_output_512.csv  # 预期输出（context=512）
│       └── generate_regression_output.py  # 生成预期输出的脚本
│
└── .gitignore                   # Git 忽略规则
```

---

## 四、核心模块详解

### 4.1 `model/module.py` — 底层模块

这是 Kronos 的基础组件库，包含所有底层的神经网络模块。

#### 4.1.1 BinarySphericalQuantizer (BSQ)

```python
class BinarySphericalQuantizer(nn.Module):
    """
    论文: https://arxiv.org/pdf/2406.07548.pdf
    将连续向量量化为二值球面码本 {-1, +1}^D
    """
```

**核心机制：**
- **量化方式**：将 L2 归一化后的向量按每个维度二值化为 -1 或 +1
- **直通估计器 (STE)**：`z + (zhat - z).detach()` 实现前向二值化、反向直通梯度
- **熵正则化**：
  - `gamma0 * persample_entropy`（惩罚每个样本的熵，促进多样性）
  - `-gamma * cb_entropy`（奖励码本熵，防止码本坍缩）
- **分组熵计算**：将 codebook_dim 分为 group_size 大小的组，分别计算熵后再汇总
- **损失函数**：`commit_loss + zeta * entropy_penalty`

#### 4.1.2 BSQuantizer（分层量化封装）

```python
class BSQuantizer(nn.Module):
    def __init__(self, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):
        self.codebook_dim = s1_bits + s2_bits  # 总码本维度
        self.bsq = BinarySphericalQuantizer(self.codebook_dim, ...)
```

- **half=True** 模式：将量化后的向量切分为 `s1_bits` 和 `s2_bits` 两部分，分别转为索引
- **half=False** 模式：整段转为单个索引

#### 4.1.3 RMSNorm

```python
class RMSNorm(nn.Module):
    # Root Mean Square Layer Normalization
    # output = x * rsqrt(mean(x^2) + eps) * weight
```

#### 4.1.4 RotaryPositionalEmbedding (RoPE)

旋转位置编码，使用复数旋转变换实现位置信息的注入。

#### 4.1.5 MultiHeadAttentionWithRoPE

- 标准多头自注意力 + RoPE
- 支持 `key_padding_mask` 用于处理不等长序列
- 使用 `F.scaled_dot_product_attention`（PyTorch 2.0+ 的高效实现）
- `is_causal=True` 实现因果掩码

#### 4.1.6 MultiHeadCrossAttentionWithRoPE

交叉注意力，用于 `DependencyAwareLayer`，query 来自 s1 嵌入，key/value 来自 Transformer 上下文。

#### 4.1.7 HierarchicalEmbedding（层级嵌入）

```python
class HierarchicalEmbedding(nn.Module):
    """
    s1_ids → Embedding(s1_vocab) → s1_emb
    s2_ids → Embedding(s2_vocab) → s2_emb
    → Concat → Linear(d_model*2, d_model) → fused_emb
    """
```

支持两种输入模式：
- **元组模式**：`([s1_ids, s2_ids])` — 直接传入分离的 token
- **复合模式**：`(composite_ids)` — 通过位运算自动分离

#### 4.1.8 DependencyAwareLayer

```python
class DependencyAwareLayer(nn.Module):
    """
    s2 的预测显式依赖 s1：
    CrossAttention(query=sibling_embed(s1), key=context, value=context)
    → residual + norm
    """
```

#### 4.1.9 TransformerBlock

标准 Pre-LN Transformer 块：
```
x → RMSNorm → SelfAttention(RoPE) → residual
  → RMSNorm → FeedForward(SwiGLU) → residual
```

#### 4.1.10 FeedForward (SwiGLU)

使用 SwiGLU 激活函数：`SiLU(w1(x)) * w3(x)` → `w2`。

#### 4.1.11 DualHead

双头预测层：
- `forward(x)` → s1_logits
- `cond_forward(x2)` → s2_logits
- `compute_loss()` → CE(s1) + CE(s2)

#### 4.1.12 TemporalEmbedding

将时间特征（分钟、小时、星期、日、月）映射为嵌入向量。支持两种模式：
- `learn_te=True`：可学习的 Embedding
- `learn_te=False`：固定的正弦位置编码（FixedEmbedding）

---

### 4.2 `model/kronos.py` — 核心模型与推理

#### 4.2.1 KronosTokenizer（编解码器）

```python
class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
```

**架构组成：**
- `embed`: Linear(d_in → d_model)
- `encoder`: N×TransformerBlock
- `quant_embed`: Linear(d_model → codebook_dim)
- `tokenizer`: BSQuantizer (s1_bits + s2_bits)
- `post_quant_embed_pre`: Linear(s1_bits → d_model) — 仅用 s1 部分重建
- `post_quant_embed`: Linear(codebook_dim → d_model) — 完整码本重建
- `decoder`: N×TransformerBlock（编解码器共享）
- `head`: Linear(d_model → d_in)

**前向传播流程：**
```
x → embed → encoder → quant_embed → BSQ
                                      ├─→ quantized[s1] → post_quant_embed_pre → decoder → head → z_pre
                                      └─→ quantized[full] → post_quant_embed → decoder → head → z
返回: ((z_pre, z), bsq_loss, quantized, z_indices)
```

**关键方法：**
- `encode(x, half=True)` → `[s1_indices, s2_indices]`（仅编码，用于推理）
- `decode(indices, half=True)` → 重建的 OHLCV
- `indices_to_bits(x, half=False)` → 将整数索引转为 {-1, +1} 二值向量

#### 4.2.2 Kronos（自回归预测器）

```python
class Kronos(nn.Module, PyTorchModelHubMixin):
```

**架构组成：**
- `embedding`: HierarchicalEmbedding(s1_bits, s2_bits, d_model)
- `time_emb`: TemporalEmbedding
- `token_drop`: Dropout（Token级别）
- `transformer`: N×TransformerBlock
- `norm`: RMSNorm
- `dep_layer`: DependencyAwareLayer
- `head`: DualHead

**核心前向传播：**
```python
def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None,
            use_teacher_forcing=False, s1_targets=None):
    x = embedding([s1_ids, s2_ids])
    x += time_emb(stamp)  # 注入时间特征
    x = token_drop(x)
    for layer in transformer:
        x = layer(x, key_padding_mask=padding_mask)
    x = norm(x)
    s1_logits = head(x)  # 预测 s1
    
    # 条件 s2 预测
    if use_teacher_forcing:
        sibling_embed = embedding.emb_s1(s1_targets)
    else:
        # 从 s1_logits 采样
        s1_probs = softmax(s1_logits)
        sample_s1_ids = multinomial(s1_probs)
        sibling_embed = embedding.emb_s1(sample_s1_ids)
    
    x2 = dep_layer(x, sibling_embed)
    s2_logits = head.cond_forward(x2)
    return s1_logits, s2_logits
```

**自回归推理的分离模式：**
- `decode_s1(s1_ids, s2_ids, stamp)` → 先预测 s1，返回 s1_logits + context
- `decode_s2(context, s1_ids)` → 基于 context 和 s1 采样结果预测 s2

这种分离设计允许在 `auto_regressive_inference` 中实现高效的逐步采样。

#### 4.2.3 auto_regressive_inference（核心推理循环）

```python
def auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp,
                               max_context, pred_len, clip=5, T=1.0,
                               top_k=0, top_p=0.99, sample_count=5,
                               verbose=False):
```

**推理流程：**
1. **数据预处理**：clip(x, -clip, clip)
2. **Tokenize**：`tokenizer.encode(x, half=True)` → s1/s2 索引
3. **多路径展开**：将输入按 `sample_count` 复制，并行生成多条路径
4. **滑动窗口自回归**（针对每条路径）：
   - 使用循环缓冲区维护 `max_context` 长度的历史 Token
   - 每步：`decode_s1` → 采样 s1 → `decode_s2` → 采样 s2
   - 将新生成的 Token 追加到缓冲区
5. **解码**：用 Tokenizer 将完整 token 序列解码回 OHLCV
6. **多路径平均**：`np.mean(preds, axis=1)` 对多条路径求平均
7. **返回**：`(batch, pred_len, 6)` 的 numpy 数组

#### 4.2.4 采样策略

```python
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, ...):
    # top-k: 保留概率最高的 k 个，其余设为 -inf
    # top-p (nucleus): 保留累积概率达到 p 的最小集合

def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k, top_p)
    probs = softmax(logits)
    return multinomial(probs)  # 或 argmax（当 sample_logits=False）
```

#### 4.2.5 KronosPredictor（高层推理接口）

```python
class KronosPredictor:
    def __init__(self, model, tokenizer, device=None, max_context=512, clip=5):
        # 自动检测设备：CUDA > MPS > CPU
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_col = 'volume'
        self.amt_vol = 'amount'
```

**核心方法：**

- **`predict(df, x_timestamp, y_timestamp, pred_len, T, top_k, top_p, sample_count, verbose)`**
  1. 验证必需列（OHLC）
  2. 自动填充缺失的 volume/amount 列
  3. 时间特征提取（calc_time_stamps）
  4. Z-score 标准化 + clip
  5. 调用 `auto_regressive_inference`
  6. 逆标准化
  7. 返回 DataFrame

- **`predict_batch(df_list, ...)`**
  批量预测多个时间序列，要求所有序列具有相同的历史长度和预测长度。

- **`generate(x, x_stamp, y_stamp, pred_len, T, ...)`**
  低级推理接口，直接处理 numpy 数组。

#### 4.2.6 calc_time_stamps

```python
def calc_time_stamps(x_timestamp):
    # 提取: minute, hour, weekday, day, month
```

时间特征维度为 5，分别对应分钟（0-59）、小时（0-23）、星期（0-6）、日（1-31）、月（1-12）。

---

### 4.3 `model/__init__.py` — 公共 API

```python
from .kronos import KronosTokenizer, Kronos, KronosPredictor

model_dict = {
    'kronos_tokenizer': KronosTokenizer,
    'kronos': Kronos,
    'kronos_predictor': KronosPredictor
}
```

只有这三个类对外暴露，其余模块为内部实现。

---

## 五、决策系统模块（decision/）🆕

### 5.1 架构概述

决策系统是 Kronos 的上层应用模块，提供端到端的股票预测与交易信号生成。架构分为三个层次：

```
DataFetcher（数据获取）→ ModelManager（模型管理）→ SignalAnalyzer（信号分析）
         ↑ 三级降级                      ↑ 懒加载+超时释放           ↑ 六规则决策矩阵
```

由 `DecisionEngine` 统一编排，外部只需调用一个方法即可获得完整 `DecisionReport`。

### 5.2 `decision/errors.py` — 层次化异常体系（25 行）

```python
KronosError(Exception)          # 基类，所有项目异常的根
├── ConfigError                 # 配置相关异常（YAML 解析失败、参数非法）
├── DataSourceError             # 数据源异常（akshare/baostock 连接失败）
├── DataValidationError         # 数据校验异常（缺少列、数据量不足）
├── ModelNotReadyError          # 模型未就绪异常（下载未完成、加载失败）
└── PredictionTimeoutError      # 预测超时异常（推理时间过长）
```

**设计原则**：所有异常均可被 `DecisionEngine` 统一捕获，转为 `DecisionReport(status='error')` 而不向上抛出，确保外部调用者无需处理异常。

### 5.3 `decision/config.py` + `config.yaml` — 配置管理（139 + 39 行）

**Config Dataclass 结构：**
```python
@dataclass
class Config:
    model: ModelConfig            # tokenizer/predictor 路径、device
    prediction: PredictionConfig  # pred_len, T, top_p, sample_count, context
    signal: SignalConfig          # buy_threshold, sell_threshold, confidence_min
    data: DataConfig              # cache_ttl, max_retries, timeout
    webui: WebUIConfig            # host, port, debug
    logging: LoggingConfig        # level, file, max_size
```

**线程安全单例模式：**
- `get_config()` — 双重检查锁（`threading.Lock`），首次调用从 YAML 加载
- `reload_config()` — 热重载，无需重启进程
- `save_config()` — 修改配置后写回 YAML 持久化

### 5.4 `decision/model_manager.py` — 模型生命周期管理（122 行）

```python
class ModelManager:
    def get_predictor() -> KronosPredictor  # 懒加载，首次调用从 HF Hub 下载
    def is_ready() -> bool                   # 模型是否已加载到 GPU
    def is_loading() -> bool                 # 是否正在下载中
    def _cleanup()                           # 空闲超时后释放 GPU 显存
    def _release_gpu()                       # 将模型移至 CPU 释放显存
```

**关键机制：**
- **懒加载**：`get_predictor()` 首次调用时才从 HuggingFace Hub 下载并实例化，避免启动时长时间阻塞
- **超时释放**：后台线程每 60 秒检查一次，若超过 `idle_timeout_minutes`（默认 10 分钟）无访问，自动调用 `_release_gpu()` 将模型移至 CPU 释放 GPU 显存
- **全局单例**：通过工厂函数 `get_model_manager()` 获取唯一实例

### 5.5 `decision/analyzer.py` — 信号分析器（224 行）

多路径预测结果 → 交易信号的决策引擎。

**输入**：`(sample_count, pred_len, 5)` 形状的预测路径数组（OHLCV）+ 当前价格

**分析流程：**

```
预测路径 (50×120×5)
  │
  ├─ 趋势分析 → TrendResult
  │    ├─ direction: 'up' | 'down' | 'sideways'
  │    ├─ strength: 0~1（预测期末价格变化率的均值/标准差）
  │    └─ consistency: 0~1（样本路径中同向比例）
  │
  ├─ 风险评估 → RiskResult
  │    ├─ volatility: 路径间标准差
  │    ├─ var_95: 95% 置信度 VaR（最大亏损）
  │    └─ reversal_risk: 0~1（短期反转概率）
  │
  └─ 决策矩阵 (6 规则) → 最终信号
       ├─ 强上升 + 高一致性 + 低风险 → BUY
       ├─ 强下降 + 高一致性 → SELL
       ├─ 弱趋势 + 低一致性 → HOLD
       ├─ 高波动 + 方向不明 → HOLD
       ├─ 极端反转风险 → HOLD（避免追涨杀跌）
       └─ 默认 → HOLD
```

**输出：**
```python
@dataclass
class SignalResult:
    signal: str               # 'BUY' | 'HOLD' | 'SELL'
    reason: str               # 中文判断理由
    trend_direction: str      # 趋势方向
    trend_strength: float     # 趋势强度 (0~1)
    trend_consistency: float  # 一致性 (0~1)
    risk_volatility: float    # 波动率
    risk_reversal: float      # 反转风险 (0~1)
    confidence: float         # 综合置信度 (0~1)
```

### 5.6 `decision/engine.py` — 决策引擎（165 行）

端到端编排器，连接数据、模型、分析三大组件。

```python
class DecisionEngine:
    def predict_and_analyze(
        stock_code: str,
        pred_len: int = 120,
        temperature: float = 1.0,
        top_p: float = 0.99,
        sample_count: int = 50
    ) -> DecisionReport
```

**编排流程：**
1. `DataFetcher.fetch_daily(stock_code)` — 获取历史 K 线
2. `ModelManager.get_predictor()` — 获取/加载模型
3. `KronosPredictor.predict()` — 自回归推理，生成多路径预测
4. `SignalAnalyzer.analyze()` — 分析预测路径，生成交易信号
5. 返回 `DecisionReport` 结构体

**异常安全设计（兜底策略）：**
- 所有异常在 Engine 层被捕获
- 异常时返回 `DecisionReport(status='error', error=错误描述)` 而非抛出
- 部分降级：数据异常返回 `status='degraded'`，模型异常返回 `status='error'`
- 外部调用者**始终**收到一个 `DecisionReport`，无需 try-except

```python
@dataclass
class DecisionReport:
    status: str              # 'ok' | 'error' | 'degraded'
    stock_code: str          # 股票代码
    stock_name: str          # 股票名称
    signal: str              # 'BUY' | 'HOLD' | 'SELL'
    reason: str              # 信号理由
    current_price: float     # 当前收盘价
    predictions: ndarray     # (pred_len, 6) 预测 OHLCV 均值
    confidence: float        # 综合置信度
    trend: TrendResult       # 趋势分析详情
    risk: RiskResult         # 风险分析详情
    elapsed_seconds: float   # 总耗时
    error: Optional[str]     # 错误信息（仅异常时）
```

---

## 六、数据管道模块（data/）🆕

### 6.1 架构概述

数据管道负责 A 股日 K 线数据的获取、缓存和校验，采用**三级降级策略**保证高可用。

### 6.2 `data/fetcher.py` — 数据获取器（207 行）

```python
class DataFetcher:
    def fetch_daily(stock_code: str) -> pd.DataFrame
```

**三级降级获取策略：**

```
请求 stock_code
  │
  ├─ 第一级：Parquet 本地缓存
  │    命中 + 未过期 → 直接返回（最快）
  │    命中 + 已过期 → 尝试远程更新
  │
  ├─ 第二级：akshare（主数据源）
  │    成功 → 标准化 → 写入缓存 → 返回
  │    失败 → 指数退避重试（2s / 4s / 8s，最多 3 次）
  │
  ├─ 第三级：baostock（备用数据源）
  │    成功 → 标准化 → 写入缓存 → 返回
  │    失败 → 尝试降级
  │
  └─ 降级：过期缓存（兜底）
       有缓存 → 使用过期数据 + 标记 degraded
       无缓存 → 抛出 DataSourceError
```

**数据校验规则：**
- 必需列：`open`, `high`, `low`, `close`
- `high >= low`（每行检查）
- 行数 ≥ 50（保证足够历史上下文）
- 不满足任一条件 → 抛出 `DataValidationError`

**标准化输出：**
- 列顺序统一：`date`, `open`, `high`, `low`, `close`, `volume`, `amount`
- 日期格式：`YYYY-MM-DD`
- 缺失 `volume`/`amount` 自动填 0

**缓存策略：**
- 缓存格式：Parquet（`data/cache/{stock_code}.parquet`）
- TTL：交易时段内 5 分钟，收盘后（16:00 后）立即过期
- 强制刷新：传入 `force_refresh=True` 跳过缓存

### 6.3 `data/pool.py` — 沪深 300 股票池（84 行）

```python
def get_hs300_pool() -> dict[str, str]  # {code: name}，如 {"600519": "贵州茅台"}
def refresh_pool() -> dict[str, str]     # 强制刷新，无视缓存
```

**刷新策略：**
- 通过 akshare 动态获取最新沪深 300 成分股
- 本地 parquet 缓存（`data/cache/pool_hs300.parquet`）
- 7 天自动刷新（避免频繁网络请求）
- Web 设置页可手动触发强制刷新
- akshare 不可用时降级使用过期缓存

---

## 七、Web UI 模块

### 7.1 架构概述

基于 Flask + Plotly + Chart.js 的可视化界面，端口 7070。提供两大功能页面：
- **预测页面**（`/`）：原有数据文件加载 + 模型预测 + K 线图表
- **决策报告页面**（`/report`）🆕：沪深 300 实时预测 + BUY/HOLD/SELL 交易信号

### 7.2 启动方式

```shell
cd webui && python run.py    # → http://localhost:7070
```

### 7.3 `webui/run.py` — 启动脚本

- 检查依赖安装状态（flask, flask_cors, pandas, numpy, plotly）
- 自动检测 Kronos 模型库可用性
- 自动安装缺失依赖
- 启动后自动打开浏览器

### 7.4 `webui/app.py` — Flask 应用

**API 路由：**

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 主页（渲染 HTML 模板） |
| `/report` | GET | 🆕 决策报告页面（渲染 `report.html`） |
| `/api/data-files` | GET | 扫描 data/ 目录，返回可用数据文件列表 |
| `/api/load-data` | POST | 加载指定数据文件，返回数据统计信息 |
| `/api/predict` | POST | 执行预测，返回 Plotly JSON 图表 |
| `/api/load-model` | POST | 从 HuggingFace 加载指定模型 |
| `/api/available-models` | GET | 返回可用模型列表及状态 |
| `/api/model-status` | GET | 返回当前模型加载状态 |
| `/api/decision-report` | POST | 🆕 生成决策报告（接收 `{stock_code, pred_len, ...}` → `DecisionEngine.predict_and_analyze()`） |
| `/api/stock-list` | GET | 🆕 获取沪深 300 股票池列表（`[{code, name}]`） |
| `/api/config` | GET / PUT | 🆕 配置查询与更新（GET 返回全量配置，PUT 逐字段更新并持久化） |

**支持的模型配置：**
```python
AVAILABLE_MODELS = {
    'kronos-mini':  {'model_id': 'NeoQuasar/Kronos-mini',
                     'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
                     'context_length': 2048, 'params': '4.1M'},
    'kronos-small': {'model_id': 'NeoQuasar/Kronos-small',
                     'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
                     'context_length': 512, 'params': '24.7M'},
    'kronos-base':  {'model_id': 'NeoQuasar/Kronos-base',
                     'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
                     'context_length': 512, 'params': '102.3M'},
}
```

**图表生成：**
- 使用 Plotly Candlestick 绘制 K 线图
- 历史数据（蓝色）、预测数据（绿色）、实际数据（橙色）三线叠加
- 自动检测时间频率并保证 x 轴连续性

**预测结果保存：**
- 自动保存到 `webui/prediction_results/` 目录
- JSON 格式，包含输入摘要、预测结果、实际对比

### 7.5 `webui/static/style.css` — 深色交易终端主题🆕（1348 行）

**设计令牌（CSS 变量体系）：**
```css
:root {
    --bg-base: #0a0e17;        /* 基座背景 */
    --bg-panel: #131820;       /* 面板背景 */
    --bg-inset: #0d121b;       /* 内嵌区域 */
    --accent: #00d4aa;         /* 青绿强调色（三档透明变体） */
    --signal-buy: #00c853;     /* 买入绿 */
    --signal-sell: #ff1744;    /* 卖出红 */
    --signal-hold: #ffab00;    /* 持有橙 */
}
```

- 字体：`Space Grotesk`（标题）+ `JetBrains Mono`（等宽数字）
- 关键组件：`.card`（卡片面板）、`.signal-badge`（信号标签 BUY/HOLD/SELL 配色）、`.metric`（指标数字）、`.kline-chart`（K 线图容器）
- 动画：`@keyframes fadeUp`（入场动画）、`reportReveal`（报告渐现）

### 7.6 `webui/templates/report.html` — 决策报告页面🆕（972 行）

**页面布局：**
1. **顶部标题栏**：logo + 标题 + 实时状态指示器
2. **控制栏**：股票代码下拉框（沪深 300 动态加载）、预测参数输入（pred_len / T / top_p）
3. **信号区域**：BUY / HOLD / SELL 大标签 + 中文理由 + 实时时钟
4. **分析面板**（四列 CSS Grid）：趋势分析、风险评估、技术指标、一致性评分
5. **K 线走势图**：Canvas + Chart.js 渲染，历史数据 + 预测叠加

**内联 JavaScript（约 500 行，IIFE 模式）：**
- `init()` → `generateReport(stockCode, params)` → 调用 `/api/decision-report`
- `renderSignal()`：渲染信号大标签 + 理由
- `renderKlineChart()`：Chart.js 渲染历史 + 预测 K 线
- `renderMetrics()`：四列指标网格
- `showAlert()` / `clearAlert()`：错误/警告提示条

---

## 八、微调模块

### 6.1 `finetune/` — 基于 Qlib 的微调流水线

#### 6.1.1 配置文件：`config.py`

```python
class Config:
    # 数据参数
    qlib_data_path = "~/.qlib/qlib_data/cn_data"
    instrument = 'csi300'  # 沪深300
    lookback_window = 90   # 输入序列长度
    predict_window = 10    # 预测长度
    max_context = 512
    feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']

    # 训练参数
    epochs = 30
    batch_size = 50
    tokenizer_learning_rate = 2e-4
    predictor_learning_rate = 4e-5

    # 回测参数
    backtest_n_symbol_hold = 50   # 持仓数量
    inference_T = 0.6
    inference_top_p = 0.9
    inference_sample_count = 5
```

#### 6.1.2 训练流程

**阶段 1：Tokenizer 微调**

```
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
```

- 损失函数：`(recon_loss_pre + recon_loss_all + bsq_loss) / 2`
- 其中 `recon_loss_pre` 使用 s1 位重建，`recon_loss_all` 使用完整码本重建
- 梯度累积支持模拟更大 batch size
- OneCycleLR 学习率调度

**阶段 2：Predictor 微调**

```
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
```

- 必须先完成 Tokenizer 微调（predictor 依赖微调后的 tokenizer 做量化）
- 损失函数：`(CE_s1 + CE_s2) / 2`（交叉熵）
- Tokenizer 冻结（eval 模式），仅更新 Predictor 参数

#### 6.1.3 数据处理

**`qlib_data_preprocess.py`：**
- 从 Qlib 加载沪深300成分股 OHLCV 数据
- 计算 `amount = avg_price * volume`
- 按时间范围切分 train/val/test
- 保存为 pickle 文件

**`dataset.py` — QlibDataset：**
- 预计算所有有效的 (symbol, start_index) 对
- 随机采样滑动窗口
- 仅用 lookback 窗口计算统计量进行归一化（防止数据泄露）

#### 6.1.4 回测

**`qlib_test.py`：**
- 加载微调后模型进行推理
- 生成多种信号：`last`, `mean`, `max`, `min`（预测收盘价相对于最后一天收盘价的变化）
- 使用 Qlib 的 `TopkDropoutStrategy` 进行回测
- 输出累计收益曲线并保存图表

### 6.2 `finetune_csv/` — 基于 CSV 的简化微调流水线

#### 6.2.1 设计理念

相比 `finetune/`，此流水线更简洁：
- 直接从 CSV 文件加载数据（无需 Qlib 依赖）
- YAML 配置文件管理所有超参数
- 顺序训练（先 Tokenizer、再 Predictor）

#### 6.2.2 启动方式

```shell
python finetune_csv/train_sequential.py --config finetune_csv/configs/your_config.yaml
```

支持命令行参数：`--skip-tokenizer`, `--skip-basemodel`, `--skip-existing`

#### 6.2.3 config_loader.py

**ConfigLoader**：YAML 配置加载器，支持：
- 动态路径解析（`{exp_name}` 占位符替换）
- 分层配置：data、training、model_paths、experiment、device、distributed

**CustomFinetuneConfig**：将 YAML 配置映射为 Python 属性：
```python
class CustomFinetuneConfig:
    # 数据: data_path, lookback_window, predict_window, train_ratio, val_ratio
    # 训练: tokenizer_epochs, basemodel_epochs, batch_size, lr, etc.
    # 模型: pretrained_tokenizer_path, pretrained_predictor_path
    # 实验: train_tokenizer, train_basemodel, skip_existing
    # 设备: use_cuda, device_id
```

#### 6.2.4 CustomKlineDataset

相比 QlibDataset 的改进：
- 直接读取 CSV（`pd.read_csv`）
- 按时间比例划分 train/val/test（非按日期）
- 训练时使用确定性哈希采样（`(idx * 9973 + epoch * 104729) % max_start`）保证可复现

#### 6.2.5 训练编排：SequentialTrainer

```python
class SequentialTrainer:
    def run_training(self):
        if config.train_tokenizer:
            self.train_tokenizer_phase()
        if config.train_basemodel:
            self.train_basemodel_phase()
```

支持：
- 从预训练权重加载或随机初始化（`pre_trained_tokenizer`, `pre_trained_predictor`）
- 自动检测已存在模型并跳过（`skip_existing`）
- 日志系统（RotatingFileHandler）

---

## 九、示例脚本

### 7.1 prediction_example.py — 基础预测

典型使用模式：
```python
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

pred_df = predictor.predict(
    df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
    pred_len=120, T=1.0, top_p=0.9, sample_count=1
)
```

### 7.2 prediction_wo_vol_example.py — 无成交量预测

展示仅使用 OHLC（无 volume/amount）的情况，模型自动用 0 填充。

### 7.3 prediction_batch_example.py — 批量预测

```python
pred_df_list = predictor.predict_batch(
    df_list=[df1, df2, ...],
    x_timestamp_list=[xt1, xt2, ...],
    y_timestamp_list=[yt1, yt2, ...],
    pred_len=120
)
```

所有序列必须具有**相同的历史长度和预测长度**。

### 7.4 predict_with_confidence.py — 置信区间预测

最复杂的示例（576 行），功能包括：
- 通过腾讯财经 API 拉取 A 股日线数据（前复权）
- 自动判断上交所/深交所（`6/9` 开头 → sh，`0/2/3` 开头 → sz）
- **分布式推理**：生成 50 条样本路径（每批 10 条），不做平均
- **统计分析**：
  - 期末价格分布（均值、中位数、标准差、min/max）
  - 涨跌概率（P(price > current)）
  - VaR（5% 和 1% 风险价值）
  - 最大回撤（平均和最差）
  - 80% 点向置信带
- **四因子综合评分**：
  - 方向因子（35%）：涨的概率
  - 收益因子（25%）：期望收益
  - 稳定性因子（20%）：波动率
  - 尾部风险因子（20%）：VaR
- **交易信号**：BUY / HOLD / SELL / WAIT
- **可视化**：双层图（价格路径+置信带 + 期末分布直方图）

### 7.5 predict_my_stocks.py — A 股预测

对农业银行（601288）、中国平安（601318）、声讯科技（003004）进行预测，每只股票单独生成图表。

### 7.6 fetch_tencent.py — 腾讯 API 测试

独立的腾讯财经数据获取测试脚本。

---

## 十、测试体系

### 8.1 测试架构

基于 pytest，使用固定的 HuggingFace 模型版本保证确定性输出。

### 8.2 回归测试

**`test_kronos_predictor_regression`：**
- 加载固定 revision 的模型
- 使用固定 seed 和采样参数（T=1.0, top_k=1, top_p=1.0）确保确定性
- 参数化：context_len = [512, 256]
- 对比预测输出与预期 CSV 文件的相对误差（容差 1e-5）

**`test_kronos_predictor_mse`：**
- 随机采样 4 个时间窗口
- 计算每个窗口的预测 MSE
- 验证平均 MSE 与预期值的差异不超过 1e-6

### 8.3 固定版本

```python
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
SEED = 123
DEVICE = "cpu"  # 测试固定在 CPU 上运行
```

---

## 十一、依赖与环境

### 9.1 核心依赖（requirements.txt）

```
numpy, pandas, torch>=2.0.0
einops==0.8.1       # 张量操作
huggingface_hub==0.33.1  # 模型下载
matplotlib==3.9.3    # 可视化
tqdm==4.67.1        # 进度条
safetensors==0.6.2  # 模型序列化
```

### 9.2 Web UI 额外依赖

```
Flask, flask-cors, plotly
```

### 9.3 微调额外依赖

```
pyqlib  (Qlib 微调流水线)
comet_ml  (可选，实验追踪)
pyyaml  (CSV 流水线配置)
```

---

## 十二、开发规范与注意事项

### 10.1 关键陷阱

> ⚠️ 来自 AGENTS.md 的重要注意事项

1. **sys.path hack**：`model/kronos.py` 使用 `sys.path.append("../")`，必须从项目根目录运行脚本
2. **模型与 Tokenizer 配对**：Kronos-mini 使用专用 Tokenizer-2k，不匹配会导致静默错误
3. **微调顺序**：必须 Tokenizer 先、Predictor 后，Predictor 依赖 Tokenizer 的量化
4. **无 CI**：所有验证需本地运行 `pytest tests/`
5. **finetune/ 中的 AI 生成注释**：由 Gemini 2.5 Pro 生成，以代码逻辑为准
6. **模型权重不进 Git**：`.pth/.pt/.ckpt/.bin` 均被排除，必须从 HuggingFace Hub 下载

### 10.2 DataFrame 约定

- 必需列：`['open', 'high', 'low', 'close']`
- 可选列：`volume`, `amount`（缺失时自动填 0）
- 时间戳列：`timestamps` / `timestamp` / `date`

### 10.3 模型保存与加载

所有模型继承 `PyTorchModelHubMixin`，支持：
```python
model.save_pretrained("./path")
model = Kronos.from_pretrained("./path")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
```

### 10.4 无 lint/formatter

项目没有 `.flake8`、`ruff.toml` 或 `pyproject.toml`，修改代码时不要引入格式化工具配置。

---

## 十三、安装与运行指南

### 11.1 环境准备

```shell
# Python 3.10+
pip install -r requirements.txt
```

### 11.2 快速预测

```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
predictor = KronosPredictor(model, tokenizer, max_context=512)

pred_df = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=120)
```

### 11.3 微调（Qlib 流水线）

```shell
# 1. 修改 finetune/config.py 中的路径
# 2. 数据预处理
python finetune/qlib_data_preprocess.py
# 3. 微调 tokenizer
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
# 4. 微调 predictor
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
# 5. 回测
python finetune/qlib_test.py --device cuda:0
```

### 11.4 微调（CSV 流水线）

```shell
python finetune_csv/train_sequential.py --config finetune_csv/configs/your_config.yaml
```

### 11.5 Web UI

```shell
cd webui && python run.py    # → http://localhost:7070
```

### 11.6 运行测试

```shell
pytest tests/
```

---

## 十四、API 参考

### KronosTokenizer

| 方法 | 输入 | 输出 |
|------|------|------|
| `from_pretrained(path)` | 模型路径/HF ID | KronosTokenizer 实例 |
| `encode(x, half=False)` | (B, T, d_in) Tensor | 量化索引（half=True 时返回 [s1_idx, s2_idx]） |
| `decode(indices, half=False)` | 量化索引 | 重建的 (B, T, d_in) Tensor |
| `forward(x)` | (B, T, d_in) Tensor | ((z_pre, z), bsq_loss, quantized, indices) |

### Kronos

| 方法 | 输入 | 输出 |
|------|------|------|
| `from_pretrained(path)` | 模型路径/HF ID | Kronos 实例 |
| `forward(s1_ids, s2_ids, stamp, ...)` | Token + 时间戳 | (s1_logits, s2_logits) |
| `decode_s1(s1_ids, s2_ids, stamp)` | Token + 时间戳 | (s1_logits, context) |
| `decode_s2(context, s1_ids)` | 上下文 + s1 采样 | s2_logits |

### KronosPredictor

| 方法 | 输入 | 输出 |
|------|------|------|
| `predict(df, x_ts, y_ts, pred_len, **sampling)` | DataFrame + 时间戳 | 预测 DataFrame |
| `predict_batch(df_list, ...)` | 多个 DataFrame | 多个预测 DataFrame |

---

## 十五、引用

```bibtex
@misc{shi2025kronos,
    title={Kronos: A Foundation Model for the Language of Financial Markets},
    author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu
            and Changshui Zhang and Jian Li},
    year={2025},
    eprint={2508.02739},
    archivePrefix={arXiv},
    primaryClass={q-fin.ST},
    url={https://arxiv.org/abs/2508.02739},
}
```

---

> 📝 文档生成时间：2026-07-09  
> 📝 最后更新：2026-07-11（新增 decision/、data/ 模块，Web UI 增强，测试扩展）  
> 📝 基于项目源码深度分析，覆盖 model/、decision/、data/、webui/、finetune/、finetune_csv/、examples/、tests/ 全部关键文件
