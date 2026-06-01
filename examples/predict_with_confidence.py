"""
Kronos-base 分布式预测脚本 -- 返回完整样本路径而非平均值
用于计算置信区间和概率分布
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings("ignore")

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn.functional as F

from model import Kronos, KronosTokenizer
from model.kronos import calc_time_stamps, sample_from_logits

# ============ 配置 ============
STOCKS = [
    {"code": "601288", "name": "农业银行"},
    {"code": "601318", "name": "中国平安"},
    {"code": "003004", "name": "声讯科技"},
]

LOOKBACK = 400
PRED_LEN = 60
MAX_CONTEXT = 512
DEVICE = "cuda:0"
SAMPLE_COUNT = 50
PATHS_PER_BATCH = 10
FETCH_DAYS = 800
CLIP = 5
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "predictions")


# ============ 数据获取 (腾讯 API) ============
def fetch_stock_data(code: str) -> pd.DataFrame:
    """通过腾讯财经 API 获取日线复权数据"""
    import requests
    print(f"  Fetching {code} daily data from Tencent...")

    s = requests.Session()
    s.trust_env = False

    # 自动判断交易所：6/9开头=上交所(sh)，0/2/3开头=深交所(sz)
    prefix = "sh" if code.startswith(("6", "9")) else "sz"

    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"{prefix}{code},day,,,{FETCH_DAYS},qfq", "_var": "kline_day"}
    r = s.get(url, params=params, timeout=15)
    text = r.text

    start = text.find("{")
    end = text.rfind("}") + 1
    data = json.loads(text[start:end])

    stock_key = f"{prefix}{code}"
    klines = data["data"][stock_key].get("qfqday", data["data"][stock_key].get("day", []))

    rows = []
    for k in klines:
        rows.append({
            "timestamps": k[0],
            "open": float(k[1]),
            "close": float(k[2]),
            "high": float(k[3]),
            "low": float(k[4]),
            "volume": float(k[5]),
            "amount": 0.0,  # 腾讯 API 无成交额，填 0
        })

    df = pd.DataFrame(rows)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    print(f"    Got {len(df)} rows, {df['timestamps'].iloc[0].date()} ~ {df['timestamps'].iloc[-1].date()}")
    print(f"    Latest close: {df['close'].iloc[-1]:.2f}")
    return df


def generate_future_dates(last_date, n):
    """生成未来 N 个交易日"""
    dates = []
    d = last_date + timedelta(days=1)
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


# ============ 分布式预测核心函数 ============
def generate_distribution(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    max_context,
    pred_len,
    clip=5,
    T=1.0,
    top_k=0,
    top_p=0.99,
    sample_count=50,
    paths_per_batch=10,
    verbose=False,
):
    """
    执行分块 GPU 推理，返回所有样本路径（不做平均）。
    
    Args:
        tokenizer: KronosTokenizer 实例
        model: Kronos 模型实例
        x: 输入张量 (seq_len, 6)
        x_stamp: 输入时间戳 (seq_len, 4)
        y_stamp: 预测时间戳 (pred_len, 4)
        max_context: 最大上下文长度
        pred_len: 预测长度
        clip: 裁剪范围
        T: 温度参数
        top_k: top-k 采样
        top_p: nucleus 采样概率
        sample_count: 总样本数
        paths_per_batch: 每批处理的路径数
        verbose: 是否显示进度条
    
    Returns:
        numpy.ndarray: 形状 (sample_count, pred_len, 6) 的所有样本路径
    """
    tokenizer.eval()
    model.eval()
    torch.manual_seed(42)
    
    all_paths = []
    remaining = sample_count
    
    with torch.no_grad():
        # 确保 3D 输入: (1, seq_len, feat)
        x = x.unsqueeze(0)
        x_stamp = x_stamp.unsqueeze(0)
        y_stamp = y_stamp.unsqueeze(0)
        
        x = torch.clip(x, -clip, clip)
        device = x.device
        
        # 分块处理
        while remaining > 0:
            chunk_size = min(paths_per_batch, remaining)
            
            # 为当前 chunk 复制输入
            x_chunk = x.unsqueeze(1).repeat(1, chunk_size, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
            x_stamp_chunk = x_stamp.unsqueeze(1).repeat(1, chunk_size, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
            y_stamp_chunk = y_stamp.unsqueeze(1).repeat(1, chunk_size, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)
            
            x_token = tokenizer.encode(x_chunk, half=True)
            
            initial_seq_len = x_chunk.size(1)
            batch_size = x_token[0].size(0)
            total_seq_len = initial_seq_len + pred_len
            full_stamp = torch.cat([x_stamp_chunk, y_stamp_chunk], dim=1)
            
            generated_pre = x_token[0].new_empty(batch_size, pred_len)
            generated_post = x_token[1].new_empty(batch_size, pred_len)
            
            pre_buffer = x_token[0].new_zeros(batch_size, max_context)
            post_buffer = x_token[1].new_zeros(batch_size, max_context)
            buffer_len = min(initial_seq_len, max_context)
            if buffer_len > 0:
                start_idx = max(0, initial_seq_len - max_context)
                pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
                post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]
            
            if verbose:
                ran = trange
            else:
                ran = range
            
            for i in ran(pred_len):
                current_seq_len = initial_seq_len + i
                window_len = min(current_seq_len, max_context)
                
                if current_seq_len <= max_context:
                    input_tokens = [
                        pre_buffer[:, :window_len],
                        post_buffer[:, :window_len]
                    ]
                else:
                    input_tokens = [pre_buffer, post_buffer]
                
                context_end = current_seq_len
                context_start = max(0, context_end - max_context)
                current_stamp = full_stamp[:, context_start:context_end, :].contiguous()
                
                s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
                s1_logits = s1_logits[:, -1, :]
                sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
                
                s2_logits = model.decode_s2(context, sample_pre)
                s2_logits = s2_logits[:, -1, :]
                sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)
                
                generated_pre[:, i] = sample_pre.squeeze(-1)
                generated_post[:, i] = sample_post.squeeze(-1)
                
                if current_seq_len < max_context:
                    pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                    post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
                else:
                    pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                    post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                    pre_buffer[:, -1] = sample_pre.squeeze(-1)
                    post_buffer[:, -1] = sample_post.squeeze(-1)
            
            full_pre = torch.cat([x_token[0], generated_pre], dim=1)
            full_post = torch.cat([x_token[1], generated_post], dim=1)
            
            context_start = max(0, total_seq_len - max_context)
            input_tokens = [
                full_pre[:, context_start:total_seq_len].contiguous(),
                full_post[:, context_start:total_seq_len].contiguous()
            ]
            z = tokenizer.decode(input_tokens, half=True)
            
            # 不做平均，直接收集 chunk 结果
            # z shape: (chunk_size, total_seq_len, 6)
            chunk_paths = z.cpu().numpy()
            all_paths.append(chunk_paths)
            
            remaining -= chunk_size
    
    # 拼接所有 chunk，取预测部分
    # all_paths: list of (chunk_size, total_seq_len, 6)
    # concat -> (sample_count, total_seq_len, 6)
    all_paths = np.concatenate(all_paths, axis=0)
    
    # 返回预测部分: (sample_count, pred_len, 6)
    return all_paths[:, -pred_len:, :]


def analyze_distribution(paths, current_price, name, code):
    """
    paths: (sample_count, pred_len, 6)  normalized raw paths from generate_distribution
    current_price: float, latest close price
    name, code: stock info for display

    Returns dict with all statistics and trading signal
    """
    n = paths.shape[0]
    close_paths = paths[:, :, 3]         # (n, pred_len)  收盘价
    final_prices = close_paths[:, -1]     # (n,)           期末价格

    # 1. 期末价格统计
    mean_final = float(np.mean(final_prices))
    median_final = float(np.median(final_prices))
    std_final = float(np.std(final_prices))
    min_final = float(np.min(final_prices))
    max_final = float(np.max(final_prices))

    # 2. 涨跌概率
    up_count = int(np.sum(final_prices > current_price))
    down_count = n - up_count
    up_prob = up_count / n * 100

    # 3. VaR (期末收盘价回报的百分位数)
    returns = final_prices / current_price - 1
    var_95 = float(current_price * (1 + np.percentile(returns, 5)))
    var_99 = float(current_price * (1 + np.percentile(returns, 1)))

    # 4. 期望收益
    expected_return = float((mean_final / current_price - 1) * 100)
    median_return = float((median_final / current_price - 1) * 100)

    # 5. 每条路径的最大回撤
    max_drawdowns = []
    for p in close_paths:
        peak = np.maximum.accumulate(p)
        dd = (p - peak) / (peak + 1e-8)
        max_drawdowns.append(float(np.min(dd)))
    avg_max_drawdown = float(np.mean(max_drawdowns) * 100)
    worst_drawdown = float(np.min(max_drawdowns) * 100)

    # 6. 均值路径 + 80% 置信带 (pointwise)
    mean_path = np.mean(close_paths, axis=0)
    p10_path = np.percentile(close_paths, 10, axis=0)
    p90_path = np.percentile(close_paths, 90, axis=0)

    # 7. 四因子综合评分 ( -1 ~ +1 )
    # 方向因子 (35%): 涨的概率越高越好
    direction_score = (up_prob - 50) / 50
    # 收益因子 (25%): 期望收益归一化到 [-1, +1]
    return_score = min(max(expected_return / 5, -1), 1)
    # 稳定性因子 (20%): 低波动得高分
    stability_score = 1 - min(std_final / (abs(mean_final) + 1e-8) * 5, 1)
    # 尾部风险因子 (20%): 尾部风险越小越好
    tail_score = 1 - min(abs(var_95 / current_price - 1) * 10, 1)

    composite_score = (
        direction_score * 0.35 +
        return_score * 0.25 +
        stability_score * 0.20 +
        tail_score * 0.20
    )

    # 8. 信号判断
    if composite_score > 0.4 and up_prob > 60 and expected_return > 2:
        signal = f"BUY  (score {composite_score:+.2f})"
        suggestion = f"Strong upward bias. Consider position {(up_prob/100 - 0.5) * 2 * 100:.0f}%, stop-loss at {var_95:.2f}"
    elif composite_score > 0.15 and up_prob > 50:
        signal = f"HOLD (score {composite_score:+.2f})"
        suggestion = "Mild positive signal. Recommend watching or light position"
    elif composite_score < -0.2:
        signal = f"SELL (score {composite_score:+.2f})"
        suggestion = f"Downside risk elevated (VaR95={var_95:.2f}). Consider reducing exposure"
    else:
        signal = f"WAIT (score {composite_score:+.2f})"
        suggestion = "No clear signal. Stay on the sidelines"

    return {
        "n_samples": n,
        "up_prob": up_prob, "up_count": up_count, "down_count": down_count,
        "mean_final": mean_final, "median_final": median_final, "std_final": std_final,
        "min_final": min_final, "max_final": max_final,
        "var_95": var_95, "var_99": var_99,
        "expected_return": expected_return, "median_return": median_return,
        "avg_max_drawdown": avg_max_drawdown, "worst_drawdown": worst_drawdown,
        "mean_path": mean_path, "p10_path": p10_path, "p90_path": p90_path,
        "composite_score": composite_score,
        "signal": signal, "suggestion": suggestion,
    }


# ============ 主流程 ============
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 64)
    print("  Kronos-base Distribution Forecast + Trading Signals")
    print(f"  Sample paths per stock: {SAMPLE_COUNT}")
    print("=" * 64)
    print("\n[1/5] Loading Kronos-base model (GPU)...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    tokenizer.eval()
    model.eval()
    tokenizer.to(DEVICE)
    model.to(DEVICE)
    print(f"      Model loaded, device: {DEVICE}")
    print("\nTODO: Tasks 2-3 — distribution analysis, plotting, and per-stock loop not yet implemented")


if __name__ == "__main__":
    main()
