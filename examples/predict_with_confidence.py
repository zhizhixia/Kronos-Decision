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
from datetime import timedelta
import json
import warnings
warnings.filterwarnings("ignore")

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

import torch

from tqdm import trange

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
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
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


def plot_distribution(res, future_dates, name, code):
    """res dict from analyze_distribution; future_dates: list of datetime"""
    hist = res["historical"]
    current = res["current_price"]
    stats = res["stats"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Upper panel: price + confidence band
    ax1.plot(hist["timestamps"], hist["close"], color="#1f77b4", linewidth=1.2, label="Historical Close")
    ax1.plot(future_dates, stats["mean_path"], color="#ff7f0e", linewidth=2.0,
             label=f"Mean Prediction ({SAMPLE_COUNT} paths)")
    ax1.fill_between(future_dates, stats["p10_path"], stats["p90_path"],
                     alpha=0.2, color="#ff7f0e", label="80% Confidence Band")
    ax1.axhline(y=current, color="gray", linestyle=":", alpha=0.5)
    
    ax1.annotate(f"Current: {current:.2f}\nTarget: {stats['mean_final']:.2f} ({stats['expected_return']:+.1f}%)\n"
                 f"Up Prob: {stats['up_prob']:.0f}%  CI: [{stats['p10_path'][-1]:.2f}, {stats['p90_path'][-1]:.2f}]",
                 xy=(future_dates[len(future_dates)//2], stats["p90_path"][len(future_dates)//2]),
                 fontsize=9, color="#ff7f0e",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
    
    ax1.set_title(f"{name} ({code}) - Distribution Forecast [{stats['signal']}]", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price (CNY)", fontsize=11)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Lower panel: terminal price distribution histogram
    ax2.hist(res["all_final_prices"], bins=20, color="#ff7f0e", alpha=0.7, edgecolor="white")
    ax2.axvline(x=current, color="gray", linestyle="--", linewidth=1.5, label=f"Current ({current:.2f})")
    ax2.axvline(x=stats["mean_final"], color="#2ca02c", linestyle="-", linewidth=2,
                label=f"Mean ({stats['mean_final']:.2f})")
    ax2.axvline(x=stats["var_95"], color="red", linestyle=":", linewidth=1.5,
                label=f"VaR 95% ({stats['var_95']:.2f})")
    ax2.set_xlabel("Predicted Final Price (CNY)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title(f"Terminal Price Distribution", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, f"{code}_{name}_confidence_chart.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return chart_path


# ============ 主流程 ============
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 64)
    print("  Kronos-base Distribution Forecast + Trading Signals")
    print(f"  Sample paths per stock: {SAMPLE_COUNT}")
    print("=" * 64)
    print("\n[1/3] Loading Kronos-base model (GPU)...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    tokenizer.eval()
    model.eval()
    tokenizer.to(DEVICE)
    model.to(DEVICE)
    print(f"      Model loaded, device: {DEVICE}")
    
    summary_rows = []
    
    # ============ Per-stock loop ============
    for i, stock in enumerate(STOCKS):
        code, name = stock["code"], stock["name"]
        print(f"\n{'=' * 64}")
        print(f"[2/3] Processing {name} ({code}) [{i+1}/{len(STOCKS)}]")
        print("=" * 64)
        
        # 1. Fetch data
        try:
            df = fetch_stock_data(code)
        except Exception as e:
            print(f"  ERROR fetching data: {e}")
            continue
        
        if len(df) < LOOKBACK + 10:
            print(f"  SKIP: Insufficient data ({len(df)} < {LOOKBACK})")
            continue
        
        # 2. Prepare inputs
        x_raw = df.iloc[-LOOKBACK:][["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
        x_ts = df.iloc[-LOOKBACK:]["timestamps"].reset_index(drop=True)
        last_date = df["timestamps"].iloc[-1]
        future_dates = generate_future_dates(last_date, PRED_LEN)
        y_ts = pd.Series(future_dates)
        current_price = float(df["close"].iloc[-1])
        hist_df = df.iloc[-LOOKBACK:]
        
        print(f"  Current price: {current_price:.2f}")
        print(f"  Lookback: {LOOKBACK} days -> Predict: {PRED_LEN} trading days")
        
        # 3. Normalize (same as KronosPredictor.predict)
        x_mean = np.mean(x_raw, axis=0)
        x_std = np.std(x_raw, axis=0)
        x_norm = (x_raw - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -CLIP, CLIP)
        
        # 4. Compute timestamps
        x_stamp = calc_time_stamps(x_ts).values
        y_stamp = calc_time_stamps(y_ts).values
        
        # 5. Convert to torch tensors
        x_tensor = torch.from_numpy(x_norm).float().to(DEVICE)
        x_stamp_tensor = torch.from_numpy(x_stamp).float().to(DEVICE)
        y_stamp_tensor = torch.from_numpy(y_stamp).float().to(DEVICE)
        
        # 6. Generate distribution
        print(f"  Running inference ({SAMPLE_COUNT} paths)...")
        paths = generate_distribution(
            tokenizer, model,
            x_tensor, x_stamp_tensor, y_stamp_tensor,
            max_context=MAX_CONTEXT,
            pred_len=PRED_LEN,
            clip=CLIP,
            T=1.0,
            top_k=0,
            top_p=0.99,
            sample_count=SAMPLE_COUNT,
            paths_per_batch=PATHS_PER_BATCH,
            verbose=False,
        )
        
        # 7. Denormalize (reverse normalization per-path)
        # paths shape: (SAMPLE_COUNT, PRED_LEN, 6)
        paths_denorm = paths * (x_std + 1e-5) + x_mean
        
        # 8. Analyze distribution
        res = analyze_distribution(paths_denorm, current_price, name, code)
        
        # Build res dict for plot_distribution
        plot_res = {
            "historical": hist_df,
            "current_price": current_price,
            "stats": res,
            "all_final_prices": paths_denorm[:, -1, 3],  # final close prices
        }
        
        # 9. Plot distribution
        chart_path = plot_distribution(plot_res, future_dates, name, code)
        print(f"  Chart saved: {chart_path}")
        
        # 10. Save confidence CSV
        close_paths = paths_denorm[:, :, 3]  # (SAMPLE_COUNT, PRED_LEN)
        mean_path = np.mean(close_paths, axis=0)
        median_path = np.median(close_paths, axis=0)
        std_path = np.std(close_paths, axis=0)
        p10_path = np.percentile(close_paths, 10, axis=0)
        p90_path = np.percentile(close_paths, 90, axis=0)
        
        # Per-timestep up probability
        up_probs = np.mean(close_paths > current_price, axis=0) * 100
        
        conf_df = pd.DataFrame({
            "date": future_dates,
            "pred_close_mean": mean_path,
            "pred_close_median": median_path,
            "pred_close_std": std_path,
            "pred_close_p10": p10_path,
            "pred_close_p90": p90_path,
            "up_probability": up_probs,
        })
        conf_csv_path = os.path.join(OUTPUT_DIR, f"{code}_{name}_confidence.csv")
        conf_df.to_csv(conf_csv_path, index=False, encoding="utf-8-sig")
        print(f"  Confidence CSV saved: {conf_csv_path}")
        
        # 11. Save paths CSV (50 columns: path_1..path_50)
        paths_df = pd.DataFrame(
            close_paths.T,  # (PRED_LEN, SAMPLE_COUNT)
            columns=[f"path_{i+1}" for i in range(SAMPLE_COUNT)],
            index=future_dates,
        )
        paths_csv_path = os.path.join(OUTPUT_DIR, f"{code}_{name}_paths.csv")
        paths_df.to_csv(paths_csv_path, encoding="utf-8-sig")
        print(f"  Paths CSV saved: {paths_csv_path}")
        
        # 12. Print detailed stats
        print()
        print("  " + "-" * 58)
        print(f"  Distribution Analysis: {name} ({code})")
        print("  " + "-" * 58)
        print(f"  Current Price:          {current_price:.2f}")
        print(f"  Mean Predicted:         {res['mean_final']:.2f}  ({res['expected_return']:+.2f}%)")
        print(f"  Median Predicted:       {res['median_final']:.2f}  ({res['median_return']:+.2f}%)")
        print(f"  Std Dev:                {res['std_final']:.2f}")
        print(f"  80% CI:                 {res['p10_path'][-1]:.2f}  ~  {res['p90_path'][-1]:.2f}")
        print(f"  Best / Worst:           {res['max_final']:.2f}  /  {res['min_final']:.2f}")
        print(f"  Up Probability:         {res['up_prob']:.0f}%  ({res['up_count']}/{SAMPLE_COUNT} paths)")
        print(f"  VaR 95% / 99%:          {res['var_95']:.2f}  /  {res['var_99']:.2f}")
        print(f"  Avg Max Drawdown:       {res['avg_max_drawdown']:.1f}%")
        print(f"  Worst Drawdown:         {res['worst_drawdown']:.1f}%")
        print("  " + "-" * 58)
        print(f"  Composite Score:        {res['composite_score']:+.2f}")
        signal_word = res['signal'].split()[0]
        print(f"  >>> SIGNAL: {signal_word}")
        print(f"  >>> {res['suggestion']}")
        print("  " + "-" * 58)
        
        # Collect for summary
        summary_rows.append({
            "name": name,
            "code": code,
            "price": current_price,
            "target": res["mean_final"],
            "ret": res["expected_return"],
            "up_prob": res["up_prob"],
            "var_95": res["var_95"],
            "score": res["composite_score"],
            "signal": signal_word,
        })
    
    # ============ Final summary table ============
    print("\n" + "=" * 64)
    print("  Trading Signal Summary")
    print("=" * 64)
    print(f"{'Stock':<12} {'Price':>8} {'Target':>8} {'Ret%':>7} {'Up%':>5} {'VaR95':>8} {'Score':>7} {'Signal':<6}")
    print("-" * 64)
    for row in summary_rows:
        print(f"{row['name']:<12} {row['price']:>8.2f} {row['target']:>8.2f} {row['ret']:>+6.1f}% {row['up_prob']:>4.0f}% {row['var_95']:>8.2f} {row['score']:>+6.2f} {row['signal']:<6}")
    print("=" * 64)
    print(f"  Processed {len(summary_rows)}/{len(STOCKS)} stocks")
    print("  Done.")


if __name__ == "__main__":
    main()
