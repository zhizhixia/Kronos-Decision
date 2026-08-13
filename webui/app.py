import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify, send_file
import sys
import warnings
import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
warnings.filterwarnings('ignore')

# Add project root directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)


@app.before_request
def reject_foreign_write_origin() -> Any | None:
    """拒绝来自非本机网页的写请求，避免本地 API 被跨站调用。"""
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return None
    origin = request.headers.get("Origin")
    if not origin:
        return None
    hostname = (urlparse(origin).hostname or "").lower()
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return None
    error = {"code": "FOREIGN_ORIGIN_BLOCKED", "message": "仅允许本机页面调用写接口。", "retryable": False, "details": {}}
    return jsonify({"status": "error", "error": error}), 403

# Global variables to store models
tokenizer = None
model = None
predictor = None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def load_data_files() -> list[dict[str, str]]:
    """Scan data directory and return available data files"""
    data_files: list[dict[str, str]] = []
    if DATA_DIRECTORY.exists():
        for file_path in sorted(DATA_DIRECTORY.glob("*.csv")):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                data_files.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    return data_files

def load_data_file(file_path: str) -> tuple[pd.DataFrame | None, str | None]:
    """Load data file"""
    try:
        allowed = DATA_DIRECTORY.resolve(strict=True)
        candidate = Path(file_path).resolve(strict=True)
        if not candidate.is_relative_to(allowed):
            return None, "DATA_FILE_OUTSIDE_ALLOWED_DIRECTORY"
        if candidate.suffix.lower() != ".csv" or not candidate.is_file():
            return None, "UNSUPPORTED_DATA_FILE"
        df = pd.read_csv(candidate)
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows containing NaN values
        df = df.dropna()
        
        return df, None
        
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as e:
        return None, f"Failed to load file: {str(e)}"


def _future_timestamps(frame: pd.DataFrame, periods: int) -> pd.DatetimeIndex:
    """按文件自身频率生成最后可见时间点之后的时间戳。"""
    values = pd.DatetimeIndex(pd.to_datetime(frame["timestamps"])).sort_values()
    if len(values) < 2:
        return pd.date_range(values[-1] + pd.Timedelta(days=1), periods=periods, freq="D")
    inferred = pd.infer_freq(values[-min(20, len(values)):])
    if inferred:
        return pd.date_range(values[-1], periods=periods + 1, freq=inferred)[1:]
    step = pd.Series(values).diff().dropna().median()
    return pd.date_range(values[-1] + step, periods=periods, freq=step)

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='Historical Data (400 data points)',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Calculate prediction data timestamps - ensure continuity with historical data
        if 'timestamps' in df.columns and len(historical_df) > 0:
            # Start from the last timestamp of historical data, create prediction timestamps with the same time interval
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            # If no timestamps, use index
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name='Prediction Data (120 data points)',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # Actual data should be in the same time period as prediction data
        if 'timestamps' in df.columns:
            # Actual data should use the same timestamps as prediction data to ensure time alignment
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                # If no prediction timestamps, calculate from the last timestamp of historical data
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name='Actual Data (120 data points)',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))
    
    # Update layout
    fig.update_layout(
        title='Kronos Financial Prediction Results - 400 Historical Points + 120 Prediction Points vs 120 Actual Points',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """重定向到决策报告页面。"""
    from flask import redirect
    return redirect('/report')

@app.route('/assets/plotly.min.js')
def plotly_asset() -> Any:
    """从本地安装的 plotly 包返回压缩版 JS，避免使用 CDN。"""
    import plotly
    asset_path = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
    if not asset_path.exists():
        error = {"code": "ASSET_NOT_FOUND", "message": "本地 plotly.min.js 不可用。", "retryable": False, "details": {}}
        return jsonify({"status": "error", "error": error}), 404
    response = send_file(str(asset_path), mimetype="application/javascript")
    response.headers["Cache-Control"] = "public, max-age=86400"
    return response

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        # Load data
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        if len(df) < lookback:
            return jsonify({'error': f'Insufficient data length, need at least {lookback} rows'}), 400
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # Process time period selection
                start_date = data.get('start_date')
                
                if start_date:
                    # Custom time period - fix logic: use data within selected window
                    start_dt = pd.to_datetime(start_date)
                    
                    # Find data after start time
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    # Ensure sufficient data: lookback + pred_len
                    if len(time_range_df) < lookback + pred_len:
                        return jsonify({'error': f'Insufficient data from start time {start_dt.strftime("%Y-%m-%d %H:%M")}, need at least {lookback + pred_len} data points, currently only {len(time_range_df)} available'}), 400
                    
                    # Use first lookback data points within selected window for prediction
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    
                    # Use last pred_len data points within selected window as actual values
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    
                    # Calculate actual time period length
                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp
                    
                    prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
                else:
                    # Use latest data
                    x_df = df.iloc[-lookback:][required_cols]
                    x_timestamp = df.iloc[-lookback:]['timestamps']
                    y_timestamp = pd.Series(_future_timestamps(df, pred_len), name='timestamps')
                    prediction_type = "Kronos model prediction (latest data)"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )
                
            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        
        # Prepare actual data for comparison (if exists)
        actual_data = []
        actual_df = None
        
        if start_date:  # Custom time period
            # Fix logic: use data within selected window
            # Prediction uses first 400 data points within selected window
            # Actual data should be last 120 data points within selected window
            start_dt = pd.to_datetime(start_date)
            
            # Find data starting from start_date
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]
            
            if len(time_range_df) >= lookback + pred_len:
                # Get last 120 data points within selected window as actual values
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        else:  # Latest data
            # 最新窗口之后没有已知真实值，不伪造历史对照。
            actual_df = None
        
        # Create chart - pass historical data start position
        if start_date:
            # Custom time period: find starting position of historical data in original df
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            # Latest data: display the same trailing window used by the model
            historical_start_idx = max(0, len(df) - lookback)
        
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
        
        # Prepare prediction result data - fix timestamp calculation logic
        if 'timestamps' in df.columns:
            if start_date:
                # Custom time period: use selected window data to calculate timestamps
                start_dt = pd.to_datetime(start_date)
                mask = df['timestamps'] >= start_dt
                time_range_df = df[mask]
                
                if len(time_range_df) >= lookback:
                    # Calculate prediction timestamps starting from last time point of selected window
                    last_timestamp = time_range_df['timestamps'].iloc[lookback-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                    future_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                else:
                    future_timestamps = []
            else:
                future_timestamps = _future_timestamps(df, pred_len)
        else:
            future_timestamps = range(len(df), len(df) + pred_len)
        
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest'
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

# === 决策报告 API ===

@app.route("/report")
def report_page():
    """v2 五态决策报告页面（主入口）。"""
    return render_template("report_v2.html")


@app.route("/report/legacy")
def report_page_legacy():
    """v1 兼容页面：允许修改采样参数，仅供旧行为对照。"""
    return render_template("report.html", show_migration_banner=True)


@app.route("/portfolio")
def portfolio_page():
    """本地组合页面（不提供下单控件）。"""
    return render_template("portfolio.html")


@app.route("/research")
def research_page():
    """正式研究评估页面。"""
    return render_template("research.html")


@app.route("/settings")
def settings_page():
    """高级设置页面：采样参数只读展示，修改会使证据档案失效。"""
    return render_template("settings.html")


@app.route("/api/decision-report", methods=["POST"])
def api_decision_report():
    """生成决策报告。

    请求: {"stock_code": "600519", "pred_len": 120, ...}
    响应: DecisionReport 的 JSON 序列化
    """
    from decision.engine import DecisionEngine
    import dataclasses, json

    data = request.get_json() or {}
    stock_code = data.get("stock_code", "")
    if not stock_code:
        return jsonify({"status": "error", "error_message": "请提供股票代码"}), 400

    engine = DecisionEngine()
    report = engine.predict_and_analyze(stock_code, data)
    return jsonify(dataclasses.asdict(report))


@app.route("/api/stock-list")
def api_stock_list():
    """获取股票池列表。"""
    try:
        from data.pool import get_hs300_pool
        pool = get_hs300_pool()
        stocks = [{"code": k, "name": v} for k, v in pool.items()]
        return jsonify({"status": "ok", "stocks": stocks})
    except Exception as e:
        return jsonify({"status": "error", "error_message": str(e)}), 500


# === v2 可审计研究 API ===

@app.route("/api/v2/decision-report", methods=["POST"])
def api_v2_decision_report():
    """生成固定采样配置的五态决策报告。"""
    from decision.engine import DecisionEngine
    from decision.errors import PredictionTimeoutError
    from decision.v2 import error_payload

    payload = request.get_json(silent=True) or {}
    stock_code = str(payload.get("stock_code", "")).strip()
    if len(stock_code) != 6 or not stock_code.isdigit():
        return jsonify(error_payload("INVALID_REQUEST", "stock_code 必须是六位股票代码", False)), 400
    requested_as_of = payload.get("as_of")
    if requested_as_of is not None:
        if not isinstance(requested_as_of, str):
            return jsonify(error_payload("INVALID_REQUEST", "as_of 必须是 YYYY-MM-DD 日期字符串或 null", False)), 400
        try:
            requested_as_of = pd.Timestamp(requested_as_of).normalize()
            if pd.isna(requested_as_of):
                raise ValueError("NaT")
        except (TypeError, ValueError):
            return jsonify(error_payload("INVALID_REQUEST", "as_of 必须是可识别的日期", False)), 400
    try:
        report = DecisionEngine().decision_report_v2(stock_code, str(payload.get("portfolio_id", "default")), bool(payload.get("include_display_paths", False)), requested_as_of)
        try:
            from portfolio.store import PortfolioStore
            version_match = bool((report["evidence_gate"].get("checks") or {}).get("VERSION_MATCH"))
            PortfolioStore().save_recommendation(
                str(payload.get("portfolio_id", "default")),
                stock_code,
                report["recommendation"]["action"],
                report["horizons"],
                "gate-v1" if version_match else "no-evidence",
                report["generated_at"],
                report["data_provenance"].get("as_of"),
                bool(report["evidence_gate"].get("passed")),
            )
        except Exception:
            pass  # 建议历史写入失败不阻断报告
        return jsonify(report)
    except PredictionTimeoutError as exc:
        return jsonify(error_payload("PREDICTION_TIMEOUT", "模型推理超过配置的时间预算", True, {"cause": str(exc)})), 504
    except Exception as exc:
        return jsonify(error_payload("DATA_UNAVAILABLE", "无法取得可验证的完整日线数据", True, {"cause": str(exc)})), 503


@app.route("/api/v2/evaluation/latest")
def api_v2_evaluation_latest():
    """返回最近一次可追溯评估运行的清单与门禁结果。"""
    from decision.v2 import error_payload
    from evaluation.binding import EvaluationBinding

    binding = EvaluationBinding.load_latest()
    if binding is None:
        return jsonify(error_payload("EVALUATION_NOT_FOUND", "尚无正式评估运行", False)), 404
    result = {"status": "ok", "run_id": binding.run_id, "manifest": binding.manifest, "gate_result": binding.gate_result}
    return jsonify(result)


@app.route("/api/v2/forward-ledger/latest")
def api_v2_forward_ledger_latest():
    """返回与历史回测隔离的最新每日建议前瞻台账。"""
    from decision.v2 import error_payload
    from evaluation.forward_ledger import load_latest_forward_report

    latest = load_latest_forward_report()
    if latest is None:
        return jsonify(error_payload("FORWARD_LEDGER_NOT_FOUND", "尚无每日建议前瞻台账", False)), 404
    return jsonify({"status": "ok", **latest})


@app.route("/api/v2/portfolio", methods=["GET", "PUT"])
def api_v2_portfolio():
    """读取或更新本地 SQLite 持仓，不保存外部交易凭据。"""
    from portfolio.store import PortfolioStore
    from decision.v2 import error_payload

    profile_id = request.args.get("portfolio_id", "default")
    store = PortfolioStore()
    try:
        if request.method == "PUT":
            return jsonify({"status": "ok", "portfolio": store.replace_portfolio(request.get_json(silent=True) or {}, profile_id)})
        return jsonify({"status": "ok", "portfolio": store.get_portfolio(profile_id)})
    except ValueError as exc:
        return jsonify(error_payload("INVALID_PORTFOLIO", str(exc), False)), 400


@app.route("/api/v2/portfolio/rebalance", methods=["POST"])
def api_v2_portfolio_rebalance():
    """门禁通过时生成模拟目标权重与订单；永不执行真实交易。"""
    from decision.v2 import error_payload
    from evaluation.binding import EvaluationBinding
    from portfolio.service import RebalanceService
    from data.fetcher import DataFetcher

    profile_id = (request.get_json(silent=True) or {}).get("portfolio_id", "default")
    binding = EvaluationBinding.load_latest()
    fetcher = DataFetcher()

    def fetch_bars(code: str):
        try:
            return fetcher.fetch_daily(code)
        except Exception:
            return None

    result = RebalanceService().rebalance(binding, profile_id, fetch_bars)
    if result["status"] == "ok":
        return jsonify({"status": "ok", "rebalance": result})
    return jsonify(error_payload("INSUFFICIENT_EVIDENCE", "证据不足，无法生成模拟调仓", False, {"reason_codes": result["reason_codes"]})), 409


@app.route("/api/v2/recommendations")
def api_v2_recommendations():
    """返回本地保存的建议历史。"""
    from portfolio.store import PortfolioStore
    from decision.v2 import error_payload

    profile_id = request.args.get("portfolio_id", "default")
    try:
        limit = int(request.args.get("limit", 100))
        if not 1 <= limit <= 500:
            raise ValueError("limit out of range")
        return jsonify({"status": "ok", "recommendations": PortfolioStore().list_recommendations(profile_id, limit)})
    except ValueError:
        return jsonify(error_payload("INVALID_REQUEST", "limit 必须为数字", False)), 400


@app.route("/api/config", methods=["GET", "PUT"])
def api_config():
    """获取或更新配置。"""
    from copy import deepcopy

    from decision.config import get_config, save_config
    import dataclasses

    if request.method == "GET":
        cfg = get_config()
        return jsonify(dataclasses.asdict(cfg))

    if request.method == "PUT":
        try:
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                raise ValueError("配置请求必须是 JSON 对象。")
            cfg = deepcopy(get_config())
            for section, values in data.items():
                if not hasattr(cfg, section) or not isinstance(values, dict):
                    raise ValueError(f"未知或无效的配置分组：{section}")
                sub = getattr(cfg, section)
                for key, val in values.items():
                    if not hasattr(sub, key):
                        raise ValueError(f"未知配置项：{section}.{key}")
                    setattr(sub, key, val)
            save_config(cfg)
            return jsonify({"status": "ok", "message": "配置已更新"})
        except Exception as e:
            return jsonify({"status": "error", "error_message": str(e)}), 400


if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    from decision.config import get_config

    web_config = get_config().webui
    app.run(debug=False, host=web_config.host, port=web_config.port, use_reloader=False)
