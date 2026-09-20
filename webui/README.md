# Kronos Web UI

Web user interface for Kronos financial prediction model, providing intuitive graphical operation interface.

## ✨ Features

- **Multi-format data support**: Supports CSV, Feather and other financial data formats
- **Smart time window**: Fixed 400+120 data point time window slider selection
- **Real model prediction**: Integrated real Kronos model, supports multiple model sizes
- **Prediction quality control**: Adjustable temperature, nucleus sampling, sample count and other parameters
- **Multi-device support**: Supports CPU, CUDA, MPS and other computing devices
- **Comparison analysis**: Detailed comparison between prediction results and actual data
- **K-line chart display**: Professional financial K-line chart display

## 启动

推荐从项目根目录使用安全启动入口：

### Windows 一键启动

```powershell
.\start.bat
```

`start.bat` 从自身路径解析项目根目录，优先使用当前项目环境或 Conda 环境 `kronos`，
检查 Flask、NumPy、Pandas、Plotly、PyYAML、scikit-learn 是否可用。缺少 Python 或依赖
时会安全退出并打印手工命令，不会自动安装、下载 Python/uv 或修改 PATH。可用
`KRONOS_CONDA_ENV` 指定其他 Conda 环境。

### 手动启动

先手工安装依赖，再启动 WebUI：

```powershell
python -m pip install -r requirements.txt
python webui/run.py
```

`webui/run.py` 可从脚本位置加载项目根目录下的应用，读取 `decision/config.yaml` 中的
`webui.host` 和 `webui.port`，默认只监听 `127.0.0.1:7070`，关闭调试和自动重载。

仅检查启动条件、不启动长期服务：

```powershell
$env:KRONOS_STARTUP_CHECK_ONLY = '1'
.\start.bat
```

检查模式下缺依赖以非零码退出，只打印手工安装命令。旧的 `webui/start.sh` 不属于本次
安全启动入口，本文不将其作为推荐方式。

启动成功后访问 <http://127.0.0.1:7070/report>。


## 📋 Usage Steps

1. **Load data**: Select financial data file from data directory
2. **Load model**: Select Kronos model and computing device
3. **Set parameters**: Adjust prediction quality parameters
4. **Select time window**: Use slider to select 400+120 data point time range
5. **Start prediction**: Click prediction button to generate results
6. **View results**: View prediction results in charts and tables

## 🔧 Prediction Quality Parameters

### Temperature (T)
- **Range**: 0.1 - 2.0
- **Effect**: Controls prediction randomness
- **Recommendation**: 1.2-1.5 for better prediction quality

### Nucleus Sampling (top_p)
- **Range**: 0.1 - 1.0
- **Effect**: Controls prediction diversity
- **Recommendation**: 0.95-1.0 to consider more possibilities

### Sample Count
- **Range**: 1 - 5
- **Effect**: Generate multiple prediction samples
- **Recommendation**: 2-3 samples to improve quality

## 📊 Supported Data Formats

### Required Columns
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price

### Optional Columns
- `volume`: Trading volume
- `amount`: Trading amount (not used for prediction)
- `timestamps`/`timestamp`/`date`: Timestamp

## 🤖 Model Support

- **Kronos-mini**: 4.1M parameters, lightweight fast prediction
- **Kronos-small**: 24.7M parameters, balanced performance and speed
- **Kronos-base**: 102.3M parameters, high quality prediction

## 🖥️ GPU Acceleration Support

- **CPU**: General computing, best compatibility
- **CUDA**: NVIDIA GPU acceleration, best performance
- **MPS**: Apple Silicon GPU acceleration, recommended for Mac users

## ⚠️ Notes

- `amount` column is not used for prediction, only for display
- Time window is fixed at 400+120=520 data points
- Ensure data file contains sufficient historical data
- First model loading may require download, please be patient

## 🔍 Comparison Analysis

The system automatically provides comparison analysis between prediction results and actual data, including:
- Price difference statistics
- Error analysis
- Prediction quality assessment

## 🛠️ Technical Architecture

- **Backend**: Flask + Python
- **Frontend**: HTML + CSS + JavaScript
- **Charts**: Plotly.js
- **Data processing**: Pandas + NumPy
- **Model**: Hugging Face Transformers

## 故障排查

### 常见问题
1. **端口被占用**：修改 `decision/config.yaml` 的 `webui.port` 后重新启动。
2. **缺少依赖**：手工执行 `python -m pip install -r requirements.txt`，然后重试。
3. **模型加载失败**：检查模型缓存和模型标识；WebUI 启动检查本身不会自动下载依赖。
4. **数据格式错误**：确保数据文件包含必需的列名和格式。

### 日志

启动检查和运行时信息会输出到控制台，错误信息包含中文诊断和可恢复步骤。

## 📄 License

This project follows the license terms of the original Kronos project.

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve this Web UI!

## 📞 Support

If you have questions, please check:
1. Project documentation
2. GitHub Issues
3. Console error messages
