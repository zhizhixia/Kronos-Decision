$ErrorActionPreference = "Stop"
$Workspace = Split-Path -Parent $PSScriptRoot
$PipTemp = Join-Path $Workspace "scripts\.pip-tmp"
New-Item -ItemType Directory -Path $PipTemp -Force | Out-Null
$env:TMP = $PipTemp
$env:TEMP = $PipTemp

Write-Host "使用清华镜像安装研究依赖（pyqlib / PyPortfolioOpt / scikit-learn / scipy）..."
python -m pip install "pyqlib==0.9.7" "PyPortfolioOpt>=1.5.6,<2.0" "scikit-learn>=1.3,<2.0" "scipy>=1.10,<2.0" `
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple `
  --cache-dir (Join-Path $PipTemp "cache")

Write-Host "安装完成。Qlib 中国数据请按官方文档准备："
Write-Host "  python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn"
