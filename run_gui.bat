@echo off
REM 启动股票预测GUI（自动激活GPU环境）
echo ========================================
echo 股票预测与回测平台
echo ========================================
echo.

REM 检查stock_gpu环境是否存在
conda env list | findstr "stock_gpu" >nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: 未找到名为 stock_gpu 的conda环境
    echo.
    echo 请先按照 Readme.md 中的 "快速开始" 指南创建环境并安装依赖:
    echo    conda create -n stock_gpu python=3.11 -y
    echo    conda activate stock_gpu
    echo    pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo ✅ 激活GPU环境...
call conda activate stock_gpu

echo ✅ 启动GUI程序...
python gui.py

REM 程序退出后
echo.
echo 程序已关闭
pause
