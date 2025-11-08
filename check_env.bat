@echo off
REM 快速环境检查脚本
echo ========================================
echo 环境检查
echo ========================================
echo.

echo [检查1] Python版本
python --version
echo.

echo [检查2] CUDA驱动
nvidia-smi | findstr "CUDA Version"
echo.

echo [检查3] PyTorch安装状态
python -c "import torch; print('✅ PyTorch已安装:', torch.__version__)" 2>nul || echo ❌ PyTorch未安装
echo.

echo [检查4] CUDA可用性
python -c "import torch; print('✅ CUDA可用' if torch.cuda.is_available() else '❌ CUDA不可用')" 2>nul || echo ❌ 无法检查（PyTorch未安装）
echo.

echo ========================================
echo 建议：
echo.
echo 如果看到 "❌ PyTorch未安装" 或 "❌ CUDA不可用"
echo 请参考 Readme.md 中的安装说明手动配置环境。
echo ========================================
pause
