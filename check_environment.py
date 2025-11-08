#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证脚本 - 检查所有依赖是否正确安装
"""

import sys

def check_imports():
    """检查所有必需的包是否可以导入"""
    print("=" * 60)
    print("依赖包导入检查")
    print("=" * 60)
    print()
    
    packages = [
        ("Python", sys.version),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Matplotlib", "matplotlib"),
        ("scikit-learn", "sklearn"),
        ("PyTorch", "torch"),
        ("TensorFlow", "tensorflow"),
        ("Keras", "keras"),
        ("mplfinance", "mplfinance"),
        ("PyQt6", "PyQt6"),
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for name, module_name in packages:
        if name == "Python":
            print(f"✅ {name}: {module_name}")
            success_count += 1
            continue
            
        try:
            if module_name == "torch":
                import torch
                cuda_info = f" (CUDA {torch.version.cuda})" if torch.cuda.is_available() else " (CPU only)"
                print(f"✅ {name}: {torch.__version__}{cuda_info}")
            elif module_name == "tensorflow":
                import tensorflow as tf
                gpu_info = f" ({len(tf.config.list_physical_devices('GPU'))} GPU)" if tf.config.list_physical_devices('GPU') else " (CPU only)"
                print(f"✅ {name}: {tf.__version__}{gpu_info}")
            elif module_name == "PyQt6":
                from PyQt6.QtCore import QT_VERSION_STR
                print(f"✅ {name}: {QT_VERSION_STR}")
            else:
                mod = __import__(module_name)
                version = getattr(mod, "__version__", "未知版本")
                print(f"✅ {name}: {version}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name}: 导入失败 - {e}")
        except Exception as e:
            print(f"⚠️  {name}: 导入成功但出现警告 - {e}")
            success_count += 1
    
    print()
    print("=" * 60)
    print(f"检查完成: {success_count}/{total_count} 包可用")
    print("=" * 60)
    
    if success_count == total_count:
        print("\n🎉 所有依赖包检查通过！环境配置正确！")
        return True
    else:
        print(f"\n⚠️  有 {total_count - success_count} 个包缺失或导入失败")
        return False

def check_gpu():
    """检查GPU支持"""
    print()
    print("=" * 60)
    print("GPU支持检查")
    print("=" * 60)
    print()
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch GPU: 可用")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("ℹ️  PyTorch GPU: 不可用（将使用CPU）")
    except Exception as e:
        print(f"❌ PyTorch GPU检查失败: {e}")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow GPU: 可用")
            print(f"   GPU数量: {len(gpus)}")
            for gpu in gpus:
                print(f"   {gpu.name}")
        else:
            print("ℹ️  TensorFlow GPU: 不可用（将使用CPU）")
    except Exception as e:
        print(f"❌ TensorFlow GPU检查失败: {e}")
    
    print()

if __name__ == "__main__":
    imports_ok = check_imports()
    check_gpu()
    
    if imports_ok:
        print("\n💡 提示: 现在可以运行 start_gpu.bat 或直接运行:")
        print(f"   {sys.executable} gui.py")
        sys.exit(0)
    else:
        print("\n❌ 环境配置不完整，请运行:")
        print("   setup_complete_environment.bat")
        sys.exit(1)
