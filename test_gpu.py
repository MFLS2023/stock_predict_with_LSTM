import torch
import sys

print("=" * 60)
print("PyTorch GPU 配置测试")
print("=" * 60)
print()

# 基本信息
print("🐍 Python信息:")
print(f"  版本: {sys.version}")
print(f"  可执行文件: {sys.executable}")
print()

# PyTorch信息
print("🔥 PyTorch信息:")
print(f"  版本: {torch.__version__}")
print(f"  安装路径: {torch.__file__}")
print()

# CUDA信息
print("🎮 CUDA信息:")
print(f"  CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA版本: {torch.version.cuda}")
    print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"  GPU数量: {torch.cuda.device_count()}")
    print()
    
    # 每个GPU的详细信息
    print("🖥️  GPU设备信息:")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}:")
        print(f"    名称: {torch.cuda.get_device_name(i)}")
        print(f"    显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"    计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    print()
    
    # 测试GPU计算
    print("🧪 GPU计算测试:")
    try:
        # 创建测试张量
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        
        # 矩阵乘法
        import time
        start = time.time()
        z = torch.mm(x, y)
        torch.cuda.synchronize()  # 等待GPU完成
        end = time.time()
        
        print(f"  ✅ 矩阵乘法测试成功！")
        print(f"  计算时间: {(end - start) * 1000:.2f} ms")
        print(f"  结果形状: {z.shape}")
        print(f"  当前GPU内存使用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  峰值GPU内存使用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print()
        
        # 清理
        del x, y, z
        torch.cuda.empty_cache()
        
        print("=" * 60)
        print("✅ 所有测试通过！GPU配置正常！")
        print("=" * 60)
        print()
        print("💡 提示:")
        print("  - 您可以在训练设置中选择具体的GPU")
        print("  - 对于小模型，GPU和CPU性能差异不大")
        print("  - 对于大数据集和复杂模型，GPU会显著加速训练")
        
    except Exception as e:
        print(f"  ❌ GPU计算测试失败: {e}")
        print()
        print("建议:")
        print("  1. 检查CUDA驱动是否正确安装")
        print("  2. 尝试重启计算机")
        print("  3. 更新显卡驱动")

else:
    print("  ❌ CUDA不可用")
    print()
    print("可能的原因:")
    print("  1. 未安装GPU版本的PyTorch")
    print("  2. 显卡驱动未正确安装")
    print("  3. CUDA工具包未安装")
    print()
    print("解决方案:")
    print("  1. 运行: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("  2. 或参考 PYTORCH_GPU_INSTALL.md 文档")

print()
