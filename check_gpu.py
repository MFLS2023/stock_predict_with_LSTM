"""
GPU 诊断和测试脚本
用于验证PyTorch GPU安装是否正确
"""

import sys

def check_pytorch_gpu():
    print("=" * 60)
    print("PyTorch GPU 诊断工具")
    print("=" * 60)
    
    # 1. 检查PyTorch是否安装
    try:
        import torch
        print(f"✅ PyTorch已安装")
        print(f"   版本: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch未安装！")
        print("   请运行: pip install torch torchvision torchaudio")
        return False
    
    # 2. 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ CUDA可用")
        print(f"   CUDA版本: {torch.version.cuda}")
    else:
        print(f"❌ CUDA不可用")
        if '+cpu' in torch.__version__:
            print(f"   原因: 你安装的是CPU版本的PyTorch ({torch.__version__})")
            print(f"   解决方案: 请查看 INSTALL_PYTORCH_GPU.md 重新安装")
        else:
            print(f"   原因: 未知（可能是驱动问题）")
        return False
    
    # 3. 检查GPU设备
    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 个GPU设备")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_capability = torch.cuda.get_device_capability(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"\n   GPU {i}:")
        print(f"     名称: {gpu_name}")
        print(f"     计算能力: {gpu_capability[0]}.{gpu_capability[1]}")
        print(f"     显存: {gpu_memory:.2f} GB")
    
    # 4. 测试GPU计算
    print("\n" + "=" * 60)
    print("GPU 计算测试")
    print("=" * 60)
    
    try:
        import time
        
        # CPU测试
        size = 5000
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start = time.perf_counter()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.perf_counter() - start
        print(f"CPU计算 ({size}x{size} 矩阵乘法): {cpu_time:.4f} 秒")
        
        # GPU测试
        device = torch.device("cuda:0")
        a_gpu = a_cpu.to(device)
        b_gpu = b_cpu.to(device)
        
        # 预热
        _ = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start
        
        print(f"GPU计算 ({size}x{size} 矩阵乘法): {gpu_time:.4f} 秒")
        print(f"🚀 加速比: {cpu_time/gpu_time:.2f}x")
        
        # 验证结果一致性
        c_gpu_cpu = c_gpu.cpu()
        diff = torch.abs(c_cpu - c_gpu_cpu).max().item()
        print(f"✅ 结果差异: {diff:.2e} (应该接近0)")
        
    except Exception as e:
        print(f"❌ GPU计算测试失败: {e}")
        return False
    
    # 5. 检查cuDNN
    print("\n" + "=" * 60)
    print("cuDNN 状态")
    print("=" * 60)
    cudnn_available = torch.backends.cudnn.enabled
    if cudnn_available:
        print(f"✅ cuDNN已启用")
        print(f"   版本: {torch.backends.cudnn.version()}")
    else:
        print(f"⚠️  cuDNN未启用")
    
    print("\n" + "=" * 60)
    print("✅ 所有检查通过！PyTorch GPU配置正确")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = check_pytorch_gpu()
    sys.exit(0 if success else 1)
