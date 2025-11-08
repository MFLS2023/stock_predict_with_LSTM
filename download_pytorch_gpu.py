"""
自动下载并安装PyTorch GPU版本
"""
import urllib.request
import os
import subprocess
import sys

# PyTorch CUDA 12.1 wheel文件的URL
WHEELS = {
    'torch': 'https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp313-cp313-win_amd64.whl',
    'torchvision': 'https://download.pytorch.org/whl/cu121/torchvision-0.20.1%2Bcu121-cp313-cp313-win_amd64.whl',
    'torchaudio': 'https://download.pytorch.org/whl/cu121/torchaudio-2.5.1%2Bcu121-cp313-cp313-win_amd64.whl',
}

DOWNLOAD_DIR = 'pytorch_wheels'

def download_file(url, filename):
    """下载文件并显示进度"""
    print(f"正在下载: {filename}")
    print(f"URL: {url}")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r下载进度: {percent}%")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✅ {filename} 下载完成")
        return True
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False

def main():
    print("=" * 60)
    print("PyTorch GPU 版本自动安装程序")
    print("=" * 60)
    print()
    
    # 创建下载目录
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # 下载所有wheel文件
    downloaded_files = []
    for name, url in WHEELS.items():
        filename = os.path.join(DOWNLOAD_DIR, url.split('/')[-1])
        
        # 如果文件已存在，跳过下载
        if os.path.exists(filename):
            print(f"✓ {filename} 已存在，跳过下载")
            downloaded_files.append(filename)
            continue
        
        if download_file(url, filename):
            downloaded_files.append(filename)
        else:
            print(f"\n⚠️  {name} 下载失败，尝试继续...")
    
    if len(downloaded_files) == 0:
        print("\n❌ 没有成功下载任何文件，无法安装")
        print("\n建议:")
        print("1. 检查网络连接")
        print("2. 尝试使用VPN/代理")
        print("3. 查看 MANUAL_INSTALL_PYTORCH.md 进行手动安装")
        return False
    
    print("\n" + "=" * 60)
    print("开始安装PyTorch...")
    print("=" * 60)
    
    # 安装下载的wheel文件
    for filename in downloaded_files:
        print(f"\n正在安装: {filename}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", filename])
            print(f"✅ {os.path.basename(filename)} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("验证安装...")
    print("=" * 60)
    
    # 验证安装
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            print("\n🎉 GPU版本安装成功！")
        else:
            print("\n⚠️  PyTorch已安装，但CUDA不可用")
            print("这可能是驱动问题，请检查NVIDIA驱动是否正确安装")
        return True
    except ImportError as e:
        print(f"❌ 验证失败: {e}")
        return False

if __name__ == "__main__":
    success = main()
    print("\n按任意键退出...")
    input()
    sys.exit(0 if success else 1)
