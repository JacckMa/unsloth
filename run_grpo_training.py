#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的GRPO训练启动脚本
"""

import subprocess
import sys
import os

def main():
    print("🚀 启动 Qwen2.5-1.5B GRPO微调...")
    
    # 检查CUDA是否可用
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ 检测到GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️ 未检测到GPU，将使用CPU训练（速度较慢）")
    except ImportError:
        print("❌ PyTorch未安装")
        return
    
    # 检查Unsloth是否安装
    try:
        import unsloth
        print("✅ Unsloth已安装")
    except ImportError:
        print("❌ Unsloth未安装，请先运行: pip install unsloth")
        return
    
    # 运行训练脚本
    script_name = "grpo_qwen25_gsm8k.py"
    if os.path.exists(script_name):
        print(f"▶️ 执行训练脚本: {script_name}")
        subprocess.run([sys.executable, script_name])
    else:
        print(f"❌ 找不到训练脚本: {script_name}")

if __name__ == "__main__":
    main() 