#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本 - Qwen2.5-1.5B-Instruct
下载完成后可供训练脚本本地加载使用
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download
import warnings
warnings.filterwarnings("ignore")

# 设置下载路径
DOWNLOAD_ROOT = "root/autodl-tmp"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_MODEL_PATH = os.path.join(DOWNLOAD_ROOT, "models", "Qwen2.5-1.5B-Instruct")

print(f"🚀 开始下载 {MODEL_NAME}")
print(f"📁 下载路径: {LOCAL_MODEL_PATH}")
print(f"{'='*80}")

# 创建下载目录
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

def download_with_progress():
    """带进度显示的下载函数"""
    try:
        print("📥 开始下载模型文件...")
        start_time = time.time()
        
        # 使用 huggingface_hub 下载
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=LOCAL_MODEL_PATH,
            local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
            resume_download=True,          # 支持断点续传
            cache_dir=None,               # 不使用缓存，直接下载到目标目录
        )
        
        end_time = time.time()
        download_time = end_time - start_time
        
        print(f"✅ 模型下载完成！")
        print(f"⏱️ 下载耗时: {download_time:.2f} 秒")
        
        # 检查下载的文件
        model_files = list(Path(LOCAL_MODEL_PATH).rglob("*"))
        print(f"📊 共下载 {len(model_files)} 个文件")
        
        # 显示主要文件
        important_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "model.safetensors"
        ]
        
        print(f"\n📋 主要模型文件:")
        for file in important_files:
            file_path = Path(LOCAL_MODEL_PATH) / file
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  ✅ {file} ({size_mb:.1f} MB)")
            else:
                print(f"  ❓ {file} (未找到)")
        
        # 查找 .safetensors 文件
        safetensor_files = list(Path(LOCAL_MODEL_PATH).glob("*.safetensors"))
        if safetensor_files:
            print(f"\n🔧 模型权重文件:")
            for file in safetensor_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  ✅ {file.name} ({size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def verify_model():
    """验证模型文件完整性"""
    print(f"\n🔍 验证模型文件...")
    
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    all_good = True
    for file in required_files:
        file_path = Path(LOCAL_MODEL_PATH) / file
        if file_path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} 缺失")
            all_good = False
    
    # 检查是否有模型权重文件
    weight_files = (
        list(Path(LOCAL_MODEL_PATH).glob("*.safetensors")) + 
        list(Path(LOCAL_MODEL_PATH).glob("*.bin"))
    )
    
    if weight_files:
        print(f"  ✅ 模型权重文件 ({len(weight_files)} 个)")
    else:
        print(f"  ❌ 未找到模型权重文件")
        all_good = False
    
    return all_good

def create_load_script():
    """创建加载脚本示例"""
    load_script = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地模型加载示例
"""

from unsloth import FastLanguageModel

# 从本地路径加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{LOCAL_MODEL_PATH}",  # 使用本地路径
    max_seq_length=2048,
    load_in_4bit=False,
    dtype=None,
)

print("✅ 模型从本地加载成功！")
print(f"📁 模型路径: {LOCAL_MODEL_PATH}")
print(f"🏷️ 模型名称: {{model.config.name_or_path}}")
'''
    
    script_path = os.path.join(DOWNLOAD_ROOT, "load_local_model.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(load_script)
    
    print(f"📝 已创建加载脚本: {script_path}")

if __name__ == "__main__":
    print("🔽 开始下载模型...")
    
    # 下载模型
    success = download_with_progress()
    
    if success:
        # 验证文件
        if verify_model():
            print(f"\n🎉 模型下载和验证完成！")
            
            # 创建加载脚本
            create_load_script()
            
            print(f"\n📋 使用方法:")
            print(f"1. 训练脚本中使用: model_name='{LOCAL_MODEL_PATH}'")
            print(f"2. 测试加载: python {DOWNLOAD_ROOT}/load_local_model.py")
            print(f"3. 模型位置: {LOCAL_MODEL_PATH}")
            
        else:
            print(f"\n⚠️ 模型文件验证失败，请检查下载完整性")
            sys.exit(1)
    else:
        print(f"\n❌ 模型下载失败")
        sys.exit(1)
    
    print(f"\n✨ 下载脚本执行完毕！") 