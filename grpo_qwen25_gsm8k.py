#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO LoRA微调 Qwen2.5-1.5B-Instruct 模型
数据集: GSM8K
保存路径: /root/autodl-tmp
"""

import os
import re
import json
import torch
import gc
from pathlib import Path
from datasets import load_dataset
from transformers import TrainingArguments
import warnings
warnings.filterwarnings("ignore")

# 设置保存根目录
SAVE_ROOT = "/root/autodl-tmp"
os.makedirs(SAVE_ROOT, exist_ok=True)

# 设置各种保存路径
MODEL_SAVE_PATH = os.path.join(SAVE_ROOT, "qwen25_grpo_model")
CHECKPOINT_PATH = os.path.join(SAVE_ROOT, "qwen25_grpo_checkpoint") 
DATA_CACHE_PATH = os.path.join(SAVE_ROOT, "gsm8k_cache")
LOG_PATH = os.path.join(SAVE_ROOT, "training_logs")

# 创建所有必要的目录
for path in [MODEL_SAVE_PATH, CHECKPOINT_PATH, DATA_CACHE_PATH, LOG_PATH]:
    os.makedirs(path, exist_ok=True)

print(f"🚀 开始GRPO微调 Qwen2.5-1.5B-Instruct")
print(f"📁 模型保存路径: {MODEL_SAVE_PATH}")
print(f"💾 检查点保存路径: {CHECKPOINT_PATH}")
print(f"📊 日志保存路径: {LOG_PATH}")
print(f"🗂️ 数据缓存路径: {DATA_CACHE_PATH}")

# ============================================================================
# 1. 模型和分词器加载
# ============================================================================

from unsloth import FastLanguageModel

max_seq_length = 2048
lora_rank = 32  # 适中的rank，平衡性能和效果

print(f"\n{'='*60}")
print("🔧 加载模型和分词器...")
print(f"{'='*60}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # 4bit量化节省显存
    fast_inference=False,  # GRPO训练时关闭
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.8,
)

print("✅ 模型加载完成！")

# ============================================================================
# 2. LoRA配置
# ============================================================================

print(f"\n{'='*60}")
print("🎯 配置LoRA...")
print(f"{'='*60}")

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
    max_seq_length=max_seq_length,
    use_rslora=False,
)

print("✅ LoRA配置完成！")

# ============================================================================
# 3. 数据集准备
# ============================================================================

print(f"\n{'='*60}")
print("📚 准备GSM8K数据集...")
print(f"{'='*60}")

# 定义格式化标签
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

def extract_answer_from_gsm8k(text):
    """从GSM8K格式中提取最终答案"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def format_gsm8k_for_grpo(example):
    """将GSM8K格式化为GRPO训练格式"""
    system_prompt = (
        f"You are a helpful math tutor. When solving problems, think step by step. "
        f"Place your reasoning between {reasoning_start} and {reasoning_end}. "
        f"Then provide your final numerical answer between {solution_start} and {solution_end}."
    )
    
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Problem: {example['question']}\n\nSolve this step by step."},
        ],
        "answer": extract_answer_from_gsm8k(example["answer"]),
        "full_solution": example["answer"]  # 保留完整解答用于参考
    }

# 加载GSM8K数据集
print("📥 下载GSM8K数据集...")
try:
    dataset = load_dataset("gsm8k", "main", split="train", cache_dir=DATA_CACHE_PATH)
    print(f"✅ 成功加载 {len(dataset)} 条训练数据")
except Exception as e:
    print(f"❌ 数据集加载失败: {e}")
    print("🔄 尝试使用备用方法...")
    dataset = load_dataset("openai/gsm8k", "main", split="train", cache_dir=DATA_CACHE_PATH)
    print(f"✅ 成功加载 {len(dataset)} 条训练数据")

# 格式化数据集
print("🔄 格式化数据集...")
formatted_dataset = dataset.map(format_gsm8k_for_grpo, num_proc=4)

# 只使用前1000条数据进行快速训练（可根据需要调整）
train_size = min(1000, len(formatted_dataset))
gsm8k_train = formatted_dataset.select(range(train_size))

print(f"✅ 数据集准备完成！使用 {len(gsm8k_train)} 条数据进行训练")

# ============================================================================
# 4. 奖励函数定义
# ============================================================================

print(f"\n{'='*60}")
print("🎯 定义奖励函数...")
print(f"{'='*60}")

def format_checker_exact(prompts, completions, ground_truth_answers, **kwargs):
    """检查是否严格按照格式输出"""
    scores = []
    for completion in completions:
        completion_text = completion.strip()
        
        # 检查是否包含所需的标签
        has_reasoning = reasoning_start in completion_text and reasoning_end in completion_text
        has_answer = solution_start in completion_text and solution_end in completion_text
        
        if has_reasoning and has_answer:
            scores.append(1.0)
        else:
            scores.append(0.0)
    
    return scores

def format_checker_flexible(prompts, completions, ground_truth_answers, **kwargs):
    """灵活的格式检查"""
    scores = []
    for completion in completions:
        completion_text = completion.strip().lower()
        
        # 检查是否包含推理相关关键词
        reasoning_keywords = ["reasoning", "think", "step", "because", "therefore", "since"]
        answer_keywords = ["answer", "result", "solution", "final"]
        
        has_reasoning_content = any(keyword in completion_text for keyword in reasoning_keywords)
        has_answer_content = any(keyword in completion_text for keyword in answer_keywords)
        
        if has_reasoning_content and has_answer_content:
            scores.append(0.8)
        elif has_reasoning_content or has_answer_content:
            scores.append(0.4)
        else:
            scores.append(0.1)
    
    return scores

def answer_correctness_checker(prompts, completions, ground_truth_answers, **kwargs):
    """检查答案正确性"""
    scores = []
    
    for completion, gt_answer in zip(completions, ground_truth_answers):
        if gt_answer is None:
            scores.append(0.0)
            continue
            
        completion_text = completion.strip()
        
        # 尝试从completion中提取数字答案
        # 优先从<answer>标签中提取
        answer_match = re.search(rf'{re.escape(solution_start)}(.*?){re.escape(solution_end)}', 
                                completion_text, re.DOTALL)
        if answer_match:
            predicted_text = answer_match.group(1).strip()
        else:
            # 如果没有标签，从整个文本中提取最后的数字
            predicted_text = completion_text
        
        # 提取数字
        predicted_numbers = re.findall(r'-?\d+(?:\.\d+)?', predicted_text)
        gt_numbers = re.findall(r'-?\d+(?:\.\d+)?', str(gt_answer))
        
        if predicted_numbers and gt_numbers:
            try:
                predicted_num = float(predicted_numbers[-1])  # 取最后一个数字
                gt_num = float(gt_numbers[-1])
                
                if abs(predicted_num - gt_num) < 1e-6:  # 数值相等
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            except ValueError:
                scores.append(0.0)
        else:
            scores.append(0.0)
    
    return scores

def reasoning_quality_checker(prompts, completions, ground_truth_answers, **kwargs):
    """检查推理质量"""
    scores = []
    
    for completion in completions:
        completion_text = completion.strip()
        
        # 基础分数
        score = 0.3
        
        # 检查推理长度（更长的推理通常更详细）
        if len(completion_text) > 100:
            score += 0.2
        
        # 检查是否包含数学相关词汇
        math_keywords = ["multiply", "divide", "add", "subtract", "calculate", 
                        "equation", "solve", "total", "sum", "difference"]
        math_count = sum(1 for keyword in math_keywords if keyword in completion_text.lower())
        score += min(math_count * 0.1, 0.3)
        
        # 检查逻辑连接词
        logic_keywords = ["therefore", "so", "thus", "hence", "because", "since", 
                         "first", "then", "next", "finally"]
        logic_count = sum(1 for keyword in logic_keywords if keyword in completion_text.lower())
        score += min(logic_count * 0.05, 0.2)
        
        scores.append(min(score, 1.0))
    
    return scores

print("✅ 奖励函数定义完成！")

# ============================================================================
# 5. GRPO训练配置
# ============================================================================

print(f"\n{'='*60}")
print("⚙️ 配置GRPO训练参数...")
print(f"{'='*60}")

# 计算最大提示长度
sample_prompts = [tokenizer.apply_chat_template(gsm8k_train[i]["prompt"], 
                                               add_generation_prompt=True, 
                                               tokenize=True) 
                 for i in range(min(100, len(gsm8k_train)))]
prompt_lengths = [len(prompt) for prompt in sample_prompts]
max_prompt_length = max(prompt_lengths) + 50  # 添加一些缓冲
max_completion_length = max_seq_length - max_prompt_length

print(f"📏 最大提示长度: {max_prompt_length}")
print(f"📏 最大完成长度: {max_completion_length}")

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    # 基础训练参数
    learning_rate=3e-6,  # 较小的学习率确保稳定训练
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",  # 8bit优化器节省显存
    
    # 批次和梯度参数
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # 增加梯度累积步数
    max_grad_norm=1.0,
    
    # GRPO特定参数
    num_generations=4,  # 每次生成4个候选答案
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    
    # 训练步数和保存
    max_steps=500,  # 适中的训练步数
    save_steps=100,
    logging_steps=10,
    eval_steps=100,
    
    # 输出和日志
    output_dir=CHECKPOINT_PATH,
    logging_dir=LOG_PATH,
    report_to="tensorboard",  # 使用tensorboard记录日志
    
    # 其他参数
    dataloader_num_workers=4,
    remove_unused_columns=False,
    seed=42,
)

print("✅ 训练参数配置完成！")

# ============================================================================
# 6. 创建训练器并开始训练
# ============================================================================

print(f"\n{'='*60}")
print("🚂 创建GRPO训练器...")
print(f"{'='*60}")

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_checker_exact,      # 严格格式检查 (权重高)
        format_checker_flexible,   # 灵活格式检查
        answer_correctness_checker, # 答案正确性检查 (权重最高)
        reasoning_quality_checker,  # 推理质量检查
    ],
    args=training_args,
    train_dataset=gsm8k_train,
)

print("✅ 训练器创建完成！")

# ============================================================================
# 7. 开始训练
# ============================================================================

print(f"\n{'='*60}")
print(f"🚀 开始GRPO训练！")
print(f"📊 训练数据量: {len(gsm8k_train)}")
print(f"🔄 最大训练步数: {training_args.max_steps}")
print(f"💾 检查点保存间隔: {training_args.save_steps} 步")
print(f"{'='*60}")

try:
    trainer.train()
    print("✅ 训练完成！")
except Exception as e:
    print(f"❌ 训练过程中出现错误: {e}")
    print("💾 尝试保存当前状态...")

# ============================================================================
# 8. 保存最终模型
# ============================================================================

print(f"\n{'='*60}")
print("💾 保存最终模型...")
print(f"{'='*60}")

try:
    # 保存LoRA适配器
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"✅ LoRA模型已保存到: {MODEL_SAVE_PATH}")
    
    # 保存训练日志总结
    log_summary = {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "dataset": "GSM8K",
        "training_method": "GRPO + LoRA",
        "lora_rank": lora_rank,
        "max_seq_length": max_seq_length,
        "training_samples": len(gsm8k_train),
        "max_steps": training_args.max_steps,
        "learning_rate": training_args.learning_rate,
        "save_paths": {
            "model": MODEL_SAVE_PATH,
            "checkpoint": CHECKPOINT_PATH,
            "logs": LOG_PATH,
            "data_cache": DATA_CACHE_PATH
        }
    }
    
    with open(os.path.join(SAVE_ROOT, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(log_summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 训练总结已保存到: {os.path.join(SAVE_ROOT, 'training_summary.json')}")
    
except Exception as e:
    print(f"❌ 模型保存失败: {e}")

# ============================================================================
# 9. 清理内存
# ============================================================================

print(f"\n{'='*60}")
print("🧹 清理内存...")
print(f"{'='*60}")

del trainer
del model
del tokenizer
torch.cuda.empty_cache()
gc.collect()

print("✅ 内存清理完成！")

# ============================================================================
# 10. 训练完成总结
# ============================================================================

print(f"\n{'='*80}")
print("🎉 GRPO微调完成！")
print(f"{'='*80}")
print(f"📁 所有文件都保存在: {SAVE_ROOT}")
print(f"🏷️ 模型保存路径: {MODEL_SAVE_PATH}")
print(f"💾 检查点路径: {CHECKPOINT_PATH}")
print(f"📊 训练日志路径: {LOG_PATH}")
print(f"🗂️ 数据缓存路径: {DATA_CACHE_PATH}")
print(f"{'='*80}")

print("\n🔥 下一步操作建议:")
print("1. 查看训练日志: tensorboard --logdir /root/autodl-tmp/training_logs")
print("2. 加载模型测试: FastLanguageModel.from_pretrained('/root/autodl-tmp/qwen25_grpo_model')")
print("3. 检查训练总结: cat /root/autodl-tmp/training_summary.json")

print("\n✨ 训练脚本执行完毕！") 