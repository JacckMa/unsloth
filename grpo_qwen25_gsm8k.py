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
MODEL_CACHE_PATH = os.path.join(SAVE_ROOT, "model_cache")  # 预训练模型缓存路径
LOCAL_MODEL_PATH = os.path.join(SAVE_ROOT, "models", "Qwen2.5-1.5B-Instruct")  # 本地模型路径

# 创建所有必要的目录
for path in [MODEL_SAVE_PATH, CHECKPOINT_PATH, DATA_CACHE_PATH, LOG_PATH, MODEL_CACHE_PATH]:
    os.makedirs(path, exist_ok=True)

# 检查本地模型是否存在
def check_local_model():
    """检查本地模型是否存在"""
    if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
        print(f"✅ 发现本地模型: {LOCAL_MODEL_PATH}")
        return True
    else:
        print(f"❌ 本地模型不存在: {LOCAL_MODEL_PATH}")
        print(f"💡 请先运行: python download_model.py")
        return False

# 设置HuggingFace缓存目录
os.environ["HF_HOME"] = MODEL_CACHE_PATH
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_PATH
os.environ["HF_HUB_CACHE"] = MODEL_CACHE_PATH

print(f"🚀 开始GRPO微调 Qwen2.5-1.5B-Instruct")
print(f"📁 模型保存路径: {MODEL_SAVE_PATH}")
print(f"💾 检查点保存路径: {CHECKPOINT_PATH}")
print(f"📊 日志保存路径: {LOG_PATH}")
print(f"🗂️ 数据缓存路径: {DATA_CACHE_PATH}")
print(f"🤗 预训练模型缓存路径: {MODEL_CACHE_PATH}")
print(f"🏠 本地模型路径: {LOCAL_MODEL_PATH}")

# 检查本地模型
if not check_local_model():
    print("\n⚠️ 请先下载模型:")
    print("   python download_model.py")
    print("\n或者修改代码使用在线下载模式")
    exit(1)

# ============================================================================
# 1. 模型和分词器加载
# ============================================================================

from unsloth import FastLanguageModel

max_seq_length = 2048
lora_rank = 32  # 适中的rank，平衡性能和效果

# 定义格式化标签（移到前面避免变量未定义错误）
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

# 添加调试变量（参照测试文件）
global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5

print(f"\n{'='*60}")
print("🔧 加载模型和分词器...")
print(f"{'='*60}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LOCAL_MODEL_PATH,  # 使用本地模型路径
    max_seq_length=max_seq_length,
    load_in_4bit=False,  # 参照测试文件，GRPO训练时不使用4bit
    fast_inference=True,  # 参照测试文件
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.7,  # 参照测试文件
    # 不需要cache_dir，因为直接从本地加载
)

print("✅ 模型加载完成！")

# 设置chat template（参照测试文件）
chat_template = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{{ '{system_prompt}' + eos_token }}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"
    "{% endif %}"
)

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

chat_template = chat_template.replace(
    "'{system_prompt}'", f"'{system_prompt}'"
).replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template

print("✅ Chat template配置完成！")

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
    lora_alpha=lora_rank * 2,  # 参照测试文件
    use_gradient_checkpointing="unsloth",
    random_state=42,  # 参照测试文件使用3407
)

print("✅ LoRA配置完成！")

# ============================================================================
# 3. 数据集准备
# ============================================================================

print(f"\n{'='*60}")
print("📚 准备GSM8K数据集...")
print(f"{'='*60}")

def extract_answer_from_gsm8k(text):
    """从GSM8K格式中提取最终答案"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def format_gsm8k_for_grpo(example):
    """将GSM8K格式化为GRPO训练格式"""
    system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
    
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example['question']},
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
        response = completion[0]["content"]  # 修复：正确访问completion内容
        
        # 检查是否包含所需的标签
        has_reasoning = reasoning_start in response and reasoning_end in response
        has_answer = solution_start in response and solution_end in response
        
        if has_reasoning and has_answer:
            scores.append(3.0)  # 修复：使用更高的分数
        else:
            scores.append(0.0)
    
    return scores

def format_checker_flexible(prompts, completions, ground_truth_answers, **kwargs):
    """灵活的格式检查"""
    scores = []
    for completion in completions:
        response = completion[0]["content"]  # 修复：正确访问completion内容
        
        score = 0
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        
        scores.append(score)
    
    return scores

def answer_correctness_checker(prompts, completions, answer, **kwargs):
    """检查答案正确性"""
    question = prompts[0][-1]["content"]  # 修复：正确访问prompt
    responses = [completion[0]["content"] for completion in completions]  # 修复：正确访问completion
    
    # 使用正则表达式匹配答案
    match_format = re.compile(
        rf"{reasoning_end}.*?"
        rf"{solution_start}(.+?){solution_end}"
        rf"[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL,
    )
    
    extracted_responses = [
        guess.group(1) if (guess := match_format.search(r)) is not None else None
        for r in responses
    ]
    
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        if guess == true_answer:
            score += 5.0
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1:
                    score += 2.0
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 1.5
                else:
                    score -= 2.5
            except:
                score -= 4.5
        scores.append(score)
    return scores

def reasoning_quality_checker(prompts, completions, answer, **kwargs):
    """检查推理质量 - 使用数字匹配（带调试输出）"""
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    
    # 匹配数字的正则表达式
    match_numbers = re.compile(
        solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})", flags=re.MULTILINE | re.DOTALL
    )
    
    extracted_responses = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in responses
    ]
    
    scores = []
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            "*" * 20 + f"Question:\n{question}",
            f"\nAnswer:\n{answer[0]}",
            f"\nResponse:\n{responses[0]}",
            f"\nExtracted:\n{extracted_responses[0]}",
        )
    PRINTED_TIMES += 1
    
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-2.5)
            continue
        try:
            true_answer = float(true_answer.strip())
            guess = float(guess.strip().replace(",", ""))
            scores.append(3.5 if guess == true_answer else -1.5)
        except:
            scores.append(0)
            continue
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

# 添加vllm采样参数（参照测试文件）
try:
    from vllm import SamplingParams
    VLLM_AVAILABLE = True
    print("✅ vLLM 已安装")
except ImportError:
    print("⚠️ vLLM 未安装，将使用默认采样参数")
    VLLM_AVAILABLE = False
    # 创建一个简单的替代类
    class SamplingParams:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

if VLLM_AVAILABLE:
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=42,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
else:
    vllm_sampling_params = None

try:
    from trl import GRPOConfig, GRPOTrainer
    TRL_AVAILABLE = True
    print("✅ TRL 已安装")
except ImportError:
    print("❌ TRL 未安装，请先安装: pip install trl")
    TRL_AVAILABLE = False
    raise ImportError("需要安装 TRL 库才能进行 GRPO 训练")

training_args = GRPOConfig(
    # vllm采样参数
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    
    # 基础训练参数
    learning_rate=5e-6,  # 参照测试文件的学习率
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",  # 参照测试文件
    optim="adamw_8bit",  # 8bit优化器节省显存
    
    # 批次和梯度参数
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # 参照测试文件
    
    # GRPO特定参数
    num_generations=4,  # 每次生成4个候选答案
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    
    # 训练步数和保存
    max_steps=100,  # 先减少步数测试
    save_steps=50,
    logging_steps=1,  # 参照测试文件
    
    # 输出和日志
    output_dir=CHECKPOINT_PATH,
    report_to="none",  # 参照测试文件，避免日志问题
    
    # 其他参数
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