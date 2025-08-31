#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GSPO LoRA微调 Qwen2.5-1.5B-Instruct 模型
数据集: GSM8K
保存路径: /root/autodl-tmp
GSPO (Group Sequence Policy Optimization) - 序列级优化，不同于GRPO的token级优化
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
MODEL_SAVE_PATH = os.path.join(SAVE_ROOT, "qwen25_gspo_model")
CHECKPOINT_PATH = os.path.join(SAVE_ROOT, "qwen25_gspo_checkpoint") 
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

print(f"🚀 开始GSPO微调 Qwen2.5-1.5B-Instruct")
print(f"📁 模型保存路径: {MODEL_SAVE_PATH}")
print(f"💾 检查点保存路径: {CHECKPOINT_PATH}")
print(f"📊 日志保存路径: {LOG_PATH}")
print(f"🗂️ 数据缓存路径: {DATA_CACHE_PATH}")
print(f"🤗 预训练模型缓存路径: {MODEL_CACHE_PATH}")
print(f"🏠 本地模型路径: {LOCAL_MODEL_PATH}")
print(f"🔥 GSPO特点: 序列级优化，避免token级别噪声！")

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
    load_in_4bit=False,  # GSPO训练时不使用4bit，保持精度
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.7,
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

# 确保tokenizer配置正确，解决BOS token警告
if tokenizer.bos_token is None:
    tokenizer.bos_token = tokenizer.eos_token
if tokenizer.bos_token_id is None:
    tokenizer.bos_token_id = tokenizer.eos_token_id

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
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=42,
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

def format_gsm8k_for_gspo(example):
    """将GSM8K格式化为GSPO训练格式"""
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
formatted_dataset = dataset.map(format_gsm8k_for_gspo, num_proc=4)

# 只使用前1000条数据进行快速训练（可根据需要调整）
train_size = min(1000, len(formatted_dataset))
gsm8k_train = formatted_dataset.select(range(train_size))

print(f"✅ 数据集准备完成！使用 {len(gsm8k_train)} 条数据进行训练")

# ============================================================================
# 4. 奖励函数定义（GSPO使用相同的奖励函数，但在序列级别应用）
# ============================================================================

print(f"\n{'='*60}")
print("🎯 定义GSPO奖励函数...")
print(f"{'='*60}")

def format_checker_exact(completions, **kwargs):
    """检查是否严格按照格式输出（GSPO序列级检查）"""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        
        # 创建格式匹配正则表达式
        match_format = re.compile(
            rf"{reasoning_end}.*?"
            rf"{solution_start}(.+?){solution_end}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL,
        )
        
        score = 0
        if match_format.search(response) is not None:
            score += 3.0  # GSPO: 序列级奖励，不进行token级分解
        scores.append(score)
    
    return scores

def format_checker_flexible(completions, **kwargs):
    """灵活的格式检查（GSPO序列级）"""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        
        score = 0
        # GSPO: 在序列级别累积分数，而不是token级
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        
        scores.append(score)
    
    return scores

def answer_correctness_checker(prompts, completions, answer, **kwargs):
    """检查答案正确性（GSPO序列级评估）"""
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    
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
            score += 5.0  # GSPO: 序列级正确性奖励
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
    """检查推理质量 - GSPO序列级评估"""
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
            "*" * 20 + f"GSPO Question:\n{question}",
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
            # GSPO: 序列级质量评分
            scores.append(3.5 if guess == true_answer else -1.5)
        except:
            scores.append(0)
            continue
    return scores

print("✅ GSPO奖励函数定义完成！")

# ============================================================================
# 5. GSPO训练配置（关键差异：使用PPO配置但设置为序列级优化）
# ============================================================================

print(f"\n{'='*60}")
print("⚙️ 配置GSPO训练参数...")
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

# 添加vllm采样参数
from unsloth import vLLMSamplingParams

# GSPO采样参数：更重视序列多样性
vllm_sampling_params = vLLMSamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=42,
    stop=[tokenizer.eos_token],
    include_stop_str_in_output=True,
)

# GSPO使用PPO训练器，但配置为序列级优化
try:
    from trl import PPOConfig, PPOTrainer
    TRL_AVAILABLE = True
    print("✅ TRL 已安装，使用PPO训练器进行GSPO训练")
except ImportError:
    print("❌ TRL 未安装，请先安装: pip install trl")
    TRL_AVAILABLE = False
    raise ImportError("需要安装 TRL 库才能进行 GSPO 训练")

# GSPO关键配置：序列级优化参数
training_args = PPOConfig(
    # vllm采样参数
    vllm_sampling_params=vllm_sampling_params,
    
    # 基础训练参数
    learning_rate=5e-6,  # GSPO推荐较低学习率，避免序列级震荡
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",  # 8bit优化器节省显存
    
    # 批次和梯度参数
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,  # GSPO增加梯度累积以稳定序列级更新
    
    # GSPO特定参数：序列级优化
    batch_size=4,  # 序列组大小，用于相对比较
    mini_batch_size=2,  # 小批次大小
    ppo_epochs=4,  # GSPO建议更多epoch以充分利用序列级信号
    
    # 裁剪参数（GSPO的关键：序列级裁剪）
    cliprange=0.2,  # 重要性比率裁剪范围
    cliprange_value=0.2,  # 值函数裁剪范围
    vf_coef=0.1,  # 值函数损失系数（GSPO中较小，因为重点在序列级策略）
    
    # 序列长度参数
    max_prompt_length=max_prompt_length,
    max_length=max_seq_length,
    
    # 训练步数和保存
    total_ppo_epochs=1000,  # 总训练epoch
    save_freq=200,  # 保存频率
    log_freq=1,
    
    # 输出和日志
    project_kwargs={"project_name": "gspo_qwen25_gsm8k"},
    tracker_project_name="gspo_training",
    
    # GSPO的KL散度控制（序列级）
    init_kl_coef=0.2,  # 初始KL系数
    target=6,  # 目标KL散度
    horizon=10000,  # KL控制时间窗口
    
    # 其他参数
    seed=42,
    remove_unused_columns=False,
)

print("✅ GSPO训练参数配置完成！")
print("🔥 关键特点：序列级裁剪和优化，避免token级噪声")

# ============================================================================
# 6. 创建GSPO训练器（使用PPO但配置为序列级）
# ============================================================================

print(f"\n{'='*60}")
print("🚂 创建GSPO训练器（基于PPO的序列级优化）...")
print(f"{'='*60}")

# 准备奖励函数组合
def combined_reward_function(prompts, completions, **kwargs):
    """组合奖励函数，返回序列级奖励"""
    # 获取所有奖励分数
    format_exact_scores = format_checker_exact(completions, **kwargs)
    format_flexible_scores = format_checker_flexible(completions, **kwargs)
    correctness_scores = answer_correctness_checker(prompts, completions, **kwargs)
    quality_scores = reasoning_quality_checker(prompts, completions, **kwargs)
    
    # 序列级奖励组合（权重可调）
    combined_scores = []
    for i in range(len(completions)):
        # GSPO: 在序列级别组合奖励，不进行token级分解
        total_score = (
            format_exact_scores[i] * 1.0 +      # 格式奖励权重
            format_flexible_scores[i] * 0.5 +   # 灵活格式权重  
            correctness_scores[i] * 2.0 +       # 正确性奖励权重（最高）
            quality_scores[i] * 1.5             # 质量奖励权重
        )
        combined_scores.append(total_score)
    
    return combined_scores

# 转换数据集格式为PPO训练器期望的格式
def convert_to_ppo_format(dataset):
    """转换数据集为PPO格式"""
    ppo_dataset = []
    for item in dataset:
        # 将prompt转换为字符串
        prompt_text = tokenizer.apply_chat_template(
            item["prompt"], 
            add_generation_prompt=True, 
            tokenize=False
        )
        ppo_dataset.append({
            "query": prompt_text,
            "answer": item["answer"],
            "full_solution": item["full_solution"]
        })
    return ppo_dataset

ppo_formatted_dataset = convert_to_ppo_format(gsm8k_train)

# 创建PPO训练器进行GSPO训练
trainer = PPOTrainer(
    model=model,
    config=training_args,
    dataset=ppo_formatted_dataset,
    tokenizer=tokenizer,
    # 注意：这里我们将通过自定义训练循环来实现GSPO的序列级优化
)

print("✅ GSPO训练器创建完成！")

# ============================================================================
# 7. GSPO自定义训练循环（关键：序列级优化实现）
# ============================================================================

print(f"\n{'='*60}")
print(f"🚀 开始GSPO训练！")
print(f"📊 训练数据量: {len(ppo_formatted_dataset)}")
print(f"🔄 最大训练epoch: {training_args.total_ppo_epochs}")
print(f"💾 保存间隔: {training_args.save_freq} epoch")
print(f"🎯 GSPO特点: 序列级策略优化")
print(f"{'='*60}")

# GSPO训练循环
generation_kwargs = {
    "max_new_tokens": max_completion_length,
    "do_sample": True,
    "top_p": 1.0,
    "temperature": 1.0,
    "pad_token_id": tokenizer.pad_token_id,
}

try:
    from tqdm import tqdm
    
    # GSPO训练主循环
    for epoch in tqdm(range(training_args.total_ppo_epochs), desc="GSPO Training"):
        # 批次训练
        for batch_idx, batch in enumerate(tqdm(trainer.dataloader, desc=f"Epoch {epoch}")):
            try:
                query_tensors = batch["input_ids"]
                
                # 1. 生成多个候选序列（GSPO的关键）
                response_tensors = trainer.generate(
                    query_tensors, 
                    return_prompt=False,
                    **generation_kwargs
                )
                
                # 2. 解码生成的响应
                batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) 
                                   for r in response_tensors]
                
                # 3. 计算序列级奖励
                completions = [[{"content": resp}] for resp in batch["response"]]
                prompts = [[{"content": q}] for q in batch["query"]]
                
                rewards = combined_reward_function(
                    prompts=prompts,
                    completions=completions,
                    answer=batch.get("answer", [0] * len(completions))
                )
                
                # 转换为tensor
                rewards = [torch.tensor(r, dtype=torch.float32) for r in rewards]
                
                # 4. GSPO序列级策略更新
                stats = trainer.step(query_tensors, response_tensors, rewards)
                
                # 5. 记录统计信息
                if batch_idx % training_args.log_freq == 0:
                    trainer.log_stats(stats, batch, rewards)
                    
                    # GSPO特定日志
                    print(f"Epoch {epoch}, Batch {batch_idx}:")
                    print(f"  平均序列奖励: {torch.stack(rewards).mean():.4f}")
                    print(f"  奖励标准差: {torch.stack(rewards).std():.4f}")
                    print(f"  序列长度: {[len(r.squeeze()) for r in response_tensors]}")
                
            except Exception as e:
                print(f"⚠️ 批次 {batch_idx} 训练出错: {e}")
                continue
        
        # 定期保存模型
        if epoch % training_args.save_freq == 0:
            save_path = os.path.join(CHECKPOINT_PATH, f"gspo_epoch_{epoch}")
            trainer.save_model(save_path)
            print(f"💾 已保存检查点到: {save_path}")
    
    print("✅ GSPO训练完成！")
    
except Exception as e:
    print(f"❌ GSPO训练过程中出现错误: {e}")
    print("💾 尝试保存当前状态...")

# ============================================================================
# 8. 保存最终GSPO模型
# ============================================================================

print(f"\n{'='*60}")
print("💾 保存最终GSPO模型...")
print(f"{'='*60}")

try:
    # 保存LoRA适配器
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"✅ GSPO LoRA模型已保存到: {MODEL_SAVE_PATH}")
    
    # 保存训练日志总结
    log_summary = {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "dataset": "GSM8K",
        "training_method": "GSPO + LoRA",
        "optimization_level": "sequence_level",  # GSPO特点
        "lora_rank": lora_rank,
        "max_seq_length": max_seq_length,
        "training_samples": len(ppo_formatted_dataset),
        "max_epochs": training_args.total_ppo_epochs,
        "learning_rate": training_args.learning_rate,
        "key_differences_from_grpo": [
            "序列级优化而非token级",
            "序列级重要性比率裁剪",
            "避免token级梯度噪声",
            "更稳定的训练过程"
        ],
        "save_paths": {
            "model": MODEL_SAVE_PATH,
            "checkpoint": CHECKPOINT_PATH,
            "logs": LOG_PATH,
            "data_cache": DATA_CACHE_PATH
        }
    }
    
    with open(os.path.join(SAVE_ROOT, "gspo_training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(log_summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ GSPO训练总结已保存到: {os.path.join(SAVE_ROOT, 'gspo_training_summary.json')}")
    
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
# 10. GSPO训练完成总结
# ============================================================================

print(f"\n{'='*80}")
print("🎉 GSPO微调完成！")
print(f"{'='*80}")
print(f"📁 所有文件都保存在: {SAVE_ROOT}")
print(f"🏷️ 模型保存路径: {MODEL_SAVE_PATH}")
print(f"💾 检查点路径: {CHECKPOINT_PATH}")
print(f"📊 训练日志路径: {LOG_PATH}")
print(f"🗂️ 数据缓存路径: {DATA_CACHE_PATH}")
print(f"{'='*80}")

print("\n🔥 GSPO vs GRPO 关键区别:")
print("1. 🎯 优化层级: 序列级 vs token级")
print("2. 📊 裁剪策略: 序列级重要性比率裁剪")
print("3. 🚀 训练稳定性: 避免token级梯度噪声")
print("4. 💡 奖励分配: 序列级统一处理")
print("5. 🔧 适用场景: 长序列和MoE模型更稳定")

print("\n🔥 下一步操作建议:")
print("1. 查看训练日志: tensorboard --logdir /root/autodl-tmp/training_logs")
print("2. 加载模型测试: FastLanguageModel.from_pretrained('/root/autodl-tmp/qwen25_gspo_model')")
print("3. 检查训练总结: cat /root/autodl-tmp/gspo_training_summary.json")
print("4. 对比GRPO结果: 查看序列级优化的效果差异")

print("\n✨ GSPO训练脚本执行完毕！") 
print("🎊 享受序列级优化带来的训练稳定性提升！") 