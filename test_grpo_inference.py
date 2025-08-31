#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO训练后的模型推理测试
加载微调后的 Qwen2.5-1.5B-Instruct 模型进行数学推理
"""

import os
import re
import json
import torch
from pathlib import Path
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

# 设置路径（与训练脚本保持一致）
SAVE_ROOT = "/root/autodl-tmp"
MODEL_SAVE_PATH = os.path.join(SAVE_ROOT, "qwen25_grpo_model")
DATA_CACHE_PATH = os.path.join(SAVE_ROOT, "gsm8k_cache")

# 格式化标签（与训练时一致）
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

print(f"🚀 GRPO微调模型推理测试")
print(f"📁 模型路径: {MODEL_SAVE_PATH}")
print("="*60)

# ============================================================================
# 1. 检查模型是否存在
# ============================================================================

def check_model_exists():
    """检查训练好的模型是否存在"""
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(os.path.join(MODEL_SAVE_PATH, "adapter_config.json")):
        print(f"✅ 发现训练好的模型: {MODEL_SAVE_PATH}")
        return True
    else:
        print(f"❌ 模型不存在: {MODEL_SAVE_PATH}")
        print("💡 请先运行训练脚本: python grpo_qwen25_gsm8k.py")
        return False

if not check_model_exists():
    exit(1)

# ============================================================================
# 2. 加载模型和分词器
# ============================================================================

print("\n🔧 加载训练好的模型...")

from unsloth import FastLanguageModel

# 加载微调后的模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_SAVE_PATH,  # 加载微调后的模型
    max_seq_length=2048,
    load_in_4bit=True,  # 推理时可以使用4bit节省显存
    fast_inference=True,
)

# 设置chat template（与训练时一致）
system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

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

chat_template = chat_template.replace(
    "'{system_prompt}'", f"'{system_prompt}'"
).replace("'{reasoning_start}'", f"'{reasoning_start}'")
tokenizer.chat_template = chat_template

print("✅ 模型加载完成！")

# ============================================================================
# 3. 准备测试数据
# ============================================================================

print("\n📚 准备测试数据...")

# 加载GSM8K测试集
try:
    test_dataset = load_dataset("gsm8k", "main", split="test", cache_dir=DATA_CACHE_PATH)
    print(f"✅ 成功加载 {len(test_dataset)} 条测试数据")
except Exception as e:
    print(f"❌ 测试集加载失败，使用备用方法: {e}")
    test_dataset = load_dataset("openai/gsm8k", "main", split="test", cache_dir=DATA_CACHE_PATH)
    print(f"✅ 成功加载 {len(test_dataset)} 条测试数据")

# ============================================================================
# 4. 推理函数定义
# ============================================================================

def extract_answer_from_gsm8k(text):
    """从GSM8K格式中提取最终答案"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def extract_solution_from_response(response):
    """从模型回答中提取解决方案"""
    # 使用正则表达式匹配解决方案
    match = re.search(rf"{solution_start}(.+?){solution_end}", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_reasoning_from_response(response):
    """从模型回答中提取推理过程"""
    # 匹配推理过程
    match = re.search(rf"{reasoning_start}(.*?){reasoning_end}", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def run_inference(question, max_new_tokens=512):
    """运行单个问题的推理"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # 应用chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    # 生成回答
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # 解码回答
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

# ============================================================================
# 5. 批量测试
# ============================================================================

def evaluate_model(num_samples=20):
    """评估模型性能"""
    print(f"\n🧪 开始评估模型 (使用前{num_samples}个样本)...")
    
    correct_count = 0
    format_correct_count = 0
    total_count = min(num_samples, len(test_dataset))
    
    results = []
    
    for i in range(total_count):
        example = test_dataset[i]
        question = example['question']
        true_answer = extract_answer_from_gsm8k(example['answer'])
        
        print(f"\n📝 测试样本 {i+1}/{total_count}")
        print(f"问题: {question}")
        print(f"正确答案: {true_answer}")
        
        try:
            # 运行推理
            response = run_inference(question)
            
            # 提取推理过程和答案
            reasoning = extract_reasoning_from_response(response)
            predicted_answer = extract_solution_from_response(response)
            
            print(f"模型回答: {response}")
            print(f"提取的推理: {reasoning}")
            print(f"提取的答案: {predicted_answer}")
            
            # 检查格式是否正确
            format_ok = (reasoning is not None and predicted_answer is not None)
            if format_ok:
                format_correct_count += 1
                print("✅ 格式正确")
            else:
                print("❌ 格式错误")
            
            # 检查答案是否正确
            answer_correct = False
            if predicted_answer and true_answer:
                # 尝试数值比较
                try:
                    pred_num = float(predicted_answer.replace(',', '').strip())
                    true_num = float(true_answer.replace(',', '').strip())
                    answer_correct = abs(pred_num - true_num) < 0.001
                except:
                    # 字符串比较
                    answer_correct = predicted_answer.strip() == true_answer.strip()
            
            if answer_correct:
                correct_count += 1
                print("✅ 答案正确")
            else:
                print("❌ 答案错误")
            
            # 保存结果
            results.append({
                "question": question,
                "true_answer": true_answer,
                "response": response,
                "reasoning": reasoning,
                "predicted_answer": predicted_answer,
                "format_correct": format_ok,
                "answer_correct": answer_correct
            })
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            results.append({
                "question": question,
                "true_answer": true_answer,
                "error": str(e)
            })
        
        print("-" * 50)
    
    # 计算评估指标
    format_accuracy = format_correct_count / total_count * 100
    answer_accuracy = correct_count / total_count * 100
    
    print(f"\n📊 评估结果:")
    print(f"{'='*50}")
    print(f"测试样本数: {total_count}")
    print(f"格式正确率: {format_correct_count}/{total_count} ({format_accuracy:.1f}%)")
    print(f"答案正确率: {correct_count}/{total_count} ({answer_accuracy:.1f}%)")
    print(f"{'='*50}")
    
    # 保存详细结果
    results_file = os.path.join(SAVE_ROOT, "inference_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_samples": total_count,
                "format_correct": format_correct_count,
                "answer_correct": correct_count,
                "format_accuracy": format_accuracy,
                "answer_accuracy": answer_accuracy
            },
            "details": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"📄 详细结果已保存到: {results_file}")
    
    return format_accuracy, answer_accuracy

# ============================================================================
# 6. 交互式测试
# ============================================================================

def interactive_test():
    """交互式测试模式"""
    print(f"\n🎯 交互式测试模式")
    print("输入数学问题，模型将生成推理过程和答案")
    print("输入 'quit' 退出")
    print("-" * 50)
    
    while True:
        question = input("\n📝 请输入数学问题: ").strip()
        
        if question.lower() in ['quit', 'exit', '退出']:
            break
        
        if not question:
            continue
        
        try:
            print("\n🤖 模型思考中...")
            response = run_inference(question)
            
            reasoning = extract_reasoning_from_response(response)
            answer = extract_solution_from_response(response)
            
            print(f"\n📋 完整回答:")
            print(response)
            
            print(f"\n🧠 推理过程:")
            print(reasoning if reasoning else "未提取到推理过程")
            
            print(f"\n💡 最终答案:")
            print(answer if answer else "未提取到答案")
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
        
        print("-" * 50)

# ============================================================================
# 7. 主程序
# ============================================================================

if __name__ == "__main__":
    print("\n🎮 请选择测试模式:")
    print("1. 批量评估 (默认20个样本)")
    print("2. 交互式测试")
    print("3. 单个样本测试")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == "2":
        interactive_test()
    elif choice == "3":
        # 单个样本测试
        sample = test_dataset[0]
        question = sample['question']
        true_answer = extract_answer_from_gsm8k(sample['answer'])
        
        print(f"\n📝 测试问题: {question}")
        print(f"📊 正确答案: {true_answer}")
        print("\n🤖 模型推理中...")
        
        response = run_inference(question)
        reasoning = extract_reasoning_from_response(response)
        predicted_answer = extract_solution_from_response(response)
        
        print(f"\n📋 完整回答:\n{response}")
        print(f"\n🧠 提取的推理:\n{reasoning}")
        print(f"\n💡 提取的答案: {predicted_answer}")
        print(f"✅ 正确答案: {true_answer}")
        
    else:
        # 默认批量评估
        num_samples = 20
        try:
            custom_num = input(f"请输入测试样本数 (默认{num_samples}): ").strip()
            if custom_num:
                num_samples = int(custom_num)
        except:
            pass
        
        evaluate_model(num_samples)
    
    print("\n✨ 推理测试完成！") 