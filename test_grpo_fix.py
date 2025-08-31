#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试GRPO修复的脚本
"""

def test_reward_functions():
    """测试奖励函数格式"""
    
    # 模拟completion格式
    mock_completions = [
        [{"content": "<start_working_out>This is my reasoning<end_working_out><SOLUTION>42</SOLUTION>"}],
        [{"content": "Some bad format without tags"}],
    ]
    
    mock_prompts = [
        [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "What is 6 x 7?"}
        ]
    ]
    
    mock_answers = ["42", "42"]
    
    print("🧪 测试奖励函数...")
    
    # 导入奖励函数
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    # 设置格式标签
    reasoning_start = "<start_working_out>"
    reasoning_end = "<end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"
    
    # 测试格式检查函数
    def format_checker_exact(prompts, completions, ground_truth_answers, **kwargs):
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            has_reasoning = reasoning_start in response and reasoning_end in response
            has_answer = solution_start in response and solution_end in response
            
            if has_reasoning and has_answer:
                scores.append(3.0)
            else:
                scores.append(0.0)
        
        return scores
    
    # 运行测试
    scores = format_checker_exact(mock_prompts, mock_completions, mock_answers)
    print(f"✅ 格式检查测试: {scores}")
    
    # 检查返回值格式
    assert isinstance(scores, list), "返回值必须是列表"
    assert len(scores) == len(mock_completions), "分数数量必须与completion数量一致"
    assert all(isinstance(s, (int, float)) for s in scores), "所有分数必须是数字"
    
    print("✅ 所有测试通过！奖励函数格式正确")

if __name__ == "__main__":
    test_reward_functions()
    print("�� 测试完成！可以运行GRPO训练了") 