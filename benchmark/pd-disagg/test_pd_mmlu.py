#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sglang.test.run_eval import run_eval
from sglang.test.simple_eval_common import (
    ChatCompletionSampler,
    set_ulimit,
)
from sglang.test.simple_eval_mmlu import subject2category

MMLU_CSV_FILE="https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv" #参考run_eval.py 可以替换为其余数据集
# 添加项目路径到Python路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root / "python"))

def test_mmlu_specific_subject(subject_to_test="machine_learning", num_examples=50, num_threads=8, 
                              model=None, host="0.0.0.0", port=30000, temperature=0.0, 
                              output_dir="/tmp/mmlu_results", dry_run=False):
    """
    测试MMLU的指定subject并详细保存模型输出
    
    Args:
        subject_to_test: 要测试的subject名称
        num_examples: 测试样本数量
        num_threads: 线程数
        model: 模型名称
        host: 服务器地址
        port: 服务器端口
        temperature: 生成温度
        output_dir: 输出目录
        dry_run: 是否只处理数据，不运行评估
    """
    print(f"开始测试MMLU subject: {subject_to_test}")
    print(f"测试样本数量: {num_examples}")
    if dry_run:
        print("DRY RUN模式: 只处理数据，不运行评估")
    
    # 加载完整数据集并筛选指定subject
    data_file = MMLU_CSV_FILE
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {data_file}")
        return None
    
    # 筛选指定subject的数据，如果subject不存在则使用全部数据
    if subject_to_test.lower() == "all" or subject_to_test == "":
        # 处理整个数据集
        subject_data = df
        print("使用整个MMLU数据集进行测试")
        print(f"数据集总共 {len(subject_data)} 个样本")
        output_subject_name = "all_subjects"
    else:
        subject_data = df[df['Subject'] == subject_to_test]
        
        if len(subject_data) == 0:
            print(f"错误: Subject '{subject_to_test}' 在数据集中不存在")
            print("可用的subjects:")
            available_subjects = sorted(df['Subject'].unique())
            for i, subject in enumerate(available_subjects, 1):
                print(f"  {i:2d}. {subject}")
            print("\n提示: 使用 '--subject all' 或不指定subject来测试整个数据集")
            return None
        
        print(f"Subject '{subject_to_test}' 总共有 {len(subject_data)} 个样本")
        output_subject_name = subject_to_test
    
    # 如果样本数量超过请求的数量，随机采样
    if len(subject_data) > num_examples:
        #subject_data = subject_data.sample(n=num_examples, random_state=42)
        subject_data = subject_data[:num_examples]
        print(f"取前 {num_examples} 个样本")
    else:
        num_examples = len(subject_data)
        print(f"使用全部 {num_examples} 个样本")
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 保存筛选后的数据到临时文件
    temp_file = f"/tmp/mmlu_{output_subject_name}.csv"
    subject_data.to_csv(temp_file, index=False)
    print(f"筛选后的数据保存到: {temp_file}")
    
    if dry_run:
        print("前3个样本示例:")
        for i, (_, row) in enumerate(subject_data.head(3).iterrows()):
            print(f"\n样本 {i+1}:")
            print(f"  问题: {row['Question'][:200]}...")
            print(f"  A: {row['A'][:100]}...")
            print(f"  B: {row['B'][:100]}...")
            print(f"  C: {row['C'][:100]}...")
            print(f"  D: {row['D'][:100]}...")
            print(f"  正确答案: {row['Answer']}")
        
        # 保存样本数据
        sample_file = output_dir / f"sample_data_{output_subject_name}.csv"
        subject_data.to_csv(sample_file, index=False, encoding='utf-8')
        print(f"\n样本数据保存到: {sample_file}")
        
        # 生成模拟的详细输出（仅用于演示格式）
        detailed_results = []
        for i, (_, row) in enumerate(subject_data.iterrows()):
            result_detail = {
                "index": i,
                "subject": row["Subject"],
                "question": row["Question"],
                "option_a": row["A"],
                "option_b": row["B"],
                "option_c": row["C"],
                "option_d": row["D"],
                "correct_answer": row["Answer"],
                "model_response": "[DRY RUN - 模拟模型响应]",
                "extracted_answer": "A",  # 模拟答案
                "is_correct": False,  # DRY RUN模式，标记为未测试
            }
            detailed_results.append(result_detail)
        
        detail_file = output_dir / f"dry_run_results_{output_subject_name}.json"
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        print(f"DRY RUN结果模板保存到: {detail_file}")
        
        return {"dry_run": True, "num_samples": num_examples}
    
    # 设置测试参数
    args = SimpleNamespace(
        base_url=f"http://{host}:{port}",  # 使用默认的本地服务
        eval_name="mmlu",
        mmlu_test_file=temp_file,  # 使用筛选后的数据文件
        num_examples=num_examples,
        num_threads=num_threads,
        model=model,
        temperature=temperature,
        top_k=1,
        top_p=1,
        min_p=0,
        frequency_penalty=0,
        presence_penalty=0,
        repetition_penalty=0,
        seed=50
    )
    
    # 运行评估
    print("开始运行评估...")
    metrics = run_eval(args)
    # metrics = {}
    
    # 保存评估指标
    metrics_file = output_dir / f"metrics_{output_subject_name}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"评估指标保存到: {metrics_file}")
    
    # 使用自定义方式保存详细输出
    set_ulimit()

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "EMPTY"
    #save_detailed_outputs(args, output_subject_name, output_dir)
    
    print(f"测试完成！总体得分: {metrics.get('score', 0):.3f}")
    
    return metrics

def save_detailed_outputs(args, subject, output_dir):
    """
    保存详细的模型输出，包括每个问题的详细信息
    """
    # 创建自定义评估器来获取详细信息
    from sglang.test.simple_eval_common import (
        ANSWER_PATTERN_MULTICHOICE,
        format_multichoice_question,
    )
    
    # 重新加载数据
    df = pd.read_csv(args.mmlu_test_file)
    examples = [row.to_dict() for _, row in df.iterrows()]
    
    # 创建采样器
    base_url = (
        f"{args.base_url}/v1" if args.base_url else f"http://{args.host}:{args.port}/v1"
    )
    sampler = ChatCompletionSampler(
        model=args.model,
        max_tokens=2048,
        base_url=base_url,
        temperature=args.temperature,
    )
    
    detailed_results = []
    
    print("开始收集详细输出...")
    for i, example in enumerate(examples):
        print(f"处理第 {i+1}/{len(examples)} 个问题...")
        
        try:
            # 格式化问题
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(example), role="user"
                )
            ]
            
            # 获取模型响应
            response_text = sampler(prompt_messages)
            
            # 提取答案
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1) if match else None
            
            # 判断正确性
            is_correct = extracted_answer == example["Answer"]
            
            # 保存详细信息
            result_detail = {
                "index": i,
                "subject": example["Subject"],
                "question": example["Question"],
                "option_a": example["A"],
                "option_b": example["B"],
                "option_c": example["C"],
                "option_d": example["D"],
                "correct_answer": example["Answer"],
                "model_response": response_text,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
            }
            
            detailed_results.append(result_detail)
            
        except Exception as e:
            print(f"处理第 {i+1} 个问题时出错: {e}")
            # 保存错误信息
            result_detail = {
                "index": i,
                "subject": example.get("Subject", ""),
                "question": example.get("Question", ""),
                "error": str(e),
                "is_correct": False,
            }
            detailed_results.append(result_detail)
    
    # 保存详细结果
    detail_file = output_dir / f"detailed_results_{subject}.json"
    with open(detail_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"详细结果保存到: {detail_file}")
    
    # 保存CSV格式的结果，便于查看
    csv_file = output_dir / f"results_{subject}.csv"
    df_results = pd.DataFrame(detailed_results)
    df_results.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"CSV结果保存到: {csv_file}")
    
    # 统计信息
    total_questions = len(detailed_results)
    correct_count = sum(1 for r in detailed_results if r.get("is_correct", False))
    accuracy = correct_count / total_questions if total_questions > 0 else 0
    
    print(f"\n=== {subject} 测试统计 ===")
    print(f"总问题数: {total_questions}")
    print(f"正确答案数: {correct_count}")
    print(f"准确率: {accuracy:.3f}")
    
    # 显示错误答案的问题
    wrong_answers = [r for r in detailed_results if not r.get("is_correct", False) and "error" not in r]
    if wrong_answers:
        print(f"\n错误的 {len(wrong_answers)} 个问题:")
        for wrong in wrong_answers[:5]:  # 只显示前5个
            print(f"  问题: {wrong['question'][:100]}...")
            print(f"  正确答案: {wrong['correct_answer']}, 模型答案: {wrong['extracted_answer']}")
            print()

def list_subjects():
    """列出所有可用的subjects"""
    print("可用的MMLU subjects:")
    print("-" * 50)
    subjects_by_category = {}
    
    for subject, category in subject2category.items():
        if category not in subjects_by_category:
            subjects_by_category[category] = []
        subjects_by_category[category].append(subject)
    
    for category in sorted(subjects_by_category.keys()):
        print(f"\n{category.upper()} ({len(subjects_by_category[category])} subjects):")
        for subject in sorted(subjects_by_category[category]):
            print(f"  - {subject}")
    
    print(f"\n总计: {len(subject2category)} 个subjects")

def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description='测试MMLU指定subject')
    parser.add_argument('--subject', type=str, default="all", help='要测试的subject名称')
    parser.add_argument('--num-examples', type=int, default=50, help='测试样本数量 (默认: 50)')
    parser.add_argument('--num-threads', type=int, default=8, help='线程数 (默认: 8)')
    parser.add_argument('--model', type=str, default=None, help='模型名称')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=30000, help='服务器端口 (默认: 30000)')
    parser.add_argument('--temperature', type=float, default=0.0, help='生成温度 (默认: 0.0)')
    parser.add_argument('--output-dir', type=str, default='/tmp/tree_result', help='输出目录 (默认: /tmp/mmlu_results)')
    parser.add_argument('--list-subjects', action='store_true', help='列出所有可用的subjects')
    parser.add_argument('--dry-run', action='store_true', help='只处理数据，不运行评估')
    
    args = parser.parse_args()
    
    if args.list_subjects:
        list_subjects()
        return
    
    # 检查subject是否有效，允许 "all" 来测试整个数据集
    if args.subject.lower() != "all" and args.subject not in subject2category:
        print(f"错误: Subject '{args.subject}' 不存在")
        print("使用 --list-subjects 查看可用的subjects")
        print("提示: 使用 '--subject all' 来测试整个数据集")
        return
    
    metrics = test_mmlu_specific_subject(
        subject_to_test=args.subject,
        num_examples=args.num_examples,
        num_threads=args.num_threads,
        model=args.model,
        host=args.host,
        port=args.port,
        temperature=args.temperature,
        output_dir=args.output_dir,
        dry_run=args.dry_run
    )
    
    if metrics:
        if args.dry_run:
            print(f"\nDRY RUN完成！处理样本数: {metrics.get('num_samples', 0)}")
        else:
            print(f"\n测试完成！准确率: {metrics.get('score', 0):.3f}")
    else:
        print("测试失败！")

if __name__ == "__main__":
    main()
