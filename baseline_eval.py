#!/usr/bin/env python3
"""
Model Evaluator - Baseline Version with Basic Prompt
使用基础 prompt: "You are a helpful assistant."
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Import from original modules
from evaluators import get_evaluator
from model_client import ModelClient
from utils import load_jsonl, save_jsonl, format_result


# 基础 Prompt - Baseline
BASELINE_SYSTEM_PROMPT = "You are a helpful assistant. Please answer the following question directly and concisely."


# 配置日志
def setup_logging(output_dir: str = "logs") -> logging.Logger:
    """配置日志记录"""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"baseline_eval_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger("baseline_evaluator")
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志文件: {log_file}")
    return logger


@dataclass
class EvalConfig:
    """评估配置类"""
    model_name: str
    model_config: Dict[str, Any]
    batch_size: int = 10
    workers: int = 10
    timeout: int = 300
    retry_max: int = 2
    retry_backoff: float = 1.0
    output_dir: str = "results"
    request_interval: float = 0.05
    system_prompt: str = BASELINE_SYSTEM_PROMPT


class BaselineEvaluator:
    """Baseline 评估器"""
    
    def __init__(self, config: EvalConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.client = ModelClient(config.model_config, config.timeout)
        self.results: List[Dict] = []
        self.output_path: Optional[Path] = None
        self.results_file: Optional[Path] = None
        self.stats_file: Optional[Path] = None
        self.logger = logger or logging.getLogger("baseline_evaluator")
        
    def evaluate_single(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个任务"""
        task_id = task.get("_meta", {}).get("task_id", "unknown")
        task_type = task.get("_meta", {}).get("task_type", "unknown")
        benchmark = task.get("_meta", {}).get("benchmark", "unknown")
        
        # 构建 prompt
        prompt = self._build_prompt(task)
        
        # 调用模型
        start_time = time.time()
        try:
            # 使用 system prompt + user prompt
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.generate_with_messages(messages)
            latency = time.time() - start_time
            
            # 评估结果
            evaluator = get_evaluator(task)
            evaluation_result = evaluator.evaluate(
                prediction=response,
                ground_truth=task
            )
            
            is_correct = evaluation_result.get("correct", False)
            
            result = {
                "task_id": task_id,
                "benchmark": benchmark,
                "task_type": task_type,
                "prompt": prompt,
                "prediction": response,
                "ground_truth": task.get("output", {}),
                "evaluation": evaluation_result,
                "latency": latency,
                "success": True
            }
        except Exception as e:
            latency = time.time() - start_time
            self.logger.error(f"任务 {task_id} 失败: {str(e)}")
            
            result = {
                "task_id": task_id,
                "benchmark": benchmark,
                "task_type": task_type,
                "prompt": prompt,
                "prediction": None,
                "ground_truth": task.get("output", {}),
                "evaluation": {"correct": False, "error": str(e)},
                "latency": latency,
                "success": False,
                "error": str(e)
            }
        
        return result
    
    def _build_prompt(self, task: Dict[str, Any]) -> str:
        """从统一格式构建 prompt"""
        input_data = task.get("input", {})
        question = input_data.get("question", "")
        context = input_data.get("context", "")
        choices = input_data.get("choices", [])
        
        parts = []
        if context:
            parts.append(f"Context: {context}")
        
        if question:
            parts.append(f"Question: {question}")
        
        # 处理多选题
        if choices:
            parts.append("Options:")
            for i, choice in enumerate(choices):
                parts.append(f"{chr(65+i)}. {choice}")
            parts.append("\nAnswer (provide the letter or the answer text):")
        else:
            parts.append("\nAnswer:")
        
        return "\n".join(parts)
    
    def evaluate_batch(self, tasks: List[Dict[str, Any]], 
                       progress_desc: str = "Evaluating") -> List[Dict[str, Any]]:
        """批量评估任务"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            future_to_task = {
                executor.submit(self.evaluate_single, task): task 
                for task in tasks
            }
            
            with tqdm(total=len(tasks), desc=progress_desc) as pbar:
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        task = future_to_task[future]
                        task_id = task.get("_meta", {}).get("task_id", "unknown")
                        results.append({
                            "task_id": task_id,
                            "error": str(e),
                            "success": False
                        })
                    pbar.update(1)
                    
                    if self.config.request_interval > 0:
                        time.sleep(self.config.request_interval)
        
        return results
    
    def _init_output(self, output_dir: str):
        """初始化输出目录"""
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.results_file = self.output_path / f"baseline_results_{self.config.model_name}.jsonl"
        self.stats_file = self.output_path / f"baseline_stats_{self.config.model_name}.json"
        
        # 清空已有结果文件
        if self.results_file.exists():
            self.results_file.unlink()
    
    def _append_results(self, results: List[Dict]):
        """追加结果到文件"""
        with open(self.results_file, "a", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    def _save_intermediate_stats(self, completed: int, total: int) -> Dict:
        """保存中间统计"""
        stats = self._compute_stats(self.results)
        stats["progress"] = {"completed": completed, "total": total}
        
        temp_stats_file = self.output_path / f"baseline_stats_{self.config.model_name}_temp.json"
        with open(temp_stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        return stats
    
    def run(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行完整评估流程"""
        total_tasks = len(tasks)
        batch_size = self.config.batch_size
        total_batches = (total_tasks + batch_size - 1) // batch_size
        
        self.logger.info(f"="*50)
        self.logger.info(f"Baseline 评估开始")
        self.logger.info(f"模型: {self.config.model_name}")
        self.logger.info(f"总任务: {total_tasks}")
        self.logger.info(f"Batch 大小: {batch_size}")
        self.logger.info(f"Workers: {self.config.workers}")
        self.logger.info(f"="*50)
        
        print(f"\n{'='*50}")
        print(f"Baseline Evaluation")
        print(f"{'='*50}")
        print(f"Model: {self.config.model_name}")
        print(f"Total tasks: {total_tasks}")
        print(f"Batch size: {batch_size}")
        print(f"Workers: {self.config.workers}")
        print(f"System prompt: {self.config.system_prompt[:50]}...")
        print(f"{'='*50}\n")
        
        all_results = []
        
        for batch_num in range(1, total_batches + 1):
            start_idx = (batch_num - 1) * batch_size
            end_idx = min(start_idx + batch_size, total_tasks)
            batch = tasks[start_idx:end_idx]
            
            self.logger.info(f"处理 Batch {batch_num}/{total_batches} ({len(batch)} tasks)")
            
            batch_results = self.evaluate_batch(
                batch, 
                progress_desc=f"Batch {batch_num}/{total_batches}"
            )
            
            all_results.extend(batch_results)
            self.results = all_results
            
            self._append_results(batch_results)
            
            completed = min(end_idx, total_tasks)
            stats = self._save_intermediate_stats(completed, total_tasks)
            
            batch_success = sum(1 for r in batch_results if r.get("success", False))
            batch_correct = sum(1 for r in batch_results if r.get("evaluation", {}).get("correct", False))
            
            self.logger.info(f"Batch {batch_num}/{total_batches} 完成: "
                           f"成功 {batch_success}/{len(batch_results)}, "
                           f"正确 {batch_correct}/{len(batch_results)}, "
                           f"准确率 {batch_correct/len(batch_results):.1%}")
            
            print(f"\n✓ Batch {batch_num}/{total_batches} completed ({len(batch_results)} tasks)")
            print(f"  Progress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")
            if stats:
                print(f"  Current accuracy: {stats['accuracy']:.1%}")
            print("-" * 50)
        
        final_stats = self._compute_stats(all_results)
        self._save_final_results(all_results, final_stats)
        
        return final_stats
    
    def _compute_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """计算评估统计"""
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        correct = sum(1 for r in results 
                     if r.get("evaluation", {}).get("correct", False))
        
        latencies = [r.get("latency", 0) for r in results if r.get("success", False)]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        # 按 benchmark 统计
        bench_stats = {}
        for r in results:
            bench = r.get("benchmark", "unknown")
            if bench not in bench_stats:
                bench_stats[bench] = {"total": 0, "correct": 0}
            bench_stats[bench]["total"] += 1
            if r.get("evaluation", {}).get("correct", False):
                bench_stats[bench]["correct"] += 1
        
        for bench, stat in bench_stats.items():
            stat["accuracy"] = stat["correct"] / stat["total"] if stat["total"] > 0 else 0
        
        return {
            "total": total,
            "successful": successful,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "success_rate": successful / total if total > 0 else 0,
            "avg_latency": avg_latency,
            "by_benchmark": bench_stats,
            "model": self.config.model_name,
            "system_prompt": self.config.system_prompt
        }
    
    def _save_final_results(self, results: List[Dict], stats: Dict):
        """保存最终结果"""
        try:
            if self.stats_file:
                with open(self.stats_file, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
                self.logger.info(f"统计结果已保存到: {self.stats_file}")
            
            print(f"\nResults saved to: {self.results_file}")
            print(f"Stats saved to: {self.stats_file}")
            
            self._print_summary(stats)
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}", exc_info=True)
            raise
    
    def _print_summary(self, stats: Dict):
        """打印评估摘要"""
        print("\n" + "="*50)
        print("Baseline Evaluation Summary")
        print("="*50)
        print(f"Model: {stats['model']}")
        print(f"System Prompt: {stats['system_prompt'][:60]}...")
        print(f"Total tasks: {stats['total']}")
        print(f"Successful: {stats['successful']} ({stats['success_rate']:.1%})")
        print(f"Correct: {stats['correct']} ({stats['accuracy']:.1%})")
        print(f"Avg latency: {stats['avg_latency']:.2f}s")
        
        if stats['by_benchmark']:
            print("\nBy Benchmark:")
            for bench, stat in sorted(stats['by_benchmark'].items()):
                print(f"  {bench}: {stat['correct']}/{stat['total']} ({stat['accuracy']:.1%})")


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Model Evaluator - 基础 Prompt 评估"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入测试数据文件 (JSONL 格式)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出结果目录"
    )
    parser.add_argument(
        "--config", "-c",
        default="config_baseline.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--model", "-m",
        default="kimi-k2.5",
        help="指定模型名称"
    )
    
    args = parser.parse_args()
    
    # 加载环境变量
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建 EvalConfig
    model_config = config.get("models", {}).get(args.model, {})
    if not model_config:
        raise ValueError(f"Model '{args.model}' not found in config")
    
    exec_config = config.get("execution", {})
    
    eval_config = EvalConfig(
        model_name=args.model,
        model_config=model_config,
        batch_size=exec_config.get("batch_size", 10),
        workers=exec_config.get("workers", 10),
        timeout=config.get("evaluation", {}).get("timeout", 300),
        output_dir=args.output,
        request_interval=exec_config.get("request_interval", 0.05),
        system_prompt=config.get("prompts", {}).get("baseline", {}).get("system", BASELINE_SYSTEM_PROMPT)
    )
    
    # 初始化
    logger = setup_logging(eval_config.output_dir)
    evaluator = BaselineEvaluator(eval_config, logger)
    evaluator._init_output(eval_config.output_dir)
    
    # 加载测试数据
    tasks = load_jsonl(args.input)
    logger.info(f"加载了 {len(tasks)} 个测试任务")
    
    # 运行评估
    stats = evaluator.run(tasks)
    
    return stats


if __name__ == "__main__":
    main()
