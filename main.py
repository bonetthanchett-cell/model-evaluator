#!/usr/bin/env python3
"""
Model Evaluator - 基于统一格式的大模型能力评估工具
支持每完成一个 batch 就保存到文件
支持从任意目录调用
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

from evaluators import get_evaluator
from model_client import ModelClient
from utils import load_jsonl, save_jsonl, format_result


# 全局变量存储程序根目录
SCRIPT_DIR = Path(__file__).parent.resolve()


# 配置日志
def setup_logging(output_dir: str, log_level: int = logging.DEBUG) -> logging.Logger:
    """配置日志记录"""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    # 创建 logger
    logger = logging.getLogger("model_evaluator")
    logger.setLevel(log_level)
    
    # 清除已有的处理器（避免重复）
    logger.handlers.clear()
    
    # 文件处理器 - 记录所有日志
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    
    # 控制台处理器 - 只记录 INFO 及以上
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志文件: {log_file}")
    return logger


@dataclass
class EvalConfig:
    """评估配置类"""
    model_name: str
    model_config: Dict[str, Any]
    batch_size: int = 5
    workers: int = 4
    timeout: int = 60
    retry_max: int = 3
    retry_backoff: float = 2.0
    output_dir: str = "results"
    request_interval: float = 0.1
    system_prompt: Optional[str] = None  # 添加 system prompt 支持


class ModelEvaluator:
    """模型评估器主类"""
    
    def __init__(self, config: EvalConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.client = ModelClient(config.model_config, config.timeout)
        self.results: List[Dict] = []
        self.output_path: Optional[Path] = None
        self.results_file: Optional[Path] = None
        self.stats_file: Optional[Path] = None
        self.logger = logger or logging.getLogger("model_evaluator")
        self.logger.debug(f"初始化 ModelEvaluator: model={config.model_name}, batch_size={config.batch_size}, workers={config.workers}")
        if config.system_prompt:
            self.logger.info(f"使用 System Prompt (长度: {len(config.system_prompt)} 字符)")
        
    def evaluate_single(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个任务"""
        task_id = task.get("_meta", {}).get("task_id", "unknown")
        task_type = task.get("_meta", {}).get("task_type", "unknown")
        
        self.logger.debug(f"开始评估任务: {task_id} (类型: {task_type})")
        
        # 构建 prompt
        prompt = self._build_prompt(task)
        
        # 调用模型
        start_time = time.time()
        try:
            # 如果有 system prompt，使用 messages 格式
            if self.config.system_prompt:
                messages = [
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt}
                ]
                response = self.client.generate_with_messages(messages)
            else:
                response = self.client.generate(prompt)
            
            latency = time.time() - start_time
            
            self.logger.debug(f"任务 {task_id} API 调用成功, 延迟: {latency:.2f}s")
            
            # 评估结果
            evaluator = get_evaluator(task)
            evaluation_result = evaluator.evaluate(
                prediction=response,
                ground_truth=task
            )
            
            is_correct = evaluation_result.get("correct", False)
            self.logger.debug(f"任务 {task_id} 评估结果: {'✓' if is_correct else '✗'}")
            
            result = {
                "task_id": task_id,
                "task_type": task_type,
                "prompt": prompt,
                "system_prompt": self.config.system_prompt,  # 记录使用的 system prompt
                "prediction": response,
                "ground_truth": task.get("output", {}),
                "evaluation": evaluation_result,
                "latency": latency,
                "success": True
            }
        except Exception as e:
            latency = time.time() - start_time
            self.logger.error(f"任务 {task_id} 评估失败: {str(e)}", exc_info=True)
            
            result = {
                "task_id": task_id,
                "task_type": task_type,
                "prompt": prompt,
                "system_prompt": self.config.system_prompt,
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
        
        # 构建完整 prompt
        parts = []
        if context:
            parts.append(f"Context:\n{context}\n")
        if question:
            parts.append(f"Question:\n{question}")
        
        return "\n".join(parts)
    
    def evaluate_batch(self, tasks: List[Dict[str, Any]], 
                       progress_desc: str = "Evaluating") -> List[Dict[str, Any]]:
        """批量评估任务"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self.evaluate_single, task): task 
                for task in tasks
            }
            
            # 收集结果
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
                    
                    # 请求间隔控制
                    if self.config.request_interval > 0:
                        time.sleep(self.config.request_interval)
        
        return results
    
    def _init_output(self, output_dir: str):
        """初始化输出目录和文件"""
        self.output_path = Path(output_dir).resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.results_file = self.output_path / f"results_{self.config.model_name}.jsonl"
        self.stats_file = self.output_path / f"stats_{self.config.model_name}.json"
        
        # 如果文件已存在，清空它们
        if self.results_file.exists():
            self.results_file.write_text("")
            self.logger.info(f"清空已存在的结果文件: {self.results_file}")
        
        self.logger.info(f"结果文件: {self.results_file}")
        self.logger.info(f"统计文件: {self.stats_file}")
        
        print(f"Results will be saved to: {self.results_file}")
        print(f"Stats will be saved to: {self.stats_file}")
    
    def _append_results(self, results: List[Dict]):
        """追加结果到文件"""
        if self.results_file:
            try:
                with open(self.results_file, "a", encoding="utf-8") as f:
                    for result in results:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                self.logger.debug(f"追加 {len(results)} 条结果到文件")
            except Exception as e:
                self.logger.error(f"保存结果到文件失败: {e}", exc_info=True)
                raise
    
    def _save_intermediate_stats(self, completed_count: int, total_count: int):
        """保存中间统计信息"""
        if not self.results_file:
            return
        
        # 计算当前统计
        stats = self._compute_stats(self.results)
        stats["progress"] = {
            "completed": completed_count,
            "total": total_count,
            "percentage": f"{(completed_count / total_count * 100):.1f}%"
        }
        
        # 保存统计
        if self.stats_file:
            try:
                with open(self.stats_file, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
                    f.flush()
                self.logger.debug(f"保存统计信息: {completed_count}/{total_count} ({stats['progress']['percentage']})")
            except Exception as e:
                self.logger.error(f"保存统计信息失败: {e}", exc_info=True)
        
        return stats
    
    def run(self, input_file: str, output_dir: str) -> Dict[str, Any]:
        """运行完整评估流程"""
        self.logger.info(f"="*50)
        self.logger.info(f"开始评估流程")
        self.logger.info(f"输入文件: {input_file}")
        self.logger.info(f"输出目录: {output_dir}")
        self.logger.info(f"模型: {self.config.model_name}")
        self.logger.info(f"Batch size: {self.config.batch_size}, Workers: {self.config.workers}")
        self.logger.info(f"="*50)
        
        # 加载测试数据
        input_path = Path(input_file).resolve()
        print(f"Loading test data from: {input_path}")
        self.logger.info(f"加载测试数据: {input_path}")
        
        try:
            tasks = load_jsonl(str(input_path))
            total_tasks = len(tasks)
            self.logger.info(f"成功加载 {total_tasks} 个任务")
        except Exception as e:
            self.logger.error(f"加载测试数据失败: {e}", exc_info=True)
            raise
        
        print(f"Loaded {total_tasks} tasks")
        
        # 初始化输出
        self._init_output(output_dir)
        
        # 批量评估
        print(f"\nEvaluating with model: {self.config.model_name}")
        print(f"Batch size: {self.config.batch_size}, Workers: {self.config.workers}")
        print(f"Total batches: {(total_tasks - 1) // self.config.batch_size + 1}")
        print("-" * 50)
        
        total_batches = (total_tasks - 1) // self.config.batch_size + 1
        all_results = []
        
        for i in range(0, total_tasks, self.config.batch_size):
            batch_num = i // self.config.batch_size + 1
            batch = tasks[i:i + self.config.batch_size]
            
            self.logger.info(f"开始处理 Batch {batch_num}/{total_batches} ({len(batch)} 个任务)")
            
            # 评估当前 batch
            batch_results = self.evaluate_batch(
                batch, 
                progress_desc=f"Batch {batch_num}/{total_batches}"
            )
            
            # 保存到内存
            all_results.extend(batch_results)
            self.results = all_results  # 更新用于统计
            
            # 立即追加到文件
            self._append_results(batch_results)
            
            # 保存中间统计
            completed = min(i + len(batch), total_tasks)
            stats = self._save_intermediate_stats(completed, total_tasks)
            
            # 打印 batch 完成信息
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
        
        # 计算最终统计
        final_stats = self._compute_stats(all_results)
        
        # 保存最终结果
        self._save_final_results(all_results, final_stats)
        
        self.logger.info(f"="*50)
        self.logger.info(f"评估流程完成")
        self.logger.info(f"总任务: {final_stats['total']}")
        self.logger.info(f"成功率: {final_stats['success_rate']:.1%}")
        self.logger.info(f"准确率: {final_stats['accuracy']:.1%}")
        self.logger.info(f"平均延迟: {final_stats['avg_latency']:.2f}s")
        self.logger.info(f"="*50)
        
        return final_stats
    
    def _compute_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """计算评估统计信息"""
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        correct = sum(1 for r in results 
                     if r.get("evaluation", {}).get("correct", False))
        
        latencies = [r.get("latency", 0) for r in results if r.get("success", False)]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        # 按 task_type 统计
        type_stats = {}
        for r in results:
            task_type = r.get("task_type", "unknown")
            if task_type not in type_stats:
                type_stats[task_type] = {"total": 0, "correct": 0}
            type_stats[task_type]["total"] += 1
            if r.get("evaluation", {}).get("correct", False):
                type_stats[task_type]["correct"] += 1
        
        for task_type, stat in type_stats.items():
            stat["accuracy"] = stat["correct"] / stat["total"] if stat["total"] > 0 else 0
        
        return {
            "total": total,
            "successful": successful,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
            "success_rate": successful / total if total > 0 else 0,
            "avg_latency": avg_latency,
            "by_type": type_stats,
            "model": self.config.model_name
        }
    
    def _save_final_results(self, results: List[Dict], stats: Dict):
        """保存最终结果"""
        try:
            if self.stats_file:
                with open(self.stats_file, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
                self.logger.info(f"最终结果已保存到: {self.stats_file}")
            
            print(f"\nResults saved to: {self.results_file}")
            print(f"Stats saved to: {self.stats_file}")
            
            # 打印摘要
            self._print_summary(stats)
        except Exception as e:
            self.logger.error(f"保存最终结果失败: {e}", exc_info=True)
            raise
    
    def _print_summary(self, stats: Dict):
        """打印评估摘要"""
        print("\n" + "="*50)
        print("Evaluation Summary")
        print("="*50)
        print(f"Model: {stats['model']}")
        print(f"Total tasks: {stats['total']}")
        print(f"Successful: {stats['successful']} ({stats['success_rate']:.1%})")
        print(f"Correct: {stats['correct']} ({stats['accuracy']:.1%})")
        print(f"Avg latency: {stats['avg_latency']:.2f}s")
        
        self.logger.info(f"评估摘要 - 模型: {stats['model']}, "
                        f"总任务: {stats['total']}, "
                        f"成功率: {stats['success_rate']:.1%}, "
                        f"准确率: {stats['accuracy']:.1%}")
        
        if stats['by_type']:
            print("\nBy Task Type:")
            for task_type, stat in stats['by_type'].items():
                print(f"  {task_type}: {stat['correct']}/{stat['total']} ({stat['accuracy']:.1%})")


def load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_env_file() -> Optional[Path]:
    """查找 .env 文件，按优先级搜索"""
    # 1. 当前工作目录
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env
    
    # 2. 脚本所在目录
    script_env = SCRIPT_DIR / ".env"
    if script_env.exists():
        return script_env
    
    # 3. workspace/.env
    workspace_env = Path.home() / ".openclaw/workspace/.env"
    if workspace_env.exists():
        return workspace_env
    
    return None


def create_eval_config(args, config: Dict) -> EvalConfig:
    """从参数和配置创建 EvalConfig"""
    model_name = args.model or config.get("default_model", "gpt-4")
    model_config = config.get("models", {}).get(model_name, {})
    
    if not model_config:
        raise ValueError(f"Model '{model_name}' not found in config")
    
    # 获取 API key
    api_key_env = model_config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"API key not found in environment: {api_key_env}")
    
    model_config["api_key"] = api_key
    
    # 执行配置
    exec_config = config.get("execution", {})
    eval_config = config.get("evaluation", {})
    
    # 读取 system prompt（如果指定了文件）
    system_prompt = None
    if args.sys_prompt:
        sys_prompt_path = Path(args.sys_prompt).resolve()
        if not sys_prompt_path.exists():
            raise FileNotFoundError(f"System prompt 文件不存在: {args.sys_prompt}")
        
        # 支持 .md 和 .txt 文件
        if sys_prompt_path.suffix.lower() in ['.md', '.txt', '']:
            with open(sys_prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
            print(f"已加载 System Prompt: {sys_prompt_path} ({len(system_prompt)} 字符)")
        else:
            raise ValueError(f"不支持的文件格式: {sys_prompt_path.suffix}，请使用 .md 或 .txt")
    
    return EvalConfig(
        model_name=model_name,
        model_config=model_config,
        batch_size=args.batch_size or exec_config.get("batch_size", 5),
        workers=args.workers or exec_config.get("workers", 4),
        timeout=eval_config.get("timeout", 60),
        retry_max=eval_config.get("retry", {}).get("max_attempts", 3),
        retry_backoff=eval_config.get("retry", {}).get("backoff", 2.0),
        output_dir=args.output,
        request_interval=exec_config.get("request_interval", 0.1),
        system_prompt=system_prompt
    )


def main():
    parser = argparse.ArgumentParser(
        description="Model Evaluator - 大模型能力评估工具 (支持任意目录调用)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入测试数据文件 (JSONL 格式，支持绝对或相对路径)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出结果目录 (支持绝对或相对路径)"
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="配置文件路径 (默认: 脚本目录下的 config.yaml)"
    )
    parser.add_argument(
        "--model", "-m",
        help="指定模型名称 (覆盖配置文件默认值)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        help="批处理大小 (覆盖配置文件)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        help="并发 workers 数 (覆盖配置文件)"
    )
    parser.add_argument(
        "--log-dir", "-l",
        default=None,
        help="日志目录 (默认: 输出目录下的 logs/)"
    )
    parser.add_argument(
        "--sys-prompt", "-s",
        default=None,
        help="System Prompt 文件路径 (.md 或 .txt)，将作为 system message 传递给大模型"
    )
    
    args = parser.parse_args()
    
    # 设置配置文件路径
    if args.config is None:
        # 默认使用脚本所在目录的 config.yaml
        args.config = str(SCRIPT_DIR / "config.yaml")
    
    # 加载 .env 文件（按优先级搜索）
    env_path = find_env_file()
    if env_path:
        load_dotenv(env_path)
        print(f"Loaded environment from: {env_path}")
    else:
        print("Warning: No .env file found")
    
    # 设置日志目录
    log_dir = args.log_dir if args.log_dir else str(Path(args.output) / "logs")
    
    try:
        # 设置日志
        logger = setup_logging(log_dir)
        logger.info(f"脚本目录: {SCRIPT_DIR}")
        logger.info(f"当前工作目录: {Path.cwd()}")
        logger.info(f"加载配置: {args.config}")
        
        # 加载配置
        config = load_config(args.config)
        
        # 创建评估配置
        eval_config = create_eval_config(args, config)
        
        # 运行评估
        evaluator = ModelEvaluator(eval_config, logger=logger)
        stats = evaluator.run(args.input, args.output)
        
        return 0 if stats["accuracy"] > 0 else 1
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"程序执行失败: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
