# Model Evaluator

基于统一格式 (Unified Format) 的大模型能力评估工具。

## 功能特性

- 支持 YAML 配置文件管理模型端点
- 从 .env 文件读取 API 密钥
- 命令行参数指定输入/输出
- 批量处理和并发 workers
- 支持多种评估指标

## 安装

```bash
cd projects/model-evaluator
pip install -r requirements.txt
```

## 配置

1. 复制配置模板:
```bash
cp config.example.yaml config.yaml
```

2. 编辑 `config.yaml` 配置模型端点

3. 创建 `.env` 文件存放 API 密钥:
```bash
OPENAI_API_KEY=your-key-here
```

## 使用

```bash
# 基本用法
python main.py --input data/test.jsonl --output results/

# 指定配置文件
python main.py --config my-config.yaml --input data/test.jsonl --output results/

# 覆盖配置中的参数
python main.py --input data/test.jsonl --output results/ --workers 8 --batch-size 10
```

## 输入数据格式

支持统一格式 JSONL:
```json
{
  "_meta": {
    "benchmark": "gsm8k",
    "task_id": "gsm8k_0",
    "task_type": "math_qa"
  },
  "input": {
    "question": "问题内容"
  },
  "output": {
    "answer": "正确答案"
  },
  "evaluation": {
    "metric": "exact_match",
    "ground_truth": "18"
  }
}
```
