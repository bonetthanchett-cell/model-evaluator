---
name: model-evaluator
description: Evaluate LLM capabilities using unified format test datasets. Supports multiple models, batch processing, custom system prompts, and automatic retry with error handling.
---

# Model Evaluator

A comprehensive tool for evaluating Large Language Model (LLM) capabilities using unified format test datasets.

## Features

- **Unified Format**: Standardized JSONL input format for all benchmark types
- **Multi-Model Support**: Works with GLM-4, Kimi, OpenAI, and other OpenAI-compatible APIs
- **Batch Processing**: Concurrent evaluation with configurable workers
- **Streaming Results**: Saves results after each batch to prevent data loss
- **Custom System Prompts**: Load system prompts from markdown files
- **Automatic Retry**: Built-in retry mechanism for API failures (1 retry with exponential backoff)
- **Error Handling**: Failed API calls are logged and marked as negative samples with empty output
- **Comprehensive Logging**: Detailed logs for debugging and monitoring

## Installation

```bash
# Clone or copy the skill to your OpenClaw skills directory
cd ~/.openclaw/workspace/skills/model-evaluator

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Evaluate with default settings
model-eval --input data/test.jsonl --output results/ --model glm-4.7

# Full example with all options
model-eval \
  --input /path/to/test_data.jsonl \
  --output /path/to/results/ \
  --model glm-4.7 \
  --batch-size 20 \
  --workers 5 \
  --sys-prompt /path/to/prompt.md
```

### CLI Options

| Option | Short | Description | Example |
|--------|-------|-------------|---------|
| `--input` | `-i` | Input test data (JSONL) | `-i data/test.jsonl` |
| `--output` | `-o` | Output directory | `-o results/` |
| `--model` | `-m` | Model name from config | `-m glm-4.7` |
| `--config` | `-c` | Config file path | `-c config.yaml` |
| `--batch-size` | `-b` | Batch size | `-b 20` |
| `--workers` | `-w` | Concurrent workers | `-w 5` |
| `--sys-prompt` | `-s` | System prompt file (.md/.txt) | `-s prompt.md` |
| `--log-dir` | `-l` | Log directory | `-l logs/` |

## Input Format

The evaluator expects data in **Unified Format** (JSONL):

```json
{
  "_meta": {
    "benchmark": "gsm8k",
    "task_id": "gsm8k_0",
    "task_type": "math_qa",
    "language": "en",
    "tags": ["arithmetic", "word_problem"],
    "domain": "math"
  },
  "input": {
    "question": "Janet's ducks lay 16 eggs per day...",
    "context": "",
    "choices": [],
    "prompt": ""
  },
  "output": {
    "answer": "18",
    "solution": "Detailed solution...",
    "code": "",
    "tests": []
  },
  "evaluation": {
    "metric": "exact_match",
    "ground_truth": "18",
    "checker": "extract_and_compare_number"
  }
}
```

### Supported Task Types

- `math_qa` - Math word problems
- `multiple_choice` - Multiple choice QA
- `code_gen` - Code generation
- `code_repair` - Code repair/debugging
- `reading_comp` - Reading comprehension
- `logical_reason`` - Logical reasoning
- `open_qa` - Open-domain QA

## Configuration

Create a `config.yaml` file:

```yaml
# Default model
default_model: "glm-4.7"

# Model configurations
models:
  glm-4.7:
    provider: "openai"
    endpoint: "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    model: "glm-4.7"
    api_key_env: "ZHIPU_API_KEY"
    max_tokens: 8192
    temperature: 0.5
    
  kimi-k2.5:
    provider: "openai"
    endpoint: "https://api.moonshot.cn/v1/chat/completions"
    model: "kimi-k2.5"
    api_key_env: "MOONSHOT_API_KEY"
    max_tokens: 8192
    temperature: 0.5

# Execution settings
execution:
  batch_size: 10
  workers: 5
  request_interval: 0.1

# Evaluation settings
evaluation:
  timeout: 300
  retry:
    max_attempts: 3
    backoff: 2.0
```

### Environment Variables

Create a `.env` file:

```bash
# API Keys
ZHIPU_API_KEY=your-zhipu-key
MOONSHOT_API_KEY=your-moonshot-key
KIMI_CODING_API_KEY=your-kimi-key
```

## System Prompts

Create a markdown file for system prompts:

```markdown
# system_prompt.md

You are a mathematical reasoning expert. 
When solving problems:
1. Identify the key variables
2. Set up equations if needed
3. Show step-by-step calculations
4. Provide the final answer clearly
```

Use it with:
```bash
model-eval -i data.jsonl -o results/ -m glm-4.7 -s system_prompt.md
```

## Output

### Results File (`results/results_{model}.jsonl`)

Each line contains a JSON object:

```json
{
  "task_id": "gsm8k_0",
  "task_type": "math_qa",
  "prompt": "Question content...",
  "system_prompt": "System prompt content...",
  "prediction": "Model's answer",
  "ground_truth": {"answer": "18", ...},
  "evaluation": {"correct": true, "metric": "exact_match"},
  "latency": 2.34,
  "success": true
}
```

### Statistics File (`results/stats_{model}.json`)

```json
{
  "total": 100,
  "successful": 99,
  "correct": 81,
  "accuracy": 0.81,
  "success_rate": 0.99,
  "avg_latency": 28.72,
  "by_type": {
    "math_qa": {"total": 100, "correct": 81, "accuracy": 0.81}
  },
  "model": "glm-4"
}
```

### Log Files (`logs/eval_YYYYMMDD_HHMMSS.log`)

Detailed logs including:
- Batch progress
- Individual task results
- API errors and retries
- Performance metrics

## Error Handling & Retry Mechanism

The evaluator includes robust error handling for API failures:

### Automatic Retry
- **Retry Count**: 1 automatic retry per failed API call
- **Backoff Strategy**: Exponential backoff (2^attempt seconds)
- **Timeout**: Configurable via `config.yaml`

### Failed Request Handling
When an API call fails (even after retry):
- `prediction` is set to empty string `""`
- `evaluation.correct` is set to `false` (negative sample)
- `success` is set to `false`
- Error details are logged at ERROR level
- Evaluation continues with remaining samples

### Example Failed Output
```json
{
  "task_id": "gsm8k_42",
  "task_type": "math_qa",
  "prompt": "Question content...",
  "prediction": "",
  "ground_truth": {"answer": "25"},
  "evaluation": {"correct": false, "error": "API rate limit exceeded"},
  "latency": 5.23,
  "success": false,
  "error": "API rate limit exceeded"
}
```

### Log Messages
```
WARNING: API 调用失败，准备重试 (1/2): Connection timeout
ERROR: API 调用失败，已重试 1 次仍失败: API rate limit exceeded
ERROR: 任务 gsm8k_42 API 调用失败: API rate limit exceeded
```

## Advanced Usage

### Running from Any Directory

```bash
# Using absolute path
python /path/to/model-evaluator/main.py \
  -i /path/to/data.jsonl \
  -o /path/to/output/ \
  -c /path/to/config.yaml \
  -m glm-4.7

# Create a global alias
alias model-eval='python /path/to/model-evaluator/main.py'
```

### Batch Evaluation Script

```bash
#!/bin/bash
# evaluate_all.sh

MODELS=("glm-4.7" "kimi-k2.5")
DATASETS=("data/gsm8k.jsonl" "data/math500.jsonl")

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    model-eval \
      -i "$dataset" \
      -o "results/${model}_$(basename $dataset .jsonl)/" \
      -m "$model" \
      -b 20 \
      -w 5
  done
done
```

### Python API

```python
from main import ModelEvaluator, EvalConfig, load_config

# Load config
config = load_config("config.yaml")

# Create eval config
eval_config = EvalConfig(
    model_name="glm-4.7",
    model_config=config["models"]["glm-4.7"],
    batch_size=10,
    workers=5,
    system_prompt="You are a math expert..."
)

# Run evaluation
evaluator = ModelEvaluator(eval_config)
stats = evaluator.run("data/test.jsonl", "results/")

print(f"Accuracy: {stats['accuracy']:.2%}")
print(f"Success Rate: {stats['success_rate']:.2%}")
```

## Troubleshooting

### API Key Not Found

```bash
Error: API key not found in environment: ZHIPU_API_KEY
```

**Solution**: Check your `.env` file is in one of these locations:
- Current working directory
- Script directory
- `~/.openclaw/workspace/.env`

### Model Not Found

```bash
Error: Model 'glm-4.7' not found in config
```

**Solution**: Verify the model name in your `config.yaml` matches the `--model` argument.

### API Rate Limit / Timeout

```bash
ERROR: API 调用失败，已重试 1 次仍失败: Rate limit exceeded
```

**Solution**: 
- Reduce `--workers` to lower concurrency
- Increase `--batch-size` to reduce API call frequency
- Check your API provider's rate limits

### Import Errors

```bash
ModuleNotFoundError: No module named 'yaml'
```

**Solution**: Install dependencies
```bash
pip install pyyaml python-dotenv requests tqdm
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License
