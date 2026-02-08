# Model Evaluator

基于统一格式 (Unified Format) 的大模型能力评估工具。

**GitHub**: https://github.com/bonetthanchett-cell/model-evaluator

## 特性

- 支持 YAML 配置文件管理模型端点
- 从 .env 文件读取 API 密钥
- 命令行参数指定输入/输出路径（支持绝对或相对路径）
- **支持从任意目录调用**
- 批量处理和并发 workers
- 每完成一个 batch 自动保存结果
- 完整的日志记录
- 支持多种评估指标

## 安装

```bash
cd projects/model-evaluator

# 激活虚拟环境（推荐）
source .venv/bin/activate

# 安装依赖（如未安装）
pip install -r requirements.txt
```

## 使用方法

### 方式一：在项目目录中运行（推荐）

```bash
cd projects/model-evaluator
source .venv/bin/activate

python main.py \
    --input data/test_100.jsonl \
    --output results/ \
    --model glm-4.7 \
    --batch-size 10 \
    --workers 5
```

### 方式二：从其他目录调用（新功能）

```bash
# 从任意目录运行，使用绝对路径
python /home/wangyh/.openclaw/workspace/projects/model-evaluator/main.py \
    --input /path/to/your/test_data.jsonl \
    --output /path/to/your/results/ \
    --config /home/wangyh/.openclaw/workspace/projects/model-evaluator/config.yaml \
    --model glm-4.7 \
    --batch-size 10 \
    --workers 5

# 或使用相对路径
python ../model-evaluator/main.py \
    --input ./my_data.jsonl \
    --output ./my_results/ \
    --model glm-4.7
```

### 快捷脚本

创建一个全局可用的脚本：

```bash
# 创建软链接到 PATH
sudo ln -s /home/wangyh/.openclaw/workspace/projects/model-evaluator/main.py /usr/local/bin/model-eval

# 然后可以在任意目录使用
model-eval --input data.jsonl --output results/ --model glm-4.7
```

## 配置

1. **复制配置模板**:
```bash
cp config.example.yaml config.yaml
```

2. **编辑 `config.yaml`** 配置模型端点

3. **创建 `.env` 文件**存放 API 密钥:
```bash
# 可以在以下位置之一创建 .env 文件：
# 1. 当前工作目录 (./.env)
# 2. 脚本所在目录 (projects/model-evaluator/.env)
# 3. ~/.openclaw/workspace/.env

ZHIPU_API_KEY=your-key-here
MOONSHOT_API_KEY=your-key-here
KIMI_CODING_API_KEY=your-key-here
```

## 命令行参数

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--input` | `-i` | 输入测试数据 (JSONL，支持绝对/相对路径) | `--input /home/user/data/test.jsonl` |
| `--output` | `-o` | 输出结果目录 (支持绝对/相对路径) | `--output ./results/` |
| `--config` | `-c` | 配置文件路径 (可选，默认脚本目录的 config.yaml) | `--config /path/to/config.yaml` |
| `--model` | `-m` | 指定模型（覆盖配置） | `--model glm-4.7` |
| `--batch-size` | `-b` | 批处理大小 | `--batch-size 20` |
| `--workers` | `-w` | 并发 workers 数 | `--workers 5` |
| `--log-dir` | `-l` | 日志目录 (可选，默认输出目录下的 logs/) | `--log-dir /var/log/eval/` |

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

## 输出文件

- `results/results_*.jsonl` - 详细结果（每行一个 JSON）
- `results/stats_*.json` - 统计摘要
- `logs/eval_*.log` - 运行日志

## 支持的模型

| 模型 | 配置名 | Endpoint |
|------|--------|----------|
| GLM-4.7 | `glm-4.7` | 智谱AI 官方 API |
| Kimi Coding | `kimi-coding` | api.kimi.com |
| Kimi K2.5 | `kimi-k2.5` | Moonshot 官方 API |

## 示例

### 基本评估
```bash
python main.py -i data/test_100.jsonl -o results -m glm-4.7
```

### 使用绝对路径
```bash
python /home/user/projects/model-evaluator/main.py \
    -i /home/user/datasets/math_500.jsonl \
    -o /home/user/experiments/exp1/ \
    -m glm-4.7 \
    -b 20 \
    -w 5
```

### 使用自定义配置
```bash
python main.py \
    -i ./test.jsonl \
    -o ./results/ \
    -c /path/to/custom_config.yaml \
    -m my-custom-model
```

## 注意事项

- 首次运行前确保已创建 `.env` 文件并配置 API 密钥
- 建议始终使用虚拟环境运行
- 从其他目录调用时，建议显式指定 `--config` 路径
- 日志文件默认保存在输出目录的 `logs/` 子目录中
