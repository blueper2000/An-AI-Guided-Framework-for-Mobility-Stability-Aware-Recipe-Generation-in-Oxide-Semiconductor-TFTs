# Oxide Semiconductor Synthesis Recipe Prediction & Evaluation

This toolkit extracts structured synthesis recipes from research papers and evaluates predictions using LLM-as-a-Judge.

## Requirements

```bash
pip install vllm>=0.10.0 transformers>=4.54.0 jsonlines fire tqdm litellm openai
```

## Quick Start

### 1. Prediction

**Using Local vLLM (EXAONE-4.0-32B)**
```bash
# Requires ~65GB VRAM (4x RTX A6000 recommended)
CUDA_VISIBLE_DEVICES=0,1,2,3 python predict.py \
    --input_file data.jsonl \
    --backend vllm \
    --tensor_parallel_size 4
```

**Using OpenRouter API (GPT-4o)**
```bash
export OPENROUTER_API_KEY=your_api_key_here

python predict.py \
    --input_file data.jsonl \
    --backend openrouter \
    --model openai/gpt-4o
```

### 2. Evaluation

```bash
export OPENROUTER_API_KEY=your_api_key_here

python judge.py LGAI-EXAONE-EXAONE-4.0-32B/prediction.jsonl \
    --model openai/gpt-4o
```

### 3. Score Aggregation

```bash
python score.py prediction_gpt-4o_judged.jsonl
```

## Input Format

Input JSONL file should have `contribution` and `recipe` fields:

```jsonl
{"contribution": "● Deposited materials: IGZO...", "recipe": "## 0. Key Contributions..."}
```

## Evaluation Criteria (14 scores, 1-5 scale)

| Category | Criteria |
|----------|----------|
| Materials | appropriateness, completeness |
| Device Structure | completeness, similarity, feasibility |
| Deposition | parameter_completeness, parameter_accuracy, procedure_feasibility |
| Post-Processing | completeness, similarity, feasibility |
| Performance | appropriateness, similarity |
| Overall | overall_score |

## Output Files

- `{model}/prediction.jsonl` - Predictions
- `prediction_{judge}_judged.jsonl` - Judge results
- `prediction_{judge}_judged_scores.json` - Aggregated scores
