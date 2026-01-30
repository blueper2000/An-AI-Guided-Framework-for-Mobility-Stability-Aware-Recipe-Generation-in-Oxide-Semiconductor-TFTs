"""
LLM-as-a-Judge Evaluation for Oxide Semiconductor Synthesis Recipes

Evaluates AI-generated recipes against ground truth using OpenRouter API.
Supports 14 evaluation criteria on a 1-5 scale.

Usage:
    export OPENROUTER_API_KEY=your_api_key

    # Basic evaluation
    python judge.py predictions.jsonl

    # With specific judge model
    python judge.py predictions.jsonl --model openai/gpt-4o-mini

    # Test with limited samples
    python judge.py predictions.jsonl --max_samples 10
"""

import os
import json
import time
import jsonlines
from tqdm import tqdm
from typing import Optional
import fire


# ============================================
# Configuration
# ============================================

DEFAULT_MODEL = "openai/gpt-4o"
REQUEST_DELAY = 0.5
MAX_RETRIES = 5
INITIAL_BACKOFF = 30


# ============================================
# User Prompt Template
# ============================================

USER_PROMPT = """Please evaluate the following:

## Key Contributions
{contribution}

# AI-Generated Recipe
{prediction}

# Ground Truth Recipe
{recipe}"""


# ============================================
# Recipe Judge
# ============================================

class RecipeJudge:
    """Judge for evaluating recipe predictions using OpenRouter API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        prompt_path: str = "prompts/judge.txt",
        api_key: Optional[str] = None,
    ):
        from litellm import completion
        self._completion = completion

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Load system prompt
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file = os.path.join(script_dir, prompt_path)
        with open(prompt_file, "r") as f:
            self.system_prompt = f.read()

        # Get API key
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        os.environ["OPENROUTER_API_KEY"] = api_key

        print(f"Using judge model: {model}")

    def judge(self, item: dict) -> str:
        """Evaluate a single prediction with retry logic."""
        model = self.model
        if not model.startswith("openrouter/"):
            model = f"openrouter/{model}"

        user_content = USER_PROMPT.format(
            contribution=item["contribution"],
            prediction=item["prediction"],
            recipe=item["recipe"],
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        backoff = INITIAL_BACKOFF
        for attempt in range(MAX_RETRIES):
            try:
                response = self._completion(
                    model=model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response["choices"][0]["message"]["content"]

            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = "rate limit" in error_msg or "429" in error_msg

                if is_rate_limit and attempt < MAX_RETRIES - 1:
                    print(f"Rate limit hit, waiting {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 300)
                elif attempt < MAX_RETRIES - 1:
                    print(f"Error: {e}, retrying...")
                    time.sleep(5)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")


# ============================================
# Main Entry Point
# ============================================

def main(
    input_file: str,
    model: str = DEFAULT_MODEL,
    prompt_path: str = "prompts/judge.txt",
    temperature: float = 0.0,
    max_tokens: int = 8192,
    max_samples: Optional[int] = None,
    api_key: Optional[str] = None,
):
    """
    Run LLM-as-a-Judge evaluation on recipe predictions.

    Args:
        input_file: Input JSONL file with predictions (id, contribution, recipe, prediction)
        model: OpenRouter model name for evaluation
        prompt_path: Path to judge prompt file
        temperature: Sampling temperature (0.0 for deterministic)
        max_tokens: Maximum output tokens
        max_samples: Maximum number of samples to evaluate (for testing)
        api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
    """
    # Load dataset
    ds = list(jsonlines.open(input_file))
    total_samples = len(ds)

    if max_samples is not None:
        ds = ds[:max_samples]
        print(f"Limiting to {max_samples} samples (out of {total_samples})")
    else:
        print(f"Total samples: {total_samples}")

    # Initialize judge
    judge = RecipeJudge(
        model=model,
        prompt_path=prompt_path,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )

    # Output filename
    model_name = model.split("/")[-1].replace(":", "-")
    output_filename = input_file.replace(".jsonl", f"_{model_name}_judged.jsonl")

    # Resume from existing progress
    skip = 0
    if os.path.exists(output_filename):
        with open(output_filename, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        skip += 1
                    except json.JSONDecodeError:
                        pass
        ds = ds[skip:]
        print(f"Resuming: skipping {skip} already processed, {len(ds)} remaining")

    print(f"Output file: {output_filename}")
    print("-" * 60)

    # Evaluate
    with jsonlines.open(output_filename, "a") as fout:
        for item in tqdm(ds, desc="Evaluating"):
            try:
                judge_result = judge.judge(item)

                item["judge_result"] = judge_result
                item["judge_model"] = model
                fout.write(item)

                # Brief preview
                print(f"\n[ID: {item.get('id', 'N/A')}]")
                preview = judge_result[:300] + "..." if len(judge_result) > 300 else judge_result
                print(preview)
                print("-" * 40)

                time.sleep(REQUEST_DELAY)

            except Exception as e:
                print(f"Error processing item {item.get('id', 'N/A')}: {e}")
                continue

    print(f"\nEvaluation complete! Results saved to: {output_filename}")


if __name__ == "__main__":
    fire.Fire(main)
