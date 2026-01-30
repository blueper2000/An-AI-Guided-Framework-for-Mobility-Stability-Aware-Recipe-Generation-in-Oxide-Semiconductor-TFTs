"""
Oxide Semiconductor Synthesis Recipe Prediction

Supports two inference backends:
1. Local vLLM (EXAONE-4.0-32B) - for on-premise GPU servers
2. OpenRouter API (GPT-4o, etc.) - for cloud-based inference

Usage:
    # vLLM backend (requires 4x RTX A6000 or equivalent ~65GB VRAM)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python predict.py --backend vllm

    # OpenRouter backend
    export OPENROUTER_API_KEY=your_api_key
    python predict.py --backend openrouter --model openai/gpt-4o

    # Test with limited samples
    python predict.py --backend vllm --max_samples 10 --debug
"""

import os
import time
import json
import jsonlines
from tqdm import tqdm
from abc import ABC, abstractmethod
import fire
from typing import List, Dict, Any, Optional


# ============================================
# Configuration
# ============================================

# Default models
DEFAULT_VLLM_MODEL = "LGAI-EXAONE/EXAONE-4.0-32B"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o"

# Rate limiting for API calls
REQUEST_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 5
INITIAL_BACKOFF = 30  # seconds


# ============================================
# Base Predictor Interface
# ============================================

class BasePredictor(ABC):
    """Abstract base class for recipe predictors."""

    def __init__(self, prompt_path: str = "prompts/prediction.txt"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file = os.path.join(script_dir, prompt_path)
        with open(prompt_file, "r") as f:
            self.prediction_prompt = f.read()

    def build_prompt(self, item: Dict[str, Any]) -> str:
        """Build prompt from dataset item."""
        return self.prediction_prompt.format(contributions=item["contribution"])

    def extract_response(self, text: str) -> str:
        """Extract final response, removing <think>...</think> tags if present."""
        if text is None:
            return ""
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        return text

    @abstractmethod
    def predict_batch(self, items: List[Dict[str, Any]], debug: bool = False) -> List[str]:
        """Generate predictions for a batch of items."""
        pass


# ============================================
# vLLM Backend (Local GPU Inference)
# ============================================

class VLLMPredictor(BasePredictor):
    """Recipe predictor using vLLM for local GPU inference."""

    def __init__(
        self,
        model_name: str = DEFAULT_VLLM_MODEL,
        max_new_tokens: int = 16384,
        enable_thinking: bool = True,
        tensor_parallel_size: int = 4,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.90,
        prompt_path: str = "prompts/prediction.txt",
    ):
        super().__init__(prompt_path)

        from vllm import LLM, SamplingParams

        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.model_name = model_name

        print(f"Loading model: {model_name}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        print(f"Max model length: {max_model_len}")
        print(f"Thinking mode: {enable_thinking}")

        # Initialize vLLM
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="bfloat16",
        )

        self.tokenizer = self.llm.get_tokenizer()

        # Sampling parameters
        if enable_thinking:
            self.sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                stop=["<|endoftext|>", "<|end_of_turn|>"],
            )
        else:
            self.sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                stop=["<|endoftext|>", "<|end_of_turn|>"],
            )

        print("Model loaded successfully")

    def _build_chat_prompt(self, user_content: str) -> str:
        """Build chat-formatted prompt using tokenizer's chat template."""
        messages = [{"role": "user", "content": user_content}]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def predict_batch(self, items: List[Dict[str, Any]], debug: bool = False) -> List[str]:
        """Generate predictions for a batch of items."""
        prompts = []
        for item in items:
            user_prompt = self.build_prompt(item)
            chat_prompt = self._build_chat_prompt(user_prompt)
            prompts.append(chat_prompt)

        if debug and prompts:
            print(f"DEBUG: First prompt (truncated):\n{prompts[0][:500]}...")

        outputs = self.llm.generate(prompts, self.sampling_params)

        predictions = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            prediction = self.extract_response(generated_text)
            predictions.append(prediction)

            if debug and i == 0:
                print(f"DEBUG: Generated (truncated): {generated_text[:500]}...")

        return predictions


# ============================================
# OpenRouter Backend (Cloud API Inference)
# ============================================

class OpenRouterPredictor(BasePredictor):
    """Recipe predictor using OpenRouter API."""

    def __init__(
        self,
        model: str = DEFAULT_OPENROUTER_MODEL,
        max_new_tokens: int = 16384,
        api_key: Optional[str] = None,
        prompt_path: str = "prompts/prediction.txt",
    ):
        super().__init__(prompt_path)

        from openai import OpenAI

        self.model = model
        self.max_new_tokens = max_new_tokens

        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        print(f"Initialized OpenRouter client")
        print(f"Model: {model}")

    def _predict_single(self, prompt: str, debug: bool = False) -> str:
        """Generate prediction for a single prompt with retry logic."""
        messages = [{"role": "user", "content": prompt}]

        backoff = INITIAL_BACKOFF
        for attempt in range(MAX_RETRIES):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                )
                if debug:
                    print(f"DEBUG: completion = {completion.choices[0]}")

                response = completion.choices[0].message.content
                return self.extract_response(response)

            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = "rate limit" in error_msg or "429" in error_msg

                if is_rate_limit and attempt < MAX_RETRIES - 1:
                    print(f"Rate limit hit, waiting {backoff}s before retry {attempt + 2}/{MAX_RETRIES}...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 300)
                elif attempt < MAX_RETRIES - 1:
                    print(f"Error: {e}, retrying {attempt + 2}/{MAX_RETRIES}...")
                    time.sleep(5)
                else:
                    print(f"Failed after {MAX_RETRIES} attempts: {e}")
                    return f"[ERROR] {str(e)}"

        return "[ERROR] Max retries exceeded"

    def predict_batch(self, items: List[Dict[str, Any]], debug: bool = False) -> List[str]:
        """Generate predictions for a batch of items (sequential for API)."""
        predictions = []
        for i, item in enumerate(items):
            prompt = self.build_prompt(item)
            prediction = self._predict_single(prompt, debug=(debug and i == 0))
            predictions.append(prediction)
            time.sleep(REQUEST_DELAY)
        return predictions


# ============================================
# Main Entry Point
# ============================================

def predict(
    input_file: str,
    output_dir: str = None,
    backend: str = "vllm",
    model: str = None,
    max_new_tokens: int = 16384,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    # vLLM specific options
    enable_thinking: bool = True,
    tensor_parallel_size: int = 4,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.90,
    # OpenRouter specific options
    api_key: Optional[str] = None,
    # Debug
    debug: bool = False,
):
    """
    Run oxide semiconductor synthesis recipe prediction.

    Args:
        input_file: Path to input JSONL file (contribution, recipe fields)
        output_dir: Directory to save outputs (default: same as input file)
        backend: Inference backend - "vllm" or "openrouter"
        model: Model name (default depends on backend)
        max_new_tokens: Maximum tokens to generate
        max_samples: Maximum number of samples to process (None for all)
        batch_size: Batch size for vLLM inference
        enable_thinking: Enable thinking mode for vLLM (uses <think> tags)
        tensor_parallel_size: Number of GPUs for vLLM tensor parallelism
        max_model_len: Maximum context length for vLLM
        gpu_memory_utilization: GPU memory utilization for vLLM (0.0-1.0)
        api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
        debug: Print debug information for first sample
    """
    # Set default model based on backend
    if model is None:
        model = DEFAULT_VLLM_MODEL if backend == "vllm" else DEFAULT_OPENROUTER_MODEL

    # Load dataset
    print(f"Loading dataset from: {input_file}")
    with jsonlines.open(input_file) as reader:
        dataset = list(reader)

    # Add index as id if not present
    for idx, item in enumerate(dataset):
        if "id" not in item:
            item["id"] = idx

    if max_samples is not None:
        dataset = dataset[:min(max_samples, len(dataset))]
    print(f"Dataset size: {len(dataset)}")

    # Setup output path
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_file))

    model_short_name = model.replace("/", "-").replace(":", "-")
    output_filename = os.path.join(output_dir, model_short_name, "prediction.jsonl")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # Resume from existing progress
    skip = 0
    if os.path.exists(output_filename):
        skip = len(list(jsonlines.open(output_filename)))
        if skip >= len(dataset):
            print(f"All {skip} samples already processed")
            return output_filename
        if skip > 0:
            dataset = dataset[skip:]
            print(f"Resuming from {skip}, {len(dataset)} items remaining")

    if len(dataset) == 0:
        print("No samples to process")
        return output_filename

    # Initialize predictor based on backend
    if backend == "vllm":
        predictor = VLLMPredictor(
            model_name=model,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    elif backend == "openrouter":
        predictor = OpenRouterPredictor(
            model=model,
            max_new_tokens=max_new_tokens,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'vllm' or 'openrouter'.")

    print(f"Starting predictions, output: {output_filename}")

    # Process in batches
    with jsonlines.open(output_filename, "a") as fout:
        for i in tqdm(range(0, len(dataset), batch_size), desc="Predicting"):
            batch = dataset[i:i + batch_size]
            predictions = predictor.predict_batch(batch, debug=(debug and i == 0))

            for item, prediction in zip(batch, predictions):
                fout.write({
                    "id": item.get("id"),
                    "contribution": item["contribution"],
                    "recipe": item["recipe"],
                    "prediction": prediction,
                })

    print(f"Predictions saved to {output_filename}")
    return output_filename


if __name__ == "__main__":
    fire.Fire(predict)
