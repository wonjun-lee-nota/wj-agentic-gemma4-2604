"""Test loading BF16 model via vLLM with various quantization options."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams

MODEL_PATH = "/workspace/gemma-4-26B-A4B-it"

# Try fp8 first — dynamic FP8 quantization at load time
# FP8 halves the model size: 48GB → ~24GB, might just fit on 1 GPU
print("[test] Attempting vLLM FP8 load on single GPU...")
try:
    llm = LLM(
        model=MODEL_PATH,
        quantization="fp8",
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        max_model_len=512,  # minimal context to save memory
        enforce_eager=True,
    )
    print("[test] FP8 model loaded!")
    params = SamplingParams(temperature=0.0, max_tokens=16)
    outputs = llm.generate(["What is 2+2?"], params)
    print(f"[test] Output: {outputs[0].outputs[0].text}")
    print("[test] SUCCESS — FP8 on single GPU works!")
except Exception as e:
    print(f"[test] FP8 single GPU FAILED: {e}")

    # Try with 2 GPUs if single fails
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
