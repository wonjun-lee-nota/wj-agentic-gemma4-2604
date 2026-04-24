"""Test vLLM with FP8 quantization on 2 GPUs (tensor parallelism)."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from vllm import LLM, SamplingParams

MODEL_PATH = "/workspace/gemma-4-26B-A4B-it"

print("[test] Attempting vLLM FP8 with tensor_parallel_size=2...")
try:
    llm = LLM(
        model=MODEL_PATH,
        quantization="fp8",
        dtype="bfloat16",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        enforce_eager=True,
    )
    print("[test] FP8 TP=2 model loaded!")
    params = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = llm.generate(["What is 2+2? Answer briefly."], params)
    print(f"[test] Output: {outputs[0].outputs[0].text}")
    print("[test] SUCCESS — FP8 TP=2 works!")
except Exception as e:
    print(f"[test] FAILED: {e}")
    import traceback
    traceback.print_exc()
