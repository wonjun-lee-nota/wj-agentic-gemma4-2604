"""Test loading GPTQ model via vLLM with moe_wna16 quantization.

This tests whether vLLM can load the t001 GPTQ model and handle
the unquantized MoE experts via moe_wna16 kernel.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from vllm import LLM, SamplingParams

MODEL_PATH = "/workspace/wj-agentic-gemma4-2604/trials/t001_gptq_w4_g128/weights"

print("[test] Loading model via vLLM with moe_wna16...")
try:
    llm = LLM(
        model=MODEL_PATH,
        quantization="moe_wna16",
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        enforce_eager=True,
    )
    print("[test] Model loaded successfully!")

    # Quick generation test
    params = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = llm.generate(["What is 2+2?"], params)
    print(f"[test] Output: {outputs[0].outputs[0].text}")
    print("[test] SUCCESS — moe_wna16 works!")

except Exception as e:
    print(f"[test] FAILED: {e}")
    import traceback
    traceback.print_exc()
