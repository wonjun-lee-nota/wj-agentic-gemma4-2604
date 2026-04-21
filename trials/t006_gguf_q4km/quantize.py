"""GGUF Q4_K_M conversion for Gemma-4-26B-A4B-it.

Uses llama.cpp's convert-hf-to-gguf.py to convert the HF model to GGUF format.
GGUF natively handles MoE experts as individual tensors.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

MODEL_PATH = "/workspace/gemma-4-26B-A4B-it"
TRIAL_DIR = Path(__file__).parent
WEIGHTS_DIR = TRIAL_DIR / "weights"
GGUF_F16 = WEIGHTS_DIR / "gemma4-26b-f16.gguf"
GGUF_Q4 = WEIGHTS_DIR / "gemma4-26b-q4_k_m.gguf"


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # Clone llama.cpp for conversion tools
    llama_cpp_dir = Path("/workspace/llama.cpp")
    if not llama_cpp_dir.exists():
        print("[gguf] Cloning llama.cpp...")
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp", str(llama_cpp_dir)],
            check=True,
        )

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"[gguf] ERROR: {convert_script} not found")
        sys.exit(1)

    # Install requirements for convert script
    req_file = llama_cpp_dir / "requirements" / "requirements-convert_hf_to_gguf.txt"
    if req_file.exists():
        print("[gguf] Installing conversion requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"], check=False)

    # Step 1: Convert HF to GGUF F16
    print(f"[gguf] Converting HF model to GGUF F16: {GGUF_F16}")
    result = subprocess.run(
        [sys.executable, str(convert_script), MODEL_PATH,
         "--outfile", str(GGUF_F16), "--outtype", "f16"],
        capture_output=True, text=True,
    )
    print(result.stdout[-500:] if result.stdout else "")
    if result.returncode != 0:
        print(f"[gguf] F16 conversion failed:\n{result.stderr[-1000:]}")
        sys.exit(1)
    print(f"[gguf] F16 GGUF created: {GGUF_F16} ({GGUF_F16.stat().st_size / 1e9:.1f} GB)")

    # Step 2: Quantize to Q4_K_M
    # Need llama-quantize binary — build it
    print("[gguf] Building llama-quantize...")
    build_dir = llama_cpp_dir / "build"
    build_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["cmake", "-B", str(build_dir), "-S", str(llama_cpp_dir),
         "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "llama-quantize", "-j4"],
        check=True, capture_output=True,
    )
    quantize_bin = build_dir / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        print("[gguf] ERROR: llama-quantize binary not built")
        sys.exit(1)

    print(f"[gguf] Quantizing to Q4_K_M: {GGUF_Q4}")
    result = subprocess.run(
        [str(quantize_bin), str(GGUF_F16), str(GGUF_Q4), "Q4_K_M"],
        capture_output=True, text=True,
    )
    print(result.stdout[-500:] if result.stdout else "")
    if result.returncode != 0:
        print(f"[gguf] Quantization failed:\n{result.stderr[-1000:]}")
        sys.exit(1)

    final_size = GGUF_Q4.stat().st_size / 1e9
    print(f"[gguf] Q4_K_M GGUF created: {GGUF_Q4} ({final_size:.1f} GB)")

    # Cleanup F16 intermediate
    if GGUF_F16.exists() and GGUF_Q4.exists():
        GGUF_F16.unlink()
        print(f"[gguf] Removed intermediate F16 GGUF")

    print(f"[gguf] Done!")


if __name__ == "__main__":
    main()
