#!/bin/bash
set -e
REPO=/workspace/wj-agentic-gemma4-2604
LLAMA_SERVER=/workspace/llama.cpp/build/bin/llama-server
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

run_trial() {
    local trial=$1
    local model_path=$2
    echo "============================================"
    echo "[run] Starting $trial — $(date)"
    echo "============================================"
    fuser -k 8080/tcp 2>/dev/null || true; sleep 3
    CUDA_VISIBLE_DEVICES=0,1,2,3 $LLAMA_SERVER \
      --model "$model_path" --port 8080 --n-gpu-layers 999 \
      --ctx-size 72000 --host 127.0.0.1 > /tmp/llama-server.log 2>&1 &
    for i in $(seq 1 120); do
        curl -s http://127.0.0.1:8080/health 2>/dev/null | grep -q "ok" && break
        sleep 3
    done
    echo "[run] Server ready, running eval..."
    cd $REPO && python3 "trials/$trial/run_aime_eval.py"
    echo "[run] $trial DONE — $(date)"
    echo ""
}

run_trial "t010_gguf_imatrix_attnQ8" \
    "$REPO/trials/t010_gguf_imatrix_attnQ8/weights/model.gguf"

run_trial "t011_gguf_imatrix_attnQ8_expQ5" \
    "$REPO/trials/t011_gguf_imatrix_attnQ8_expQ5/weights/model.gguf"

run_trial "t012_gguf_imatrix_attnQ8_edgeQ5" \
    "$REPO/trials/t012_gguf_imatrix_attnQ8_edgeQ5/weights/model.gguf"

fuser -k 8080/tcp 2>/dev/null || true
echo "============================================"
echo "[run] ALL 3 EVALS COMPLETE — $(date)"
echo "============================================"
