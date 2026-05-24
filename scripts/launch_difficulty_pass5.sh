#!/bin/bash
# DSMLP launch wrapper for difficulty sampling against the PASS-5 GRPO model
# (TrevorDuong/qwen3-4b-thinking-grpo-pass5), to PRE-STAGE the curriculum for GRPO PASS 6.
#
# Byte-for-byte clone of launch_difficulty_pass4.sh except MODEL_ID + OUT_SUFFIX
# (pass-4 -> pass-5). Same MATCHED-SAMPLER recipe that produced pass-4's board gain:
#   N=8, TEMPERATURE=1.0, TOP_K=-1, TOP_P=1.0 (vLLM-side equivalent of training's
#   top_k=None/top_p=1.0), MAX_NEW_TOKENS=5120 (the training budget), CHUNK_PROMPTS=12.
# See launch_difficulty_pass4.sh for the full rationale on each param.
#
# WHY RUN THIS NOW (2026-05-23): exp_029 (pass-5 stage-1-only board test) is in flight on
# Kaggle and is the gate on whether GRPO continues. Curriculum sampling is the ~4-8h long
# pole that would otherwise block a pass-6 launch. Pre-sampling here removes that latency
# from the critical path: IF exp_029 >= ~0.610, fire pass-6 training immediately. IF exp_029
# ties ~0.600, we STOP GRPO and simply discard this output (cost was only idle GPU time).
# This does NOT commit us to pass-6 — that decision still gates on the board score.
#
# Output (PVC, persists across pods):
#   data/difficulty_samples_pass5.jsonl   (one row/prompt: num_correct, num_clipped, lengths)
#   data/sweet_spot_pass5_ids.json        (default end-of-run curriculum, N=8 strict band)
#
# Then build the pass-6 curriculum from these samples WITHOUT re-running sampling:
#   python scripts/filter_curriculum_v2.py \
#     --in  data/difficulty_samples_pass5.jsonl \
#     --out experiments/exp_031_grpo_pass6/curriculum_pass6.json \
#     --min-correct 2 --max-correct 6 --allow-clipped --ff-mcq-ratio 2.0
#
# After launch:
#   kubectl get pods ; kubectl logs -f <pod_name> ; kubectl delete pod <pod_name>

set -e

K8S_TIMEOUT_SECONDS=43200 launch.sh \
  -g 1 -v a5000 -m 48 -c 8 -B \
  -i ghcr.io/ucsd-ets/scipy-ml-notebook:stable \
  -- bash -c '
    set -e

    export PYTHONNOUSERSITE=1
    rm -rf "$HOME/.local/lib/python3.11/site-packages"
    rm -rf "$HOME/.local/bin"

    export HF_HOME=/tmp/hf-cache
    export TRANSFORMERS_CACHE=/tmp/hf-cache
    export HF_HUB_CACHE=/tmp/hf-cache
    export PIP_CACHE_DIR=/tmp/pip-cache
    mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

    cd "$HOME/151B_SP26_Competition" && git fetch origin main && git reset --hard FETCH_HEAD

    VENV="$HOME/.venv-difficulty-v2"
    if [ -d "$VENV" ] && ! "$VENV/bin/pip" freeze 2>/dev/null | grep -q "^vllm==0.8.5$"; then
      echo "--- venv has wrong/missing vllm, rebuilding ---"
      rm -rf "$VENV"
    fi
    if [ ! -d "$VENV" ]; then
      echo "--- creating venv at $VENV ---"
      python -m venv "$VENV"
    fi
    source "$VENV/bin/activate"

    pip install -q --upgrade pip
    pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    pip install -q vllm==0.8.5 sympy "antlr4-python3-runtime==4.11" "transformers<5.0.0"

    echo "--- venv env sanity ---"
    python -c "import torch, vllm, transformers; print(f\"torch={torch.__version__}  vllm={vllm.__version__}  transformers={transformers.__version__}\")"
    python -c "import torch; assert torch.cuda.is_available(); print(f\"CUDA OK: {torch.cuda.get_device_name(0)}\")"

    # ---- pass-5 model, matched sampler (T=1.0, top_k disabled) at the 5120 training
    #      budget. LIMIT is env-overridable from the OUTER shell:
    #      `LIMIT=20 bash scripts/launch_difficulty_pass5.sh` smokes 20 prompts;
    #      bare `bash ...` (LIMIT=0) runs the full 1126. ----
    LIMIT='"${LIMIT:-0}"' \
    MODEL_ID="TrevorDuong/qwen3-4b-thinking-grpo-pass5" \
    OUT_SUFFIX="pass5" \
    NUM_SAMPLES=8 \
    TEMPERATURE=1.0 \
    TOP_K=-1 \
    TOP_P=1.0 \
    MAX_NEW_TOKENS=5120 \
    CHUNK_PROMPTS=12 \
    HF_TOKEN=$(cat "$HOME/.hf_token") \
      python scripts/sample_difficulty_v2.py
  '

echo ""
echo "Pod launched (FULL RUN, pass-5 model, N=8, top_k disabled)."
echo "Find name with: kubectl get pods"
echo "Tail logs with: kubectl logs -f <pod_name>"
echo ""
echo "Outputs (in PVC, persist across pods):"
echo "  ~/151B_SP26_Competition/data/difficulty_samples_pass5.jsonl"
echo "  ~/151B_SP26_Competition/data/sweet_spot_pass5_ids.json"
echo ""
echo "Expected runtime: ~4-8h. First chunk in ~10-15 min."
echo "SMOKE TEST FIRST:  LIMIT=20 bash scripts/launch_difficulty_pass5.sh"
echo ""
echo "This is PASS-6 PREP and is conditional: only build the curriculum + train pass-6"
echo "if exp_029 (pass-5 stage-1 board) comes back >= ~0.610. If it ties ~0.600, STOP GRPO"
echo "and discard data/difficulty_samples_pass5.jsonl."
echo ""
echo "Then build the pass-6 curriculum WITHOUT re-sampling:"
echo "  python scripts/filter_curriculum_v2.py \\"
echo "    --in data/difficulty_samples_pass5.jsonl \\"
echo "    --out experiments/exp_031_grpo_pass6/curriculum_pass6.json \\"
echo "    --min-correct 2 --max-correct 6 --allow-clipped --ff-mcq-ratio 2.0"
