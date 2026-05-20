#!/bin/bash
# DSMLP launch wrapper for difficulty sampling against the PASS-3 GRPO model
# (TrevorDuong/qwen3-4b-thinking-grpo-pass3), to build the curriculum for GRPO PASS 4.
# Clone of launch_difficulty_pass2.sh with two deliberate curriculum-QUALITY changes
# aimed at the "only ~10% of steps produced useful gradient" problem:
#
#   1. SAMPLER MATCHED TO TRAINING. The old runs sampled difficulty at top_k=20 /
#      top_p=0.95 but GRPO training samples at temperature=1.0 with truncation
#      effectively DISABLED (pass-4 plan: set GRPOConfig top_k=0). A difficulty band
#      measured under a different per-token distribution doesn't predict training-time
#      reward variance -> dead steps. So we sample here with top_k DISABLED too.
#      NOTE backend convention: vLLM disables top_k with TOP_K=-1 and top_p with
#      TOP_P=1.0; that is the vLLM-side equivalent of TRL/transformers top_k=0.
#
#   2. NUM_SAMPLES 4 -> 8. Finer p_correct resolution so the [min,max]-correct band
#      better isolates prompts whose true p_correct is near 0.5 (max reward variance
#      for any training group size). Caveat: an 8-sample p_correct estimate still has
#      ~+/-18% noise, so the band edges are porous -- do not oversell precision.
#      (N=16 would sharpen but risks the 12h timeout; N=8 is the right tradeoff.)
#
# CHUNK_PROMPTS dropped 24 -> 12 to keep CHUNK_PROMPTS*NUM_SAMPLES ~= 96 seqs/call
# (the A5000 OOM'd at higher concurrency on the logit path).
#
# >>> GATE: run this ONLY after the top_k measurement on DSMLP:
#       python -c "from trl import GRPOConfig; c=GRPOConfig(output_dir='/tmp/x'); print(c.top_k, c.top_p)"
#     - top_k=20/50  -> truncation hypothesis confirmed; this matched re-sample is the
#                       primary curriculum fix. Launch as-is.
#     - top_k=None/0 -> training already disables top_k; low entropy is BnB peakedness.
#                       The matched re-sample still helps (old samples used top_k=20),
#                       but the PRIMARY fix becomes the num_generations bump (pilot).
#                       Still fine to launch this; just prioritize the pilot.
#
# Output (PVC, persists across pods):
#   data/difficulty_samples_pass3.jsonl   (one row/prompt: num_correct, num_clipped, per-sample length)
#   data/sweet_spot_pass3_ids.json        (default end-of-run curriculum, N=8 strict band)
#
# Then build the pass-4 curriculum from these samples WITHOUT re-running sampling:
#   python scripts/filter_curriculum_v2.py \
#     --in  data/difficulty_samples_pass3.jsonl \
#     --out experiments/exp_022_<slug>/curriculum_pass4.json \
#     --min-correct 2 --max-correct 6 \      # p_correct in [0.25,0.75] at N=8
#     --max-length <pilot max_completion> \  # post-hoc clip at the training budget
#     --ff-mcq-ratio 2.0
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

    # ---- FULL RUN against the pass-3 model, sampler matched to a top_k-disabled
    #      T=1.0 training. Smoke test first by prepending LIMIT=10. ----
    MODEL_ID="TrevorDuong/qwen3-4b-thinking-grpo-pass3" \
    OUT_SUFFIX="pass3" \
    NUM_SAMPLES=8 \
    TEMPERATURE=1.0 \
    TOP_K=-1 \
    TOP_P=1.0 \
    MAX_NEW_TOKENS=6144 \
    CHUNK_PROMPTS=12 \
    HF_TOKEN=$(cat "$HOME/.hf_token") \
      python scripts/sample_difficulty_v2.py
  '

echo ""
echo "Pod launched (FULL RUN, pass-3 model, N=8, top_k disabled)."
echo "Find name with: kubectl get pods"
echo "Tail logs with: kubectl logs -f <pod_name>"
echo ""
echo "Outputs (in PVC, persist across pods):"
echo "  ~/151B_SP26_Competition/data/difficulty_samples_pass3.jsonl"
echo "  ~/151B_SP26_Competition/data/sweet_spot_pass3_ids.json"
echo ""
echo "Expected runtime: ~4-8h (N=8 is 2x the pass-2 N=4 run). First chunk in ~10-15 min."
echo "SMOKE TEST FIRST: prepend LIMIT=10 to the python line and confirm throughput."
