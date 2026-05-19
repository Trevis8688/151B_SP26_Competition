#!/bin/bash
# DSMLP launch wrapper for difficulty sampling against the EXP_015 PASS-2 GRPO model.
# Same infra/dependencies as launch_difficulty_v2.sh -- only the model id and output
# filenames differ. Outputs are written under data/difficulty_samples_pass2.jsonl and
# data/sweet_spot_pass2_ids.json so they cannot clash with the original v2 outputs.
#
# Use case: after exp_015 GRPO pass 2 finishes, the v2 curriculum is again stale
# (sampled from exp_009 policy, but the policy has moved). This regenerates the
# curriculum from the new pass-2 policy so we can launch GRPO pass 3 immediately
# if exp_017 confirms pass-2 is an improvement.
#
# After launch:
#   kubectl get pods                          # see pod name
#   kubectl logs -f <pod_name>                # tail live
#   kubectl delete pod <pod_name>             # kill (frees GPU quota)

set -e

K8S_TIMEOUT_SECONDS=43200 launch.sh \
  -g 1 -v a5000 -m 48 -c 8 -B \
  -i ghcr.io/ucsd-ets/scipy-ml-notebook:stable \
  -- bash -c '
    set -e

    export PYTHONNOUSERSITE=1
    rm -rf "$HOME/.local/lib/python3.11/site-packages"
    rm -rf "$HOME/.local/bin"

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

    # ---- FULL RUN against the pass-2 model ----
    # MODEL_ID + OUT_SUFFIX are read by scripts/sample_difficulty_v2.py.
    MODEL_ID="TrevorDuong/qwen3-4b-thinking-grpo-pass2" \
    OUT_SUFFIX="pass2" \
    HF_TOKEN=$(cat "$HOME/.hf_token") \
      python scripts/sample_difficulty_v2.py
  '

echo ""
echo "Pod launched (FULL RUN, pass-2 model)."
echo "Find name with: kubectl get pods"
echo "Tail logs with: kubectl logs -f <pod_name>"
echo ""
echo "Outputs (in PVC, persist across pods):"
echo "  ~/151B_SP26_Competition/data/difficulty_samples_pass2.jsonl"
echo "  ~/151B_SP26_Competition/data/sweet_spot_pass2_ids.json"
echo ""
echo "Expected runtime: ~2-4h. First chunk should print in ~10-15 min."
