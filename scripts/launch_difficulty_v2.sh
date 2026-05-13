#!/bin/bash
# DSMLP launch wrapper for difficulty sampling v2.
# Run from the dsmlp-login node (the host that has launch.sh on PATH).
#
# What this does:
#   1. Launch a 12h batch container with A5000 GPU, 48GB mem, 8 CPU cores
#   2. Inside the container: clone latest main, install torch 2.6 + vllm 0.8,
#      run sample_difficulty_v2.py
#   3. Output (data/difficulty_samples_v2.jsonl + data/sweet_spot_v2_ids.json)
#      lives in $HOME, persists across containers.
#
# After launch:
#   kubectl get pods                          # see pod name
#   kubectl logs -f <pod_name>                # tail live
#   kubectl delete pod <pod_name>             # kill (frees GPU quota)

set -e

# --- one-time prereqs (only if not already done) ---
# Save HF token to ~/.hf_token with chmod 600:
#   echo "hf_..." > ~/.hf_token && chmod 600 ~/.hf_token

# --- launch ---
K8S_TIMEOUT_SECONDS=43200 launch.sh \
  -g 1 -v a5000 -m 48 -c 8 -B \
  -i ghcr.io/ucsd-ets/scipy-ml-notebook:stable \
  -- bash -c '
    set -e
    cd "$HOME/151B_SP26_Competition" && git pull origin main

    # Upgrade torch -> 2.6 (vllm 0.7+ requires it; needed for Qwen3 support)
    pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --extra-index-url https://download.pytorch.org/whl/cu124

    # Install vLLM + judger deps
    pip install -q vllm==0.8.5
    pip install -q sympy "antlr4-python3-runtime==4.11"

    # Run sampling (resumable: re-launch if it times out)
    HF_TOKEN=$(cat "$HOME/.hf_token") python scripts/sample_difficulty_v2.py
  '

echo ""
echo "Pod launched. Find name with: kubectl get pods"
echo "Tail logs with:               kubectl logs -f <pod_name>"
