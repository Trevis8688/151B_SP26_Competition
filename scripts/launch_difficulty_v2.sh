#!/bin/bash
# DSMLP launch wrapper for difficulty sampling v2 (HF generate version).
# Run from the dsmlp-login node.
#
# Uses the scipy-ml-notebook:stable container as-is (torch 2.5, transformers,
# numpy 1.x). Does NOT install vllm or upgrade torch — that path triggers a
# numpy 2.x ABI conflict with the container's pre-compiled scipy/sklearn.
# Slower than vllm (~2-3h vs ~5h) but reliable.
#
# Workflow:
#   1. Launch a 12h batch container with A5000 GPU, 48GB mem, 8 CPU cores
#   2. Inside the container: pip install only judger deps, run script
#   3. Outputs (data/difficulty_samples_v2.jsonl + data/sweet_spot_v2_ids.json)
#      live in $HOME, persist across containers
#
# After launch:
#   kubectl get pods                          # see pod name
#   kubectl logs -f <pod_name>                # tail live
#   kubectl delete pod <pod_name>             # kill (frees GPU quota)

set -e

# --- one-time prereqs ---
# Save HF token to ~/.hf_token with chmod 600:
#   echo "hf_..." > ~/.hf_token && chmod 600 ~/.hf_token

# --- launch ---
K8S_TIMEOUT_SECONDS=43200 launch.sh \
  -g 1 -v a5000 -m 48 -c 8 -B \
  -i ghcr.io/ucsd-ets/scipy-ml-notebook:stable \
  -- bash -c '
    set -e

    # ---- IMPORTANT: wipe contaminated user-site ----
    # A previous launch installed torch 2.6 + numpy 2.x + transformers into
    # ~/.local/lib/python3.11/site-packages/ via `pip install --user`. Those
    # files persist in the PVC across pods, and Python prefers them over
    # /opt/conda. They break the container scipy/sklearn (numpy 1.x ABI).
    # Wipe before doing anything so the pristine container env is used.
    rm -rf "$HOME/.local/lib/python3.11/site-packages"
    rm -rf "$HOME/.local/bin"

    cd "$HOME/151B_SP26_Competition" && git pull origin main

    # Re-install only judger deps. These are pure-Python so the user-site is fine.
    pip install -q --user sympy "antlr4-python3-runtime==4.11"

    echo "--- Python env sanity ---"
    python -c "import torch, numpy, transformers; print(f\"torch={torch.__version__}  numpy={numpy.__version__}  transformers={transformers.__version__}\")"

    # Resumable: if the previous pod wrote partial output, this picks up where it left off.
    HF_TOKEN=$(cat "$HOME/.hf_token") python scripts/sample_difficulty_v2.py
  '

echo ""
echo "Pod launched. Find name with: kubectl get pods"
echo "Tail logs with:               kubectl logs -f <pod_name>"
echo "Kill if stuck:                kubectl delete pod <pod_name>"
