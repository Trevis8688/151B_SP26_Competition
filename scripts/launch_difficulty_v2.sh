#!/bin/bash
# DSMLP launch wrapper for difficulty sampling v2 (vLLM in a clean venv).
# Run from the dsmlp-login node.
#
# WHY a clean venv: `pip install vllm` into the container's shared conda env pulls
# torch 2.6 + numpy 2.x, which breaks the pre-compiled scipy/sklearn (numpy 1.x
# ABI). A first attempt used HF `generate` to dodge that -- but HF generate has no
# continuous batching and the full 4464 long thinking-model generations clocked in
# at ~93h. The fix: run vLLM inside an isolated `python -m venv`. vLLM brings its
# own torch/numpy into the venv and never touches the container conda env. This
# job needs nothing from that env (judger uses sympy + antlr4, both pure Python).
#
# Workflow:
#   1. Launch a 12h batch container with A5000 GPU, 48GB mem, 8 CPU cores
#   2. Inside: wipe contaminated user-site, build/reuse a venv, pip install vllm
#   3. SMOKE TEST first (LIMIT=10) -- eyeball throughput, ~2 min expected
#   4. If throughput looks right, relaunch without LIMIT for the full ~2-4h run
#   5. Outputs land in $HOME (PVC, persists across pods); script is resumable
#
# After launch:
#   kubectl get pods                          # see pod name
#   kubectl logs -f <pod_name>                # tail live
#   kubectl delete pod <pod_name>             # kill (frees GPU quota)
#
# NOTE: this script runs the SMOKE TEST by default. Once throughput is confirmed,
# remove `LIMIT=10` from the python invocation below and relaunch for the full job.

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

    # ---- ignore + wipe contaminated user-site ----
    # A previous launch did `pip install --user` of torch 2.6 + numpy 2.x into
    # ~/.local/lib/python3.11/site-packages/. That dir persists in the PVC across
    # pods. PYTHONNOUSERSITE=1 makes both the system python and the venv ignore
    # it; the rm is belt-and-suspenders hygiene.
    export PYTHONNOUSERSITE=1
    rm -rf "$HOME/.local/lib/python3.11/site-packages"
    rm -rf "$HOME/.local/bin"

    cd "$HOME/151B_SP26_Competition" && git pull origin main

    # ---- clean venv (isolated from the container conda env) ----
    # Self-healing: if the venv exists but has the wrong vllm, rebuild it. The
    # first attempt installed unpinned vllm (0.20.x) which pulled torch 2.11 +
    # cu130 and could not init CUDA on this node.
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

    # The DSMLP node driver supports only CUDA 12.2 (reports 12020). Unpinned
    # `pip install vllm` grabs the newest vllm, which pulls torch 2.11 + cu130
    # (CUDA 13.x) -- a major-version jump that needs a 13.x driver and fails
    # with "NVIDIA driver too old". Pin vllm==0.8.5 (first vllm with Qwen3 arch
    # support; requires torch==2.6.0) and install torch explicitly from the
    # cu124 index: CUDA 12.4 binaries run on the 12.2 driver via CUDA
    # minor-version compatibility (12.x -> 12.x is supported; 13.x is not).
    pip install -q --upgrade pip
    pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    pip install -q vllm==0.8.5 sympy "antlr4-python3-runtime==4.11"

    echo "--- venv env sanity ---"
    python -c "import torch, vllm, transformers; print(f\"torch={torch.__version__}  vllm={vllm.__version__}  transformers={transformers.__version__}\")"
    # Fail fast (seconds) if CUDA still cannot init, before the model download.
    python -c "import torch; assert torch.cuda.is_available(); print(f\"CUDA OK: {torch.cuda.get_device_name(0)}\")"

    # ---- SMOKE TEST (LIMIT=10). Remove LIMIT=10 once throughput is confirmed. ----
    LIMIT=10 HF_TOKEN=$(cat "$HOME/.hf_token") python scripts/sample_difficulty_v2.py
  '

echo ""
echo "Pod launched (SMOKE TEST mode, LIMIT=10)."
echo "Find name with: kubectl get pods"
echo "Tail logs with: kubectl logs -f <pod_name>"
echo ""
echo "Cold start is ~10-15 min (vllm wheels + model download) on first run."
echo "Once you see the per-chunk timing line, check it: a 48-prompt chunk"
echo "should finish in well under 10 min. If it does, remove 'LIMIT=10' from"
echo "this script and relaunch for the full ~2-4h job."
