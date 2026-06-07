#!/bin/bash
# DSMLP launch wrapper for the exp_040 tool-reasoning PROBE (vLLM in a clean venv).
# Run from the dsmlp-login node.
#
# WHAT: 40-question dev free-form mini-eval across 4 prompt conditions
#   (baseline / fullprec_A / pal_nodemo / pal_demo). Confirms the model emits
#   runnable final code AND whether one-shot PAL net-improves dev FF accuracy.
#   Short job (~160 generations); the full report PRINTS to the log (kubectl logs).
#
# TWO PHASES (postcomp/DEVLOG.md 2026-06-07 — the v1 probe lost 32 min of GPU when
# the judge hung 6h and outputs were only written after judging):
#   phase 1  probe_run.py    GPU: generate + checkpoint -> probe_generations.jsonl
#   phase 2  probe_judge.py  CPU: sandbox + judge (per-item SIGKILL timeout) -> report
# Phase 1 checkpoints the expensive work first; phase 2 cannot lose it or hang.
# If phase 2 ever dies, re-run it alone (no GPU) on the login node:
#   ~/.venv-difficulty-v2/bin/python \
#     postcomp/experiments/exp_040_tool_reasoning/probe_judge.py
#
# VENV: reuses $HOME/.venv-difficulty-v2 (vllm==0.8.5 + sympy + antlr4 +
#   transformers<5) — exactly the probe's stack, and the 5GB PVC quota means we
#   keep only one venv. Self-heals if the pin is wrong/missing.
#
# After launch:
#   kubectl get pods                          # see pod name
#   kubectl logs -f <pod_name>                # tail live; the PROBE REPORT prints here
#   kubectl delete pod <pod_name>             # kill (frees GPU quota)
#
# Prereq (one-time): echo "hf_..." > ~/.hf_token && chmod 600 ~/.hf_token

set -e

K8S_TIMEOUT_SECONDS=21600 launch.sh \
  -g 1 -v a5000 -m 48 -c 8 -B \
  -i ghcr.io/ucsd-ets/scipy-ml-notebook:stable \
  -- bash -c '
    set -e

    # ignore + wipe any contaminated user-site (persists in PVC across pods)
    export PYTHONNOUSERSITE=1
    rm -rf "$HOME/.local/lib/python3.11/site-packages" "$HOME/.local/bin"

    # fetch + reset to the POSTCOMP branch (main stays the clean graded snapshot;
    # all post-competition code lives on origin/postcomp). Works in detached HEAD /
    # no tracking branch on the PVC workdir.
    cd "$HOME/151B_SP26_Competition" && git fetch origin postcomp && git reset --hard FETCH_HEAD

    # reuse the difficulty venv if it has the right vllm; else rebuild
    VENV="$HOME/.venv-difficulty-v2"
    if [ -d "$VENV" ] && ! "$VENV/bin/pip" freeze 2>/dev/null | grep -q "^vllm==0.8.5$"; then
      echo "--- venv has wrong/missing vllm, rebuilding ---"
      rm -rf "$VENV"
    fi
    if [ ! -d "$VENV" ]; then
      echo "--- creating venv at $VENV ---"
      python -m venv "$VENV"
      source "$VENV/bin/activate"
      pip install -q --upgrade pip
      pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
      pip install -q vllm==0.8.5 sympy "antlr4-python3-runtime==4.11" "transformers<5.0.0"
    else
      source "$VENV/bin/activate"
      # ensure judger deps present even if reusing a leaner venv
      pip install -q sympy "antlr4-python3-runtime==4.11" "transformers<5.0.0" >/dev/null 2>&1 || true
    fi

    echo "--- venv env sanity ---"
    python -c "import torch, vllm, transformers, sympy, mpmath; print(f\"torch={torch.__version__} vllm={vllm.__version__} transformers={transformers.__version__} sympy={sympy.__version__}\")"
    python -c "import torch; assert torch.cuda.is_available(); print(f\"CUDA OK: {torch.cuda.get_device_name(0)}\")"

    # ---- phase 1: generate + checkpoint (GPU) ----
    HF_TOKEN=$(cat "$HOME/.hf_token") python postcomp/experiments/exp_040_tool_reasoning/probe_run.py

    # ---- phase 2: sandbox + judge + report (CPU; fresh process, no CUDA fork) ----
    # Generations are already checkpointed, so a hang here loses nothing and the
    # per-item SIGKILL timeout means it cannot stall for 6h like the v1 probe.
    python postcomp/experiments/exp_040_tool_reasoning/probe_judge.py
  '

echo ""
echo "Probe pod launched."
echo "  kubectl get pods                 # find the name"
echo "  kubectl logs -f <pod_name>       # the PROBE REPORT + GATE verdict print here"
echo "  kubectl delete pod <pod_name>    # free the GPU when done"
echo ""
echo "Cold start ~10-15 min if the venv must be rebuilt or the model re-downloaded;"
echo "otherwise a few minutes. Outputs also saved to the experiment dir on the PVC:"
echo "  postcomp/experiments/exp_040_tool_reasoning/probe_{report.json,outputs.jsonl}"
