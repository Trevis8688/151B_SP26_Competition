#!/bin/bash
# DSMLP discriminator #2: does matching run_inference.py's V0 engine fix the degeneration?
# Run from the dsmlp-login node.
#
# WHY (postcomp/DEVLOG.md 2026-06-07/08): the exp_040 probe showed ~50% of completions
# never close </think> and collapse into "0 0 0" repetition — across ALL prompt arms, so
# it's the generation environment, not PAL. Discriminator #1 ruled OUT dtype (float16
# degenerates identically: 23/40 no-</think> vs the bf16 control 20/40). The ONLY remaining
# divergence from the known-good run_inference.py (0.581/0.660) is the engine: it runs V0
# (VLLM_USE_V1=0); the probe defaulted to V1. The V1 engine log even shows it falling back
# to a PyTorch-native top-p/top-k sampler (no FlashInfer) — a plausible cause of long-gen
# repetition collapse.
#
# This matches run_inference.py EXACTLY for the baseline arm: float16 + V0 + baseline x40.
# Compare the printed "degeneration check: N/40 never closed </think>":
#   control V1+bf16  20/40  |  V1+fp16  23/40  |  THIS (V0+fp16)  ?/40
# If V0 collapses no-</think> to ~0 and lifts accuracy toward the Kaggle ~54%, the V1
# sampling path was the cause -> set VLLM_USE_V1=0 in the probe + re-run the full 4-arm
# probe for a clean PAL read. If V0 ALSO degenerates ~50%, the degeneration is fundamental
# to DSMLP for this model -> evaluate on Kaggle (known-good fp16+V0+T4), reserve DSMLP for
# training, and design Phase-2 GRPO to tolerate/curb the repetition.
#
# After launch:
#   kubectl get pods ; kubectl logs -f <pod>
#   kubectl delete pod <pod>

set -e

K8S_TIMEOUT_SECONDS=10800 launch.sh \
  -g 1 -v a5000 -m 48 -c 8 -B \
  -i ghcr.io/ucsd-ets/scipy-ml-notebook:stable \
  -- bash -c '
    set -e
    export PYTHONNOUSERSITE=1
    rm -rf "$HOME/.local/lib/python3.11/site-packages" "$HOME/.local/bin"

    cd "$HOME/151B_SP26_Competition" && git fetch origin postcomp && git reset --hard FETCH_HEAD

    VENV="$HOME/.venv-difficulty-v2"
    if [ -d "$VENV" ] && ! "$VENV/bin/pip" freeze 2>/dev/null | grep -q "^vllm==0.8.5$"; then
      echo "--- venv has wrong/missing vllm, rebuilding ---"; rm -rf "$VENV"
    fi
    if [ ! -d "$VENV" ]; then
      echo "--- creating venv at $VENV ---"
      python -m venv "$VENV"; source "$VENV/bin/activate"
      pip install -q --upgrade pip
      pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
      pip install -q vllm==0.8.5 sympy "antlr4-python3-runtime==4.11" "transformers<5.0.0"
    else
      source "$VENV/bin/activate"
      pip install -q sympy "antlr4-python3-runtime==4.11" "transformers<5.0.0" >/dev/null 2>&1 || true
    fi
    python -c "import torch, vllm; print(f\"torch={torch.__version__} vllm={vllm.__version__}\")"

    # ---- discriminator #2: baseline x40, float16 + V0 engine (full run_inference.py match) ----
    export PROBE_CONDITIONS=baseline
    export PROBE_DTYPE=float16
    export PROBE_V0=1
    HF_TOKEN=$(cat "$HOME/.hf_token") python postcomp/experiments/exp_040_tool_reasoning/probe_run.py
    python postcomp/experiments/exp_040_tool_reasoning/probe_judge.py
  '

echo ""
echo "V0 discriminator pod launched (baseline x40, float16 + V0 engine)."
echo "  kubectl get pods ; kubectl logs -f <pod>"
echo "Read: '[probe/gen] degeneration check: N/40 never closed </think>'"
echo "  control V1+bf16 20/40 | V1+fp16 23/40 | THIS V0+fp16 ?/40   (target: ~0)"
echo "  and baseline accuracy in the PROBE REPORT (control 7-8/40; target -> Kaggle ~54%)."
echo "Note: the engine log should now say 'V0 LLM engine', not 'V1 LLM engine'."
