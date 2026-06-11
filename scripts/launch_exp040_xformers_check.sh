#!/bin/bash
# DSMLP discriminator #3: does the xformers attention backend fix the degeneration?
# Run from the dsmlp-login node.
#
# WHY (postcomp/DEVLOG.md 2026-06-11): discriminators #1 (fp16) and #2 (V0 engine)
# both still degenerate 20-23/40 no-</think>. Disc #2 matched run_inference.py
# (0.581/0.660) on EVERY software axis — model, prompt, sampling, fp16, V0 — yet
# still collapsed. The last high-prior divergence is the ATTENTION BACKEND: the
# A5000 (sm 8.6) auto-selects Flash Attention; Kaggle T4 (sm 7.5, no FA2) falls
# back to xformers. The project already documents "DSMLP is not a Kaggle proxy"
# (24.6% vs 56.8%, same model) — this tests whether the backend is the mechanism.
#
# VLLM_ATTENTION_BACKEND=XFORMERS is read natively by vLLM at engine init
# (xformers is a hard dependency of vllm 0.8.5 cuda, already in the venv).
#
# Readout — compare "degeneration check: N/40 never closed </think>":
#   V1+bf16+FA 20/40 | V1+fp16+FA 23/40 | V0+fp16+FA 20/40 | THIS (V0+fp16+XF) ?/40
# - N drops toward ~0-4: the FA backend on A5000 was the cause -> pin XFORMERS,
#   re-run the full 4-arm probe on DSMLP for a clean PAL read; DSMLP is salvaged
#   for ALL Phase-2 inference (rollouts, difficulty sampling).
# - N stays ~50%: degeneration is fundamental to DSMLP numerics for this model ->
#   move the PAL gate to Kaggle, reserve DSMLP for training (<=4096 completions,
#   repetition penalty in reward, filter degenerate rollouts from curricula).
#
# VOID-CHECK: the engine log MUST say "Using XFormers backend." — if it still
# says "Using Flash Attention backend.", the env var was ignored and the run is void.
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
    python -c "import xformers; print(f\"xformers={xformers.__version__}\")"

    # ---- discriminator #3: baseline x40, fp16 + V0 + XFORMERS backend ----
    export PROBE_CONDITIONS=baseline
    export PROBE_DTYPE=float16
    export PROBE_V0=1
    export VLLM_ATTENTION_BACKEND=XFORMERS
    HF_TOKEN=$(cat "$HOME/.hf_token") python postcomp/experiments/exp_040_tool_reasoning/probe_run.py
    python postcomp/experiments/exp_040_tool_reasoning/probe_judge.py
  '

echo ""
echo "xformers discriminator pod launched (baseline x40, fp16 + V0 + XFORMERS)."
echo "  kubectl get pods ; kubectl logs -f <pod>"
echo "VOID-CHECK first: log must say 'Using XFormers backend.' (not Flash Attention)."
echo "Read: '[probe/gen] degeneration check: N/40 never closed </think>'"
echo "  V1+bf16+FA 20/40 | V1+fp16+FA 23/40 | V0+fp16+FA 20/40 | THIS ?/40  (target ~0-4)"
echo "  and baseline accuracy in the PROBE REPORT (FA runs 7-8/40; target -> Kaggle ~54%)."
