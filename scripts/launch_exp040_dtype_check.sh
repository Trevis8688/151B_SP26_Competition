#!/bin/bash
# DSMLP discriminator: is the ~50% generation degeneration caused by dtype=bfloat16?
# Run from the dsmlp-login node.
#
# WHY (postcomp/DEVLOG.md 2026-06-07): the exp_040 probe ran clean (no hang, no data
# loss) but ALL FOUR conditions showed ~50% of completions never closing </think> and
# collapsing into "0 0 0" repetition — a generation-quality problem, not PAL-specific,
# matching the known "DSMLP A5000+bf16 generates longer chains; 24.6% vs 56.8% same
# model" finding. Diffing the probe vs the known-good run_inference.py (0.581/0.660)
# leaves two divergences: dtype (bf16 vs the Kaggle/run_inference float16) and engine
# (V1 vs V0). dtype is the prime suspect.
#
# This runs the cheapest single-variable discriminator: baseline arm only x40 under
# PROBE_DTYPE=float16 (everything else identical to the bf16 control run). Compare the
# printed "degeneration check: N/40 never closed </think>" against the bf16 control
# (20/40). If it collapses to ~2/40 and baseline accuracy jumps toward the Kaggle ~54%,
# dtype is the cause -> flip config.json to float16. If not, next test adds V0
# (VLLM_USE_V1=0) to fully match run_inference.py.
#
# After launch:
#   kubectl get pods ; kubectl logs -f <pod>     # the degeneration line + PROBE REPORT print here
#   kubectl delete pod <pod>                      # free GPU quota

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

    # ---- discriminator: baseline x40 under float16 ----
    export PROBE_CONDITIONS=baseline
    export PROBE_DTYPE=float16
    HF_TOKEN=$(cat "$HOME/.hf_token") python postcomp/experiments/exp_040_tool_reasoning/probe_run.py
    python postcomp/experiments/exp_040_tool_reasoning/probe_judge.py
  '

echo ""
echo "Discriminator pod launched (baseline x40, float16)."
echo "  kubectl get pods ; kubectl logs -f <pod>"
echo "Read: '[probe/gen] degeneration check: N/40 never closed </think>' (control: 20/40)"
echo "  and the baseline accuracy in the PROBE REPORT (control: 7/40; target: toward Kaggle ~54%)."
