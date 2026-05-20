#!/bin/bash
# DSMLP launch wrapper for the exp_018 stage-1 variance check.
# Run from the dsmlp-login node.
#
# PURPOSE: measure run-to-run generation noise of exp_018's stage-1 inference,
# so we can tell whether exp_021's -1.7pp Kaggle move (0.628 -> 0.611) is real
# signal or sampling noise. n=NUM_SAMPLES independent decodes from the same
# prompt == NUM_SAMPLES Kaggle reruns with different seeds, but on one GPU and
# zero Kaggle slots.
#
# DEPENDENCY STACK: byte-identical to launch_difficulty_v2.sh (proven working):
# clean python -m venv, torch==2.6.0 cu124 (runs on the node's 12.2 driver via
# CUDA minor-version compat), vllm==0.8.5 (first vllm with Qwen3 support),
# sympy + antlr4 for the judger. Never touches the container conda env.
#
# TWO PHASES (set via MODE):
#   MODE=smoke (default): LIMIT=100, NUM_SAMPLES=2  -> ~15-25 min. Prints the
#                         diff-fraction: fraction of prompts whose 2 decodes
#                         disagreed on the boxed answer. Cheap go/no-go.
#   MODE=full           : LIMIT=0,  NUM_SAMPLES=3   -> ~3-5h. Full 1126-prompt
#                         3x run; prints per-sample accuracy + std + the
#                         Kaggle-scaled sigma estimate.
#
# Interpreting the smoke diff-fraction:
#   < ~8%  -> generation noise is small; exp_021's regression is likely real.
#             You may still want MODE=full for the hard sigma number.
#   > ~25% -> noise is large; exp_021 is plausibly within the band. MODE=full
#             will quantify it.
#
# After launch:
#   kubectl get pods                  # see pod name
#   kubectl logs -f <pod_name>        # tail live; grep VARIANCE_CHECK_SUMMARY
#   kubectl delete pod <pod_name>     # kill (frees GPU quota)

set -e

MODE="${MODE:-smoke}"
GPU="${GPU:-a5000}"

if [ "$MODE" = "smoke" ]; then
  RUN_LIMIT=100
  RUN_NSAMPLES=2
  RUN_OUT='$HOME/151B_SP26_Competition/data/variance_smoke.jsonl'
elif [ "$MODE" = "full" ]; then
  RUN_LIMIT=0
  RUN_NSAMPLES=3
  RUN_OUT='$HOME/151B_SP26_Competition/data/variance_full.jsonl'
else
  echo "ERROR: MODE must be 'smoke' or 'full' (got '$MODE')." >&2
  exit 1
fi

echo "Launching variance check (MODE=$MODE, LIMIT=$RUN_LIMIT, NUM_SAMPLES=$RUN_NSAMPLES) on $GPU ..."

K8S_TIMEOUT_SECONDS=43200 launch.sh \
  -g 1 -v "$GPU" -m 48 -c 8 -B \
  -i ghcr.io/ucsd-ets/scipy-ml-notebook:stable \
  -- bash -c "
    set -e

    export PYTHONNOUSERSITE=1
    rm -rf \"\$HOME/.local/lib/python3.11/site-packages\"
    rm -rf \"\$HOME/.local/bin\"

    cd \"\$HOME/151B_SP26_Competition\" && git fetch origin main && git reset --hard FETCH_HEAD

    export PIP_CACHE_DIR=/tmp/pip-cache
    export HF_HOME=/tmp/hf-cache
    mkdir -p \"\$PIP_CACHE_DIR\" \"\$HF_HOME\"

    # Reuse the difficulty-v2 venv if it already has the right vllm (saves the
    # ~10 min cold install); otherwise build it fresh. Same stack as that job.
    VENV=\"\$HOME/.venv-difficulty-v2\"
    if [ -d \"\$VENV\" ] && ! \"\$VENV/bin/pip\" freeze 2>/dev/null | grep -q '^vllm==0.8.5\$'; then
      echo '--- venv has wrong/missing vllm, rebuilding ---'
      rm -rf \"\$VENV\"
    fi
    if [ ! -d \"\$VENV\" ]; then
      echo \"--- creating venv at \$VENV ---\"
      python -m venv \"\$VENV\"
    fi
    source \"\$VENV/bin/activate\"

    pip install -q --upgrade pip
    pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    pip install -q vllm==0.8.5 sympy 'antlr4-python3-runtime==4.11' 'transformers<5.0.0'

    echo '--- venv env sanity ---'
    python -c \"import torch, vllm, transformers; print(f'torch={torch.__version__}  vllm={vllm.__version__}  transformers={transformers.__version__}')\"
    python -c \"import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.get_device_name(0)}')\"

    LIMIT=$RUN_LIMIT NUM_SAMPLES=$RUN_NSAMPLES OUT=\"$RUN_OUT\" \
      HF_TOKEN=\$(cat \"\$HOME/.hf_token\") python scripts/variance_check.py
  "

echo ""
echo "Pod launched (MODE=$MODE) on $GPU."
echo "Find name:  kubectl get pods"
echo "Tail logs:  kubectl logs -f <pod_name>"
echo "Result line to grep:  VARIANCE_CHECK_SUMMARY"
echo ""
if [ "$MODE" = "smoke" ]; then
  echo "When the smoke run finishes (~15-25 min), check the diff-fraction line."
  echo "If throughput + diff look right, relaunch the full run:"
  echo "    MODE=full bash scripts/variance_check.sh"
fi
