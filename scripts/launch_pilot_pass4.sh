#!/bin/bash
# DSMLP launch wrapper for the GRPO pass-4 PILOT (scripts/pilot_grpo_pass4.py).
# Same training stack as exp_019 (torch 2.5.1 + trl 0.21 + peft + bnb), installed
# into a fresh /tmp venv. Measures dead-step rate + entropy + peak memory across
# the candidate (G, T, max_completion) recipes so we can pick pass-4's config
# from data, not a guess. NO HF push, NO checkpoint saving.
#
# Run from the dsmlp-login node:
#   bash scripts/launch_pilot_pass4.sh
#   PILOT_STEPS=4 bash scripts/launch_pilot_pass4.sh        # faster, noisier read
#   EXTRA_CONFIG="8,2560,1.0" bash scripts/launch_pilot_pass4.sh  # add a G=8 probe
#
# After launch:
#   kubectl get pods ; kubectl logs -f <pod_name> ; kubectl delete pod <pod_name>
#
# Output (PVC): data/pilot_pass4_results.json + the per-step logs in the pod log.
# Expected runtime: ~60 min/config at PILOT_STEPS=6 → ~3h for 3 configs (HF generate
# is the bottleneck). First config's first step prints in ~12-15 min (cold start + load).

set -e

GPU="${GPU:-a5000}"
PILOT_STEPS="${PILOT_STEPS:-6}"
EXTRA_CONFIG="${EXTRA_CONFIG:-}"

echo "Launching GRPO pass-4 pilot on $GPU (PILOT_STEPS=$PILOT_STEPS, EXTRA_CONFIG='$EXTRA_CONFIG') ..."

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
    export TRANSFORMERS_CACHE=/tmp/hf-cache
    mkdir -p \"\$PIP_CACHE_DIR\" \"\$HF_HOME\"

    VENV=\"/tmp/.venv-grpo-pass3\"
    echo \"--- creating fresh venv at \$VENV ---\"
    rm -rf \"\$VENV\"
    python -m venv \"\$VENV\"
    source \"\$VENV/bin/activate\"

    pip install -q --upgrade pip
    pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124
    pip install -q -r experiments/exp_019_grpo_pass3/requirements.txt

    echo '--- venv env sanity ---'
    python -c \"import torch, trl, peft, bitsandbytes, transformers; print(f'torch={torch.__version__}  trl={trl.__version__}  peft={peft.__version__}  bnb={bitsandbytes.__version__}  transformers={transformers.__version__}')\"
    python -c \"import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.get_device_name(0)}')\"

    PILOT_STEPS=$PILOT_STEPS EXTRA_CONFIG=\"$EXTRA_CONFIG\" \
    HF_TOKEN=\$(cat \"\$HOME/.hf_token\") python scripts/pilot_grpo_pass4.py
  "

echo ""
echo "Pilot pod launched on $GPU."
echo "Find name with:  kubectl get pods"
echo "Tail logs with:  kubectl logs -f <pod_name>"
echo ""
echo "Read the final 'PILOT SUMMARY' table at the end of the log, or:"
echo "  cat ~/151B_SP26_Competition/data/pilot_pass4_results.json"
