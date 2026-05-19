#!/bin/bash
# DSMLP launch wrapper for exp_019 GRPO pass 3 training.
# Byte-for-byte clone of launch_grpo_pass2.sh except EXP. Same recipe — same
# stack pins (torch 2.5.1, trl 0.21, peft), same venv pattern, same caches in
# /tmp to dodge the 5GB PVC quota, same A5000 GPU (only working option).
#
# Run from the dsmlp-login node.
#
# Resume: HFPushAdapterCallback uploads the full trainer state to
# adapter_checkpoints_repo every save_steps. If the 12h container is killed
# mid-training, the next pod's _try_resume_from_hf() picks up where it left off.
#
# Usage:
#   bash scripts/launch_grpo_pass3.sh
#
# After launch:
#   kubectl get pods
#   kubectl logs -f <pod_name>
#   kubectl delete pod <pod_name>

set -e

GPU="${GPU:-a5000}"
EXP="exp_019_grpo_pass3"

echo "Launching GRPO pass 3 on $GPU ..."
echo "(only a5000 is accessible on this account; a100 and a6000 are gated)"

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
    pip install -q -r experiments/$EXP/requirements.txt

    echo '--- venv env sanity ---'
    python -c \"import torch, trl, peft, bitsandbytes, transformers; print(f'torch={torch.__version__}  trl={trl.__version__}  peft={peft.__version__}  bnb={bitsandbytes.__version__}  transformers={transformers.__version__}')\"
    python -c \"import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.get_device_name(0)}')\"

    HF_TOKEN=\$(cat \"\$HOME/.hf_token\") python experiments/$EXP/train_grpo.py
  "

echo ""
echo "Pod launched on $GPU."
echo "Find name with:  kubectl get pods"
echo "Tail logs with:  kubectl logs -f <pod_name>"
echo ""
echo "Cold start ~10-15 min. Expect ~10 min/step at max_completion=4096."
echo "Curriculum is 72 prompts at batch 4 = ~18 grad updates per epoch."
