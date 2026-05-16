#!/bin/bash
# DSMLP launch wrapper for exp_015 GRPO pass 2 training.
# Run from the dsmlp-login node.
#
# Stack: torch 2.5.1 + bitsandbytes + trl 0.21 + peft (matches exp_009 known-good).
# Isolated in a clean python venv so a contaminated user-site / wrong-numpy install
# from a prior pod can't break it. See CLAUDE.md DSMLP pitfalls 2-3.
#
# GPU: A5000 (24GB, sm 8.6, bf16 + FA2). max_completion_length=4096 clears
# the exp_010 OOM on this card.
# NOTE: this user does NOT have access to A100 (gputype=a100 nodes carry the
# `gpu-class: large` taint which our pods can't tolerate -- 2026-05-15 ticket
# observed FailedScheduling for 10+ min). A6000 is rejected by launch.sh
# outright. A5000 (gpu-class: medium) is the only path for this account.
#
# Resume: HFPushAdapterCallback uploads the full trainer state to
# adapter_checkpoints_repo every save_steps. If the 12h container is killed
# mid-training, the next pod's _try_resume_from_hf() picks up where it left off.
#
# Usage:
#   bash scripts/launch_grpo_pass2.sh           # A5000 (only working option)
#
# After launch:
#   kubectl get pods
#   kubectl logs -f <pod_name>
#   kubectl delete pod <pod_name>

set -e

GPU="${GPU:-a5000}"
EXP="exp_015_grpo_pass2"

echo "Launching GRPO pass 2 on $GPU ..."
echo "(only a5000 is accessible on this account; a100 and a6000 are gated)"

K8S_TIMEOUT_SECONDS=43200 launch.sh \
  -g 1 -v "$GPU" -m 48 -c 8 -B \
  -i ghcr.io/ucsd-ets/scipy-ml-notebook:stable \
  -- bash -c "
    set -e

    # Ignore + wipe the user-site that prior pods may have polluted.
    # PYTHONNOUSERSITE=1 makes both system python and the venv ignore ~/.local;
    # the rm is belt-and-suspenders.
    export PYTHONNOUSERSITE=1
    rm -rf \"\$HOME/.local/lib/python3.11/site-packages\"
    rm -rf \"\$HOME/.local/bin\"

    # fetch + reset is robust to detached-HEAD / no-tracking-branch PVCs.
    cd \"\$HOME/151B_SP26_Competition\" && git fetch origin main && git reset --hard FETCH_HEAD

    # Clean venv, isolated from the container conda env.
    # Self-healing: rebuild if torch version drifts.
    VENV=\"\$HOME/.venv-grpo-pass2\"
    if [ -d \"\$VENV\" ] && ! \"\$VENV/bin/pip\" freeze 2>/dev/null | grep -q '^torch==2.5.1\$'; then
      echo '--- venv has wrong/missing torch, rebuilding ---'
      rm -rf \"\$VENV\"
    fi
    if [ ! -d \"\$VENV\" ]; then
      echo \"--- creating venv at \$VENV ---\"
      python -m venv \"\$VENV\"
    fi
    source \"\$VENV/bin/activate\"

    pip install -q --upgrade pip
    # Pre-install torch from the cu124 index to match the DSMLP CUDA 12.2 driver
    # (CUDA 12.x minor-compat). Then the rest from requirements.txt.
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
echo "Cold start ~10-15 min (torch + bnb + trl wheels + model download)."
echo "Expect ~10 min/step at max_completion=4096; ~70 steps total per epoch."
