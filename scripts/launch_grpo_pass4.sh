#!/bin/bash
# DSMLP launch wrapper for exp_022 GRPO pass 4 training.
# Byte-for-byte clone of launch_grpo_pass3.sh except EXP. Same recipe (G=4, T=1.0,
# max_completion=5120). The intentional pass-4 change is the CURRICULUM, not the
# training stack — see experiments/exp_022_grpo_pass4/config.json _comment.
#
# Prereq: experiments/exp_022_grpo_pass4/curriculum_pass4.json must exist. Build
# it from data/difficulty_samples_pass3.jsonl with filter_curriculum_v2.py — see
# the command at the top of launch_difficulty_pass3.sh.
#
# Run from the dsmlp-login node:
#   bash scripts/launch_grpo_pass4.sh
#
# Resume: HFPushAdapterCallback uploads full trainer state to the
# adapter_checkpoints_repo every save_steps. If the 12h container is killed,
# the next pod's _try_resume_from_hf() picks up where it left off.
#
# After launch:
#   kubectl get pods
#   kubectl logs -f <pod_name>
#   kubectl delete pod <pod_name>

set -e

GPU="${GPU:-a5000}"
EXP="exp_022_grpo_pass4"

# Refuse to launch if the curriculum hasn't been built yet (saves a 10-min cold
# start that would just crash in train_grpo.py's curriculum-existence check).
if [ ! -f "experiments/$EXP/curriculum_pass4.json" ]; then
  echo "ERROR: experiments/$EXP/curriculum_pass4.json missing." >&2
  echo "Build it first: " >&2
  echo "  python scripts/filter_curriculum_v2.py \\" >&2
  echo "    --in data/difficulty_samples_pass3.jsonl \\" >&2
  echo "    --out experiments/$EXP/curriculum_pass4.json \\" >&2
  echo "    --min-correct 2 --max-correct 6 --allow-clipped --ff-mcq-ratio 2.0" >&2
  exit 1
fi

echo "Launching GRPO pass 4 on $GPU ..."

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

    VENV=\"/tmp/.venv-grpo-pass4\"
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
echo "Expected: 60-100 prompt curriculum -> ~15-25 grad updates per epoch."
echo "Inline merge writes to /tmp (bug-082 fix); if push fails for any reason,"
echo "use experiments/$EXP/merge_and_push.ipynb on Kaggle as the recovery path."
