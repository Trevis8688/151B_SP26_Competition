#!/bin/bash
# DSMLP launch wrapper for exp_034 SFT v2 (QLoRA on qwen3-4b-thinking-grpo-pass2).
#
# One batch pod runs the whole pipeline:
#   prepare_data.py  ->  train_sft.py --phase probe  ->  eval_dev.py (GATE)
#   -> [gate PASS] train_sft.py --phase full (merges + pushes to HF)
#   -> [gate FAIL] abort before the 6-8h full train (catches exp_008-style
#      MCQ catastrophic forgetting in ~3h instead of ~2 days).
#
# Single venv: training (trl SFT) and the dev-probe eval (HF generate, NOT vLLM)
# both run in it — keeps vLLM-on-DSMLP brittleness off the safety-critical gate.
#
# Run from the dsmlp-login node:
#   cd ~/151B_SP26_Competition && git fetch origin main && git reset --hard FETCH_HEAD
#   bash scripts/launch_sft_v2.sh
#
# Resume after a 12h-cap kill: just re-run this script. The full phase's
# _try_resume_from_hf() picks up from the latest pushed full checkpoint; if the
# kill happened during probe, probe re-runs cheaply (~1h).
#
# Monitor:  kubectl get pods ; kubectl logs -f <pod> ; kubectl delete pod <pod>

set -e

GPU="${GPU:-a5000}"
EXP="exp_034_sft_v2"

echo "Launching SFT v2 on $GPU ..."

K8S_TIMEOUT_SECONDS=43200 launch.sh \
  -g 1 -v "$GPU" -m 48 -c 8 -B \
  -i ghcr.io/ucsd-ets/scipy-ml-notebook:stable \
  -- bash -c "
    set -e

    # --- isolate from the PVC-contaminating ~/.local (CLAUDE.md pitfall #2/#3) ---
    export PYTHONNOUSERSITE=1
    rm -rf \"\$HOME/.local/lib/python3.11/site-packages\"
    rm -rf \"\$HOME/.local/bin\"

    cd \"\$HOME/151B_SP26_Competition\" && git fetch origin main && git reset --hard FETCH_HEAD

    export PIP_CACHE_DIR=/tmp/pip-cache
    export HF_HOME=/tmp/hf-cache
    export TRANSFORMERS_CACHE=/tmp/hf-cache
    mkdir -p \"\$PIP_CACHE_DIR\" \"\$HF_HOME\"

    # --- fresh venv (isolated from container conda; vLLM never installed here) ---
    VENV=\"/tmp/.venv-sft-v2\"
    echo '--- creating fresh venv ---'
    rm -rf \"\$VENV\"
    python -m venv \"\$VENV\"
    source \"\$VENV/bin/activate\"

    pip install -q --upgrade pip
    pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124
    pip install -q -r experiments/$EXP/requirements.txt

    echo '--- env sanity (numpy<2 canary: scipy/sklearn must import) ---'
    python -c \"import scipy, sklearn; print('scipy/sklearn OK')\"
    python -c \"import torch, trl, peft, bitsandbytes, transformers, datasets; print(f'torch={torch.__version__} trl={trl.__version__} peft={peft.__version__} transformers={transformers.__version__}')\"
    python -c \"import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.get_device_name(0)}')\"
    python -c \"import transformers; assert tuple(int(x) for x in transformers.__version__.split('.')[:2]) >= (4,51), 'need transformers>=4.51 for Qwen3'\"

    export HF_TOKEN=\$(cat \"\$HOME/.hf_token\")

    echo '======================================================'
    echo '=== STEP 1/4: prepare data ==='
    echo '======================================================'
    python experiments/$EXP/prepare_data.py --out_dir /tmp/sft_data

    echo '======================================================'
    echo '=== STEP 2/4: train PROBE (~1h) ==='
    echo '======================================================'
    python experiments/$EXP/train_sft.py --phase probe --data_dir /tmp/sft_data --ckpt_root /tmp

    echo '======================================================'
    echo '=== STEP 3/4: dev-probe GATE (~30-60 min, HF generate) ==='
    echo '======================================================'
    set +e
    python experiments/$EXP/eval_dev.py --adapter_dir /tmp/sft_adapter-probe
    GATE_RC=\$?
    set -e
    if [ \"\$GATE_RC\" -ne 0 ]; then
      echo '======================================================'
      echo \"=== PROBE GATE FAILED (rc=\$GATE_RC). ABORTING before full train. ===\"
      echo '=== exp_008-style forgetting/format drift caught at ~3h, not ~2 days. ==='
      echo '=== Inspect experiments/$EXP/dev_probe_responses.jsonl, revise data plan. ==='
      echo '======================================================'
      exit \$GATE_RC
    fi
    echo '=== PROBE GATE PASSED — proceeding to full train ==='

    echo '======================================================'
    echo '=== STEP 4/4: train FULL (~6-8h) + merge + push to HF ==='
    echo '======================================================'
    python experiments/$EXP/train_sft.py --phase full --data_dir /tmp/sft_data --ckpt_root /tmp

    echo 'ALL DONE. Set Kaggle config.model_id to the merged repo (see config.json output_merged_repo).'
  "

echo ""
echo "Pod launched on $GPU."
echo "Find name:  kubectl get pods"
echo "Monitor:    kubectl logs -f <pod_name>"
