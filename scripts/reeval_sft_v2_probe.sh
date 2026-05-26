#!/bin/bash
# DSMLP launcher to RE-RUN the exp_034 SFT v2 probe gate.
#
# Why this script exists: the original launch_sft_v2.sh aborted at the probe gate
# (full-train repos were never created on HF Hub — only the probe adapter is there),
# but the pod was deleted before anyone read the gate output. We don't know which
# threshold tripped (MCQ forgetting? boxed format degradation? eval OOM?). This
# script reconstructs the gate evaluation from the probe adapter on HF Hub.
#
# It also pushes the per-question responses and a summary JSON back to HF Hub
# so the numbers survive pod deletion this time.
#
# Run from dsmlp-login:
#   cd ~/151B_SP26_Competition && git fetch origin main && git reset --hard FETCH_HEAD
#   bash scripts/reeval_sft_v2_probe.sh
# Monitor:
#   kubectl get pods ; kubectl logs -f <pod>

set -e

GPU="${GPU:-a5000}"
EXP="exp_034_sft_v2"
PROBE_REPO="TrevorDuong/qwen3-4b-pass2-sft-v2-probe"
CKPT_SUBDIR="checkpoint-final"

echo "Re-evaluating SFT v2 probe gate on $GPU ..."

# K8S_TIMEOUT_SECONDS=21600 (6h). First attempt used 7200 (2h) "just for eval" but
# HF generate on a 4B model with max_new_tokens=4096 over 200q runs ~6h on A5000.
# The 2h ceiling SIGKILLed the pod at q ~112/200 — no traceback, no kubectl event,
# just Error status. Bump to 6h cap, which is well over expected wallclock.
K8S_TIMEOUT_SECONDS=21600 launch.sh \
  -g 1 -v "$GPU" -m 32 -c 4 -B \
  -i ghcr.io/ucsd-ets/scipy-ml-notebook:stable \
  -- bash -c "
    set -e

    # --- isolate from PVC-contaminating ~/.local (CLAUDE.md pitfall #2/#3) ---
    export PYTHONNOUSERSITE=1
    rm -rf \"\$HOME/.local/lib/python3.11/site-packages\"
    rm -rf \"\$HOME/.local/bin\"

    cd \"\$HOME/151B_SP26_Competition\" && git fetch origin main && git reset --hard FETCH_HEAD

    export PIP_CACHE_DIR=/tmp/pip-cache
    export HF_HOME=/tmp/hf-cache
    mkdir -p \"\$PIP_CACHE_DIR\" \"\$HF_HOME\"

    # --- fresh venv (inference-only; no TRL/bitsandbytes needed) ---
    VENV=\"/tmp/.venv-reeval\"
    echo '--- creating fresh venv ---'
    rm -rf \"\$VENV\"
    python -m venv \"\$VENV\"
    source \"\$VENV/bin/activate\"

    pip install -q --upgrade pip
    pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124
    # Slim inference set. transformers>=4.51 for Qwen3 (CLAUDE.md pitfall #6).
    pip install -q 'numpy<2' 'transformers>=4.51,<5' 'tokenizers>=0.21' \
        peft accelerate huggingface_hub sympy 'antlr4-python3-runtime==4.11'

    echo '--- env sanity ---'
    python -c \"import torch, transformers, peft; print(f'torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__}')\"
    python -c \"import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.get_device_name(0)}')\"

    export HF_TOKEN=\$(cat \"\$HOME/.hf_token\")

    echo '======================================================'
    echo '=== STEP 1/3: download probe adapter from HF Hub ==='
    echo '======================================================'
    huggingface-cli download $PROBE_REPO --include '$CKPT_SUBDIR/*' \
        --local-dir /tmp/probe_adapter --token \"\$HF_TOKEN\"
    ls -la /tmp/probe_adapter/$CKPT_SUBDIR

    echo '======================================================'
    echo '=== STEP 2/3: run dev gate eval (~30-60 min) ==='
    echo '======================================================'
    # set +e: we need the exit code to drive the upload step regardless of pass/fail
    set +e
    python experiments/$EXP/eval_dev.py \
        --adapter_dir /tmp/probe_adapter/$CKPT_SUBDIR \
        --out /tmp/dev_probe_responses.jsonl 2>&1 | tee /tmp/eval_dev.log
    GATE_RC=\$?
    set -e

    echo '======================================================'
    echo \"=== STEP 3/3: push results to HF Hub (gate_rc=\$GATE_RC) ===\"
    echo '======================================================'
    # Build a small summary JSON from the eval_dev log + push EVERYTHING that
    # exists. Each upload is wrapped: a missing partial file (e.g. mid-crash)
    # must NOT abort the rest of the uploads.
    python -c \"
import json, os, re, traceback
from pathlib import Path
from huggingface_hub import HfApi
log_path = Path('/tmp/eval_dev.log')
log = log_path.read_text() if log_path.exists() else ''
def grab(label):
    m = re.search(label + r'\s+(\d+)/\s*(\d+)\s+([\d.]+)%', log)
    return (int(m.group(1)), int(m.group(2)), float(m.group(3))) if m else (None, None, None)
mcq_c, mcq_n, mcq_pct = grab('MCQ')
ff_c, ff_n, ff_pct = grab('Free-form')
box_c, box_n, box_pct = grab('boxed rate')
gate_pass = ('PROBE GATE: PASS' in log)
gate_fail = ('PROBE GATE: FAIL' in log)
last_infer = re.findall(r'\[infer\]\s+(\d+)/(\d+)', log)
last_infer = last_infer[-1] if last_infer else None
summary = {
    'gate_rc': $GATE_RC,
    'gate_pass': gate_pass,
    'gate_fail': gate_fail,
    'completed_eval': gate_pass or gate_fail,
    'last_progress': last_infer,
    'mcq': {'correct': mcq_c, 'total': mcq_n, 'pct': mcq_pct, 'gate_min': 60.0},
    'ff':  {'correct': ff_c,  'total': ff_n,  'pct': ff_pct,  'gate_min': 53.0},
    'boxed_extraction': {'correct': box_c, 'total': box_n, 'pct': box_pct, 'gate_min': 95.0},
    'adapter': '$PROBE_REPO/$CKPT_SUBDIR',
}
print('SUMMARY:', json.dumps(summary, indent=2))
Path('/tmp/dev_probe_summary.json').write_text(json.dumps(summary, indent=2))
api = HfApi(token=os.environ['HF_TOKEN'])
for local, remote in [
    ('/tmp/dev_probe_summary.json',  'dev_probe_summary.json'),
    ('/tmp/dev_probe_responses.jsonl','dev_probe_responses.jsonl'),
    ('/tmp/eval_dev.log',            'eval_dev.log'),
]:
    if not Path(local).exists():
        print(f'  skip (missing): {local}')
        continue
    try:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id='$PROBE_REPO', repo_type='model')
        print(f'  uploaded: {remote}')
    except Exception as e:
        traceback.print_exc()
        print(f'  upload FAILED ({remote}): {e!r} — continuing')
print('upload step done.')
\"

    echo 'DONE. Pull the summary anytime:'
    echo \"  huggingface-cli download $PROBE_REPO dev_probe_summary.json --local-dir /tmp\"
  "

echo ""
echo "Pod launched on $GPU."
echo "Find name:  kubectl get pods"
echo "Monitor:    kubectl logs -f <pod_name>"
echo ""
echo "After it finishes, the gate numbers live at:"
echo "  https://huggingface.co/TrevorDuong/qwen3-4b-pass2-sft-v2-probe/blob/main/dev_probe_summary.json"
