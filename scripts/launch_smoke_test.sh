#!/bin/bash
# DSMLP smoke test for run_inference.py — validates the GPU path end-to-end
# (the only part not testable on a laptop). Run from the dsmlp-login node.
#
# WHAT IT PROVES:
#   1. stage-1 vLLM (qwen3-4b-thinking-grpo-pass2) loads + generates, subprocess exits
#   2. its VRAM is reclaimed, then stage-2 vLLM (qwen3-4b-thinking-grpo-strict70)
#      loads in a fresh subprocess WITHOUT OOM  <-- the real risk
#   3. merge + submission CSV are written
#
# HOW IT FORCES stage 2 to run: RUN_INFER_STAGE1_MAX_TOKENS=48 truncates every
# stage-1 response before it can emit \boxed, so all smoke questions become rescue
# candidates and the second model is guaranteed to load. These env overrides are
# TEST-ONLY; the real submission run uses the hardcoded defaults (8192 / 4096).
#
# WHY a clean venv: `pip install vllm` into the container conda env breaks its
# numpy-1.x-ABI scipy/sklearn. See scripts/launch_difficulty_v2.sh for the full
# rationale. Same pinning (vllm==0.8.5 + torch==2.6.0 cu124) used here.
#
# PREREQ (one-time):  echo "hf_..." > ~/.hf_token && chmod 600 ~/.hf_token
#
# After launch:
#   kubectl get pods                 # find pod name
#   kubectl logs -f <pod_name>       # tail live (look for the two "loading ..." lines)
#   kubectl delete pod <pod_name>    # kill, free GPU quota
#
# Expect ~10-15 min cold start (vllm wheels + 2 model downloads), then ~2-3 min run.

set -e

launch.sh \
  -g 1 -v a5000 -m 48 -c 8 -B \
  -i ghcr.io/ucsd-ets/scipy-ml-notebook:stable \
  -- bash -c '
    set -e

    # ignore + wipe any contaminated user-site (persists in PVC across pods)
    export PYTHONNOUSERSITE=1
    rm -rf "$HOME/.local/lib/python3.11/site-packages" "$HOME/.local/bin"

    cd "$HOME/151B_SP26_Competition" && git fetch origin main && git reset --hard FETCH_HEAD

    # clean isolated venv (rebuild if vllm version is wrong)
    VENV="$HOME/.venv-smoke"
    if [ -d "$VENV" ] && ! "$VENV/bin/pip" freeze 2>/dev/null | grep -q "^vllm==0.8.5$"; then
      echo "--- venv has wrong/missing vllm, rebuilding ---"; rm -rf "$VENV"
    fi
    [ -d "$VENV" ] || python -m venv "$VENV"
    source "$VENV/bin/activate"

    pip install -q --upgrade pip
    pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    pip install -q -r requirements.txt

    echo "--- env sanity ---"
    python -c "import torch, vllm, transformers; print(f\"torch={torch.__version__} vllm={vllm.__version__} transformers={transformers.__version__}\")"
    python -c "import torch; assert torch.cuda.is_available(); print(f\"CUDA OK: {torch.cuda.get_device_name(0)}\")"

    # build an 8-question smoke subset (4 MCQ + 4 FF) from the tracked public set,
    # stripped to the private-file schema (id, question, options)
    python - <<PY
import json
rows = [json.loads(l) for l in open("data/public.jsonl")]
mcq  = [r for r in rows if r.get("options")][:4]
ff   = [r for r in rows if not r.get("options")][:4]
sub  = [{k: r[k] for k in ("id", "question", "options") if k in r} for r in mcq + ff]
with open("data/smoke_subset.jsonl", "w") as f:
    for r in sub:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(sub)} smoke questions ({len(mcq)} MCQ + {len(ff)} FF)")
PY

    echo "--- running run_inference.py (forced rescue path) ---"
    RUN_INFER_STAGE1_MAX_TOKENS=48 RUN_INFER_STAGE2_MAX_TOKENS=96 \
    HF_TOKEN=$(cat "$HOME/.hf_token") python run_inference.py \
      --private-path data/smoke_subset.jsonl \
      --output-csv /tmp/smoke_submission.csv \
      --tensor-parallel-size 1

    echo "--- SMOKE TEST RESULT ---"
    echo "row count (expect 8 + header = 9):"; wc -l /tmp/smoke_submission.csv
    head -3 /tmp/smoke_submission.csv
    echo "SMOKE TEST PASSED if you see two model loads above, 9 lines here, and no OOM/traceback."
  '

echo ""
echo "Pod launched. Find name: kubectl get pods"
echo "Tail:           kubectl logs -f <pod_name>"
echo "PASS criteria in the logs:"
echo "  - '[stage 1] generating ...' then the subprocess exits"
echo "  - '[stage 2] ... loading TrevorDuong/qwen3-4b-thinking-grpo-strict70' WITHOUT an OOM"
echo "  - 'Wrote submission: /tmp/smoke_submission.csv  (8 rows)'"
echo "Then delete the pod: kubectl delete pod <pod_name>"
