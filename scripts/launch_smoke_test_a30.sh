#!/bin/bash
# DSMLP smoke test — BROAD GPU POOL variant (use when A5000 is saturated).
#
# Identical job to scripts/launch_smoke_test.sh, but launched through the class
# wrapper `launch-sp26-cuda128.sh` (image ghcr.io/ucsd-ets/sp26-cuda128:main)
# with `-l gpu-class=medium -W CSE151B_SP26_A00` and NO `-v` pin. That lets the
# scheduler place the pod on ANY free medium-class GPU (A30 24GB / A5000 / ...)
# instead of only the single A5000 node — far less queueing near a deadline.
# run_inference.py runs unchanged on any CUDA GPU (dtype=float16, tp=1).
#
# Flag recipe is from the class announcement @218 (cerebrum). If `launch.sh`
# rejects `-W` ("illegal option"), drop it and relaunch — `-l gpu-class=medium`
# alone may suffice.
#
# Run from dsmlp-login after: git fetch origin main && git reset --hard FETCH_HEAD
# PREREQ (one-time):  echo "hf_..." > ~/.hf_token && chmod 600 ~/.hf_token
#
# After launch:
#   kubectl get pods                 # wait for STATUS: Running (not Pending)
#   kubectl logs -f <pod_name>       # tail
#   kubectl delete pod <pod_name>    # free the GPU when done

set -e

launch-sp26-cuda128.sh \
  -g 1 -m 48 -c 8 -B \
  -l gpu-class=medium -W CSE151B_SP26_A00 \
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

    # 8-question smoke subset (4 MCQ + 4 FF) from the tracked public set,
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
echo "Pod launched on the medium GPU class. Find name: kubectl get pods"
echo "If STATUS is Pending > ~2 min, run: kubectl describe pod <name> | tail -25"
echo "PASS criteria in the logs:"
echo "  - two model loads (grpo-pass2 then grpo-strict70) with NO OOM"
echo "  - 'Wrote submission: /tmp/smoke_submission.csv  (8 rows)'"
echo "Then: kubectl delete pod <name>"
