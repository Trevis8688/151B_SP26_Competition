# Experiment: grpo_v2

**Date:** 2026-05-11
**Status:** Scaffolded — awaiting DSMLP runbook + training
**Baseline:** exp_009_grpo (local 55.95%, Kaggle **0.583**)

## Hypothesis

exp_009 trained on only **70 prompts** (strict curriculum: zero clipping, sweet-spot only) for **80% of one epoch** (checkpoint-56/70, aborted by Drive-full) and gained **+3.2pp Kaggle** with **no MCQ regression**. Scaling the same recipe — more data, more steps, more stable LR — should extract more signal.

**Specific bet:** the 70-prompt run was data-starved; the 196-prompt curriculum (Phase 1 + Phase 1B, full set) gives 2.8× more training signal, and a full 2-epoch run gives ~5× more gradient updates than the 56-step exp_009 run.

## Change from baseline (exp_009)

| Knob | exp_009 (strict-70) | exp_010 (v2) | Rationale |
|---|---|---|---|
| Curriculum file | `sweet_spot_ids_clean.json` (70) | `sweet_spot_ids.json` (196) | Use the full Phase 1 + 1B sweet-spot set, not just zero-clipping |
| Epochs | ~0.8 (checkpoint-56/70) | 2 (full) | Was data-starved + cut short; more steps |
| LR | 2e-5 | **1e-5** | Less aggressive; reduce policy collapse risk on bigger data |
| BETA (KL) | 0.01 | 0.02 | Slightly stronger KL anchor; protect base-model MCQ skill |
| NUM_GENERATIONS | 4 | 4 | Keep — same throughput |
| MAX_COMPLETION_LEN | 6144 | 6144 | Keep — 8192 caused KV-cache OOM in earlier pilots |
| Save strategy | every 4 steps to Drive | every 25 steps to container disk | DSMLP has plenty of local disk; no Drive bottleneck |
| Compute target | Colab Pro+ A100 (~5min/step) | DSMLP V100/A5000 (TBD) | Colab Pro+ exhausted; DSMLP has 12h batch jobs |

**Training math:**
- 196 prompts / (1 × 4 grad accum) = 49 grad steps per epoch
- 2 epochs = 98 grad steps
- ~5 min/step on V100-class = ~8 hours → **fits in one DSMLP 12-hour batch job**

## Why DSMLP (not Colab/Kaggle)

| Platform | Limit | Verdict |
|---|---|---|
| Colab Pro+ | Idle disconnect; A100 by quota only; Drive bottleneck | Exhausted user quota |
| Kaggle | T4 x2 (SM 7.5, no bf16, no FA2), 9hr session | Too slow for 4B GRPO training |
| **DSMLP** | `launch.sh -g 1 -B` → 12hr exclusive GPU, batch mode | **Best fit** |
| RTX 3060 (local) | 12GB VRAM, QLoRA only, no FA2 | Backup; tight on seq=6144 |

DSMLP gotchas (from the IT services KB doc):
- Don't run on `dsmlp-login` directly — always via `launch.sh`
- `-B` for batch (unattended); collect output via `kubectl logs <pod>`
- `K8S_TIMEOUT_SECONDS=43200` env var bumps timeout to 12h
- `-v <gpu_type>` (e.g. `-v 1080ti`) requests a specific GPU; check DataHub status page for current cluster GPU types
- Wait time for premium GPUs (V100, A100) can be long during peak; queue early

## Plan

### Phase 0 — DSMLP onboarding (paused 2026-05-11 — partial)
1. ✅ SSH into `dsmlp-login.ucsd.edu` (user `trduong`, Duo every 8h)
2. ✅ GPU menu surveyed via `launch.sh -h`: `1080 | 1080ti | 2080ti | a30 | a5000 | a100 | h100 | rtxtitan | l40s`. Default `launch.sh -g 1` (no `-v`) landed on **RTX 2070 / 8 GB — too small** for our settings. Must use `-v a5000` (24 GB Ampere, bf16, FA2) or fallback `-v a30`/`-v l40s`.
3. ✅ Custom Docker image — **skipped**. Stock `scipy-ml-notebook:stable` + `pip install -r requirements.txt` works.
4. ⏸ Pilot — **not yet run**. Scripts are pushed to main (commit `4d42c71`); resume by re-launching a5000 pod and running `pilot.py`.

### Resume checklist (where to pick up next session)

**Environment confirmed working:**
- Pod image: `ghcr.io/ucsd-ets/scipy-ml-notebook:stable`
- Pre-installed: torch 2.2.1+cu121, transformers 4.44.2, datasets 3.0.0, accelerate 0.34.2
- Driver: 535.161.08 (CUDA 12.2 max); nvcc in container is 12.0
- Disk: `/home/trduong` is 135 TB NFS with 80 TB free (plenty for checkpoints)
- Decision: `requirements.txt` keeps cu124 pin → pip will reinstall torch (3–5 min)

**To resume — exact commands:**
```bash
# 1. From local laptop
ssh trduong@dsmlp-login.ucsd.edu

# 2. On dsmlp-login (NOT inside a pod yet)
launch.sh -g 1 -v a5000              # interactive, for pilot
# wait until prompt changes to trduong@trduong-XXXXX

# 3. Inside the pod
cd ~
git clone https://github.com/Trevis8688/151B_SP26_Competition.git
cd 151B_SP26_Competition
pip install -q -r experiments/exp_010_grpo_v2/requirements.txt
python experiments/exp_010_grpo_v2/pilot.py 2>&1 | tee pilot.log
# success line: "✅ Pilot complete. Stack works."

# 4. If pilot passes, exit pod and launch full training as batch
exit
# back on dsmlp-login:
K8S_TIMEOUT_SECONDS=43200 launch.sh -g 1 -v a5000 -m 32 -c 8 -B \
  -- bash -c '
    set -e
    cd ~ && git clone https://github.com/Trevis8688/151B_SP26_Competition.git
    cd 151B_SP26_Competition
    pip install -q -r experiments/exp_010_grpo_v2/requirements.txt
    export HF_TOKEN=<paste token here>
    python experiments/exp_010_grpo_v2/train_grpo_v2.py
  '
# Save the returned pod name; monitor with: kubectl logs -f <pod_name>
```

**Open items / known unknowns:**
- cu124 wheels with driver 12.2 — should work (forward-compat), but if pilot fails on `libcuda.so` errors, fall back to cu121 wheels (one-liner in runbook §0.3).
- HF_TOKEN required only for full train (not pilot). Get from huggingface.co/settings/tokens (write scope for the push at the end).
- `nvcc` in container is 12.0; bitsandbytes uses `BNB_CUDA_VERSION=124` env var in the script — if it complains about mismatched CUDA, set to `121` instead.
- a5000 was idle at 2026-05-11 15:40 PT; availability may differ on resume. If a5000 won't schedule, try `-v a30` → `-v l40s` → `-v a100` (in order).

### Phase 1 — Full training
1. `launch.sh -g 1 -v <best_avail> -m 32 -B -- python train_grpo_v2.py`
2. Monitor via `kubectl logs -f <pod>` from local laptop
3. Adapter pushed to HF Hub at end of training
4. Output: `TrevorDuong/qwen3-4b-thinking-grpo-v2`

### Phase 2 — Eval
1. Kaggle inference notebook (point at new HF model)
2. Score locally on public.jsonl, compare vs exp_009
3. Submit to Kaggle if local ≥ exp_009 baseline

## Files

```
experiments/exp_010_grpo_v2/
├── notes.md                    This file
├── config.json                 Training hyperparams (not Kaggle inference config — that comes later)
├── prompts.py                  Copy of exp_009 prompts (must match for distribution alignment)
├── sweet_spot_ids.json         196 curriculum IDs (Phase 1 + 1B sweet-spot)
├── train_grpo_v2.py            Real training script (ported from exp_009/train_grpo.ipynb)
├── pilot.py                    5-step sanity check — run before the 12hr batch
├── requirements.txt            pip deps on top of scipy-ml-notebook:stable
└── dsmlp_runbook.md            launch.sh playbook + monitoring + recovery
```

Decision: **no custom Docker image.** DSMLP's stock `scipy-ml-notebook:stable`
already has torch + scipy stack; we install GRPO deps (`trl`, `peft`, `bitsandbytes`)
via `pip install -r requirements.txt` inside the container. Custom Docker only
needed if we run >5 jobs and install time becomes a bottleneck.

## Risks / open questions

1. **DSMLP GPU type unknown.** If only 1080Ti available, training time balloons (~3× V100). Plan B: pilot a 1-epoch run anyway.
2. **vLLM in TRL inside Docker.** Exp_009 abandoned this on Colab due to install cascade. In a clean Dockerfile we *might* get vLLM-accelerated rollouts, cutting ~3× off training time. Worth trying in Phase 0.
3. **Scale could hurt, not help.** More data + more steps could trigger policy collapse (the 70-prompt curriculum may have been favorable precisely because it was narrow). The lower LR + higher BETA are the hedge; if reward variance flattens after ~30 steps, kill and try LR=5e-6.

## Conclusion
- [ ] Keep
- [ ] Discard
- [ ] Needs variant — next experiment idea:
