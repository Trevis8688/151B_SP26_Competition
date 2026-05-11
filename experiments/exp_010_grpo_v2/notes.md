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

### Phase 0 — DSMLP onboarding (in progress)
1. ✅ SSH into `dsmlp-login.ucsd.edu` (works for user `trduong`, Duo every 8h)
2. GPU survey — `kubectl get nodes` is blocked for students; check DataHub status page in browser
3. ❌ Custom Docker image — **skipped**. Stock `scipy-ml-notebook:stable` + pip install our deps is simpler for this use case
4. Pilot: 5-step GRPO smoke test via `pilot.py` (~20 min) — see `dsmlp_runbook.md` §0.3

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
