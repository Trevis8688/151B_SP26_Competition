# Experiment: grpo_v2

**Date:** 2026-05-11 (scaffold) / 2026-05-12 (Run 1 aborted, Run 2 pending)
**Status:** Run 1 aborted — recipe pivoted from full-196 → strict-70
**Baseline:** exp_009_grpo (local 55.95%, Kaggle **0.583**)

## Run 1 post-mortem (2026-05-12)

Launched 03:46 PT on a5000 with the full 196-prompt curriculum, max_completion=6144, 2 epochs. Killed at step 59/386 (~15% complete, 11h elapsed) after the data showed:

| Symptom | Cause |
|---|---|
| 67% of steps had `frac_reward_zero_std=1.0` (no gradient signal) | All-clipping on Phase 1B prompts — all 4 generations hit 6144 cap with no `\boxed{}` → all reward=0, advantage=0 |
| 11 min/step average (vs 80s pilot estimate) | Each clipped step generates 6144 tokens × 4 gens, dominating compute |
| ETA 72h vs 12h timeout | Would have hit DeadlineExceeded at ~17% with negligible learning |

**The original exp_010 hypothesis — "more data is better, 196 > 70" — was wrong.** The strict-70 curriculum's zero-clipping filter is load-bearing, not a nice-to-have. Phase 1B prompts need >6144 tokens to converge; including them throws away >60% of compute on dead gradient signal.

## Run 2 plan (pending launch)

Pivot back to exp_009's strict-70 curriculum, scale up the *training depth* instead of the data breadth.

| Knob | exp_009 (actual) | Run 1 (aborted) | **Run 2** | Rationale |
|---|---|---|---|---|
| Curriculum file | `sweet_spot_ids_clean.json` (70) | `sweet_spot_ids.json` (196) | **`sweet_spot_ids_clean.json` (70)** | Restore zero-clipping filter; every step has reward variance |
| Epochs | 0.8 (interrupted at ckpt-56) | 2 | 2 | 2.5× exp_009's effective update count (~35 vs ~14) |
| LR | 2e-5 | 1e-5 | 1e-5 | Keep |
| BETA | 0.01 | 0.02 | 0.02 | Keep — stronger KL anchor on smaller curriculum |
| MAX_COMPLETION | 6144 | 6144 | **4096** | strict-70 prompts all fit under 4096 in Phase 1 sampling; halves step time |
| save_steps | 4 (Drive-mirrored) | 25 | **10** | More frequent recoverable checkpoints |
| Adapter HF push | none | none | **every save_steps** | NEW: pushes adapter to `qwen3-4b-thinking-grpo-v2-ckpt` on each save. Survives DeadlineExceeded |

**Training math (Run 2):**
- 70 prompts × 2 epochs / (1 × 4 grad accum) = ~35 update steps
- Strict-70 was selected for zero-clipping at 4096 → typical step time on a5000 ~150-300s (pilot evidence)
- 35 update steps × ~250s = ~9 hours wall time (under 12h timeout with comfortable margin)

**Why this should beat exp_009:**
- 2.5× more gradient updates (35 vs ~14)
- LR is gentler (1e-5 vs 2e-5) → less collapse risk on extended training
- BETA is stronger (0.02 vs 0.01) → MCQ skill is better protected
- Same curriculum, same prompt distribution → directly comparable

**Risk:** marginal yield from steps 15→35 is unknown. exp_009 might have already been plateauing at step 14. If so, gains will be modest. But cost is one a5000 night, which is cheap.

## Run 2 v2 update (2026-05-12, post-design-review)

Realized that the per-step ETA in Run 2 v1 was based on rollout-bound generation with HF `model.generate()` — same backend as Run 1 — and revisiting Run 1's per-step times on non-clipping steps shows ~10 min/step, not 4-5 min. With 140 steps for 2 epochs that's ~23h, well over the 12h timeout.

**Root cause:** the script had `use_vllm=False`. All 4 rollouts per step were running through plain HF generate with no continuous batching.

**Run 2 v2 changes:**

| Knob | Run 2 v1 | **Run 2 v2** | Why |
|---|---|---|---|
| Generation backend | HF `model.generate` | **vLLM colocate** | 3-5× faster rollouts via continuous batching + paged attention |
| Base model quant | 4-bit BnB | **bf16/fp16** | vLLM can't read BnB checkpoints; needs vLLM-compatible weights |
| Attention | implicit (likely SDPA) | **explicit FA2** | `attn_implementation="flash_attention_2"` — O(n) memory |
| `vllm_gpu_memory_utilization` | — | **0.45** | Conservative on a5000 24GB; raise if first 5 steps stable |
| Gradient checkpointing | on | **on** | Memory budget is tight (8GB base + 11GB vLLM + 3-5GB train residual) |
| Pilot before full | optional | **mandatory** | LoRA-sync overhead in colocate is the failure-mode-of-record |

**Revised ETA:** ~3-4 min/step × 140 steps ≈ **9-10 hours**. Fits 12h container, but not with comfortable margin.

**Pilot validates (5 steps, ~25-30 min):**
1. `GRPOConfig vLLM fields:` log line — confirms TRL exposes the kwargs we pass
2. fp16/bf16 base loads without OOM at gpu_memory_util=0.45
3. vLLM init banner appears, no OOM mid-init
4. First step's rollouts finish in <90s
5. Steps 2-5 wallclock <4 min each (no growing LoRA-sync stall)

**Launch sequence after pilot passes:**
```
K8S_TIMEOUT_SECONDS=43200 launch.sh -g 1 -v a5000 -m 48 -c 8 -B \
  -- bash -c 'cd /home/$USER/151B_SP26_Competition && \
              pip install -q -r experiments/exp_010_grpo_v2/requirements.txt && \
              pip install -q --no-deps vllm==0.6.6.post1 && \
              HF_TOKEN=$(cat /home/$USER/.hf_token) \
                python experiments/exp_010_grpo_v2/train_grpo_v2.py'
```

The `--no-deps` on vllm is load-bearing: vllm 0.6.6 has its own torch pin that conflicts with the cu124 stack pinned in requirements.txt. We let pip-check warnings show after but don't let it pull a replacement torch.

**Open risks (not yet validated):**
- TRL 0.21 colocate API surface — exact kwarg names verified via the `_vllm_fields` printout
- LoRA hot-swap cost per step — unknown overhead, the pilot is mostly to measure this
- Inference dtype: training base is bf16 (a5000), Kaggle T4 inference is fp16. Adapter applies linearly so drift should be negligible, but worth noting if Phase 2 score is unexpectedly low

## Run 2 v3 (torch 2.6 upgrade) — 2026-05-12

Run 2 v2's pilot exposed the closing of the version trap: **vllm 0.6.6.post1 does NOT support `Qwen3ForCausalLM`.** The error chain:

```
torch 2.5.1 (pinned for exp_009 stack)
  → requires vllm ≤ 0.6.6
  → vllm 0.6.6 doesn't know Qwen3ForCausalLM (only Qwen2)
  → Qwen3 support landed in vllm 0.7.3+
  → vllm 0.7+ requires torch ≥ 2.6
```

The monkey-patch worked (`dropped: ['model_impl']` was correctly filtered), but no patch can teach vllm 0.6.6 a model architecture it has no implementation for.

**Run 2 v3 = bite the bullet, upgrade torch.**

| Dep | Run 2 v2 | **Run 2 v3** | Why |
|---|---|---|---|
| torch | 2.5.1+cu124 | **2.6.0+cu124** | vllm 0.7+ requires it |
| torchvision/torchaudio | 0.20.1 / 2.5.1 | **0.21.0 / 2.6.0** | Match torch |
| vllm | 0.6.6.post1 | **0.8.5** | First stable line with Qwen3 |
| flash-attn | >=2.6.0 | **>=2.7.0 (2.7.4.post1 pinned)** | 2.6 wheels are torch-2.5 only |
| bitsandbytes | >=0.46.1 | unchanged | Already supports torch 2.6 |
| trl/peft/accelerate/transformers | unchanged | unchanged | Forward-compat |

**Install order is load-bearing** — requirements.txt no longer includes flash-attn or vllm because they need flags pip refuses to combine with -r:
```
pip install -r requirements.txt                              # torch 2.6 + bnb/trl/etc.
pip install --no-build-isolation flash-attn==2.7.4.post1     # compiles against installed torch
pip install --no-deps vllm==0.8.5                            # don't let it pull torch back
```

**Risk that landed v3 here, not earlier:**
I should have caught the Qwen3 architecture support gap when picking vllm 0.6.6 in v2 — that's on me. The Qwen2 vs Qwen3 distinction is exactly the kind of compat detail that costs a pilot run to surface, and in retrospect a 30-second `grep Qwen3 vllm/model_executor/models/` would have spotted it before pushing v2.

**Pilot still mandatory** — same 5-step validation as v2, just on the upgraded stack. Same goalposts: (1) bf16 base loads no-OOM, (2) vllm init succeeds, (3) first step <90s rollout, (4) steps 2-5 <4 min each.

---


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

### 🌙 OVERNIGHT RUN IN PROGRESS — started 2026-05-12 03:46 PT

**Pod name:** `trduong-1240942` (a5000, 48 GB RAM, 8 CPU)
**Expected completion:** ~12:15 PT (8.5h training + ~30 min HF merge/upload)
**Timeout:** 12:46 PT (K8S_TIMEOUT_SECONDS=43200)

Step 1 health check passed:
- 79s/step → ~8.5h ETA for 386 dataloader iterations (= 96 optimizer steps × grad_accum=4)
- max_completion=6144 fits on a5000 24GB (flash-attn FA2 backend doing its job)
- 650-token completions on step 1 — nowhere near the cap, room to spare
- bf16 active, no OOM

**Morning checklist:**
```bash
ssh trduong@dsmlp-login.ucsd.edu
kubectl get pods                                 # check status: Completed / Running / OOMKilled / Error / DeadlineExceeded
kubectl logs trduong-1240942 | tail -100         # see last steps + final messages
```

Expected good outcomes (in order of preference):
1. **`ALL DONE.`** at end of logs → model pushed to `https://huggingface.co/TrevorDuong/qwen3-4b-thinking-grpo-v2`. Move to Phase 2.
2. **`Pushing to https://huggingface.co/...`** but no ALL DONE → upload still in progress; wait 5 more min.
3. **Training done but HF push failed** → adapter is at `experiments/exp_010_grpo_v2/adapter_final/` on NFS. Re-run just the HF push manually from a small pod.
4. **OOMKilled / DeadlineExceeded mid-run** → checkpoints at `experiments/exp_010_grpo_v2/checkpoints/checkpoint-25,-50,-75` (every 25 steps) survive on NFS. Recoverable.

**Don't forget after collecting results:** `kubectl delete pod trduong-1240942`

---

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
launch.sh -g 1 -v a5000 -m 48 -c 8   # 48 GB RAM — 16 GB default OOMs during 4-bit quant load
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
K8S_TIMEOUT_SECONDS=43200 launch.sh -g 1 -v a5000 -m 48 -c 8 -B \
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
