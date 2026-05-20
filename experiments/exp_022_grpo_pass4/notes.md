# Experiment: grpo_pass4

**Date:** 2026-05-20
**Baselines:**
- exp_019_grpo_pass3 (training run — checkpoint-69 merged → pass-3 model)
- exp_020_pass3_stage1 (local 58.61% — pass-3 stage-1 reference)
- exp_021_pass3_rescue (local **62.17%**, Kaggle TBD — current best stack)

## Hypothesis (Path: matched-sampler curriculum, no G/T change)

Pass-2, pass-3, and the pass-4 pilot all confirmed the same thing: only ~10% of GRPO steps produced useful gradient (reward_std > 0). The pilot at PILOT_STEPS=6 was too noisy to discriminate between G=4 and G=6 (`frac_reward_zero_std=0` across all configs is a sample-size artifact, not signal), and the plain read was that G=6 actively *hurt* correctness variance vs G=4 — so we explicitly chose NOT to change G or T.

The intervention pass-4 tests is upstream: every prior curriculum was sampled at vLLM `top_k=20, top_p=0.95`, but GRPO training has always sampled at the TRL/transformers defaults `top_k=None, top_p=1.0` (truncation disabled — confirmed from TRL 0.21 source). That distribution mismatch was baked into every curriculum we've ever used. A prompt that lands at 2/4 correct in `top_k=20` sampling can sit at 4/4 or 0/4 in training (because the per-token distribution differs), and that's a real source of dead steps independent of G or T.

For pass-4 we close this gap. Difficulty is re-sampled from the pass-3 policy with the **matched** per-token distribution (vLLM `top_k=-1`, `top_p=1.0`, `T=1.0`, `N=8`). The N=8 samples give tighter p_correct estimates than prior runs' N=4. We then filter to the band `2 ≤ num_correct ≤ 6` (p_correct in [0.25, 0.75]), `max_length=5120` (training's actual max_completion), FF:MCQ ≥ 2.

### Realistic expectations

| Outcome | P | E[stage-1 lift over exp_020 (58.61%)] |
|---|---:|---:|
| Matched curriculum raises in-band-prompt fraction enough to materially reduce dead steps | 0.50 | +0.2 to +0.5pp |
| Modest improvement — band more accurate but the policy is approaching convergence | 0.30 | +0.05 to +0.2pp |
| The top_k=20→disabled gap was not the dominant dead-step driver; matched curriculum changes little | 0.15 | -0.05 to +0.1pp |
| Pass-3 model regressed (over-trained or curriculum has wrong direction) | 0.05 | < -0.1pp |

**Headline E[Δ]: +0.18pp at stage-1** vs exp_020 (range: -0.05 to +0.50pp). Smaller projection than pass-3's +1.77pp lift, because (a) the policy has had two prior passes already and KL room is shrinking, (b) this is a curriculum intervention, not a recipe change.

**Decision point** (vs exp_020's 56.84% pass-2 / 58.61% pass-3 stage-1 floors): if pass-4 lifts ≥ +0.3pp stage-1, layer rescue (exp_023) and submit. If < +0.2pp, the GRPO well is converging and the next move is SFT v2 — not pass-5.

## Change from baseline (exp_019)

**Single variable changed:** `curriculum_file` from `curriculum_pass3.json` (88 prompts sampled at vLLM `top_k=20`) → **`curriculum_pass4.json`** (sampled from the pass-3 policy at training-matched `top_k=-1`, N=8).

Everything else byte-identical to exp_019:
- base_model = `TrevorDuong/qwen3-4b-thinking-grpo-pass3` (the freshly-merged pass-3)
- G=4, T=1.0, max_completion=5120, lr=2e-5, beta=0.01, length_bonus(max=0.05, cap=16384)
- per_device_train_batch_size=1, gradient_accumulation_steps=4, lora_r=16, lora_alpha=32, lora_dropout=0.0

**One safety fix** in `train_grpo.py`: `MERGED_DIR` points to `/tmp/merged_final` instead of the experiment dir, to dodge bug-082 (the 5GB PVC quota crash that killed both pass-2 and pass-3's inline merge). The merge is pushed to HF Hub regardless; the local copy is throwaway.

## Plan

### 1. Difficulty sampling (DSMLP, ~4-8h)
```
cd ~/151B_SP26_Competition && git pull origin main
bash scripts/launch_difficulty_pass3.sh
# Optional: smoke-test first by editing the launch script to prepend LIMIT=10
# to the python line — verifies throughput in ~10 min before committing to 4-8h.
```
Outputs land in PVC at `data/difficulty_samples_pass3.jsonl`.

### 2. Build curriculum
```
python scripts/filter_curriculum_v2.py \
  --in  data/difficulty_samples_pass3.jsonl \
  --out experiments/exp_022_grpo_pass4/curriculum_pass4.json \
  --min-correct 2 --max-correct 6 \
  --max-length 5120 \
  --ff-mcq-ratio 2.0
```
Expected output: 60-100 prompts (target band similar to exp_019's 88-prompt strict curriculum). If the band yields < 40 prompts, the policy may be too peaked even at the matched sampler — see "Watch-list" below.

### 3. Train (DSMLP, ~3-5h)
```
git add experiments/exp_022_grpo_pass4/curriculum_pass4.json
git commit -m "exp_022: pass-4 curriculum from matched-sampler pass-3 difficulty"
git push origin main
bash scripts/launch_grpo_pass4.sh
```
Inline fp16 merge → HF Hub. If push fails, fall back to `experiments/exp_022_grpo_pass4/merge_and_push.ipynb` on Kaggle.

### 4. Stage-1 inference + rescue
Same workflow as exp_020/exp_021: scaffold exp_023 (stage-1) + exp_024 (rescue) after pass-4 finishes.

## Watch-list during training

- Per-step `correctness_reward/std`. If it's > 0 on substantially more than the ~10% rate we saw in pass-3, the matched-curriculum hypothesis is confirmed mid-run. If it stays at ~10%, the dead-step root cause is policy peakedness (BnB quantization) and the next pass should test G or bf16 LoRA, not curriculum.
- Per-step `frac_reward_zero_std`. Same signal, from the trainer side.
- Mean entropy. Should be ≥ 0.22 (the pilot's G=4/T=1.0 reference); a meaningful drop would indicate the matched curriculum pulled in harder prompts.
- KL. Anything > 0.02 cumulative is unusual for a same-recipe continuation; would suggest a regime shift.

## Success / abort criteria

| Stage-1 (vs exp_020 58.61%) | Interpretation | Action |
|---|---|---|
| ≥ +0.3pp | Matched-sampler curriculum is a real lever | Rescue (exp_024), submit if ≥ exp_021's 62.17% |
| +0.1 to +0.3pp | Modest win consistent with E[Δ] | Rescue and evaluate; pass-5 worth a longer pilot |
| -0.1 to +0.1pp | No effect; the gap wasn't the bottleneck | Skip rescue submit; **pivot to SFT v2** |
| < -0.1pp | Regression | Discard; investigate via earlier checkpoints |

## Results

_(to be filled after training + stage-1 inference)_

| Metric | exp_019 pass-3 train | exp_022 pass-4 train | Δ |
|--------|---------------------:|---------------------:|---:|
| Curriculum prompts | 88 | TBD | — |
| Steps with reward_std > 0 (%) | ~10% | TBD | — |
| Mean entropy | TBD (have logs) | TBD | — |
| Final KL | 0.005 | TBD | — |

| Stage-1 (vs exp_020 58.61%) | exp_020 | exp_023 (pass-4 stage-1) | Δ |
|---|---:|---:|---:|
| Local overall | 58.61% | TBD | — |
| Local MCQ | 66.67% | TBD | — |
| Local free-form | 54.59% | TBD | — |

## Conclusion

_(to be filled)_
