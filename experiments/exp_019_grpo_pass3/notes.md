# Experiment: grpo_pass3

**Date:** 2026-05-19
**Baseline compared against:** exp_018_pass2_rescue (Kaggle 0.628, local 60.39%) — current best
**Stage-1 reference:** exp_017_pass2_stage1 (local 56.84%)

## Hypothesis

Pass 2 produced a clean +0.89pp local lift at stage-1 over pass 1, with the gain compounding perfectly through the exp_014 rescue stack (+0.017 Kaggle, translation rate ~1.91). The pass-over-pass lift is **growing, not shrinking** (pass 0→1 was +0.62pp at stage-1; pass 1→2 was +0.89pp), which contradicts the theoretical diminishing-returns story.

If pass 3 holds the trend, expect **+0.5 to +0.9pp local at stage-1** → roughly +0.010 to +0.020 Kaggle if the rescue compounding repeats → projected stack score in the **0.638-0.648** range.

Realistic counterarguments:
- The "learning band" (1-3 correct) shrunk from exp_015's pool to 246 prompts here; even after loose filtering we got 72 vs exp_015's 58. Slightly more material to work with, not less.
- Pass-2 already clips 43% of samples — pass-3 chains may run even longer. We may need to drop `max_completion_length` further if OOM recurs (exp_010 lesson).
- Reward signal may be tighter — with most prompts at 0/4 (710) or 4/4 (170), the 1-3 band is concentrated near the policy's decision boundary, which is exactly what we want, but reward_std could collapse again if length_bonus isn't enough.

## Change from baseline (exp_015)

**Two variables changed:** base model + curriculum file. Everything else byte-identical to exp_015:
- `base_model`: `qwen3-4b-thinking-grpo-strict70` → **`qwen3-4b-thinking-grpo-pass2`**
- `curriculum_file`: `curriculum_v2.json` (58 prompts) → **`curriculum_pass3.json` (72 prompts: 24 MCQ / 48 FF)**

Recipe held constant from exp_015 (proven good):
- `max_completion_length=4096` — prevents exp_010 OOM in `entropy_from_logits`
- `length_bonus` reward (max 0.05, cap 16384 chars) — prevents reward_std=0
- `lora_r=16, lora_alpha=32, lora_dropout=0.0`
- `learning_rate=2e-5, beta=0.01, num_generations=4`
- `gradient_accumulation_steps=4` → effective batch 4 prompts × 4 gens = 16 per step
- Format reward weights (has_think_close, has_boxed, boxed_post_think, single_boxed_post_think)
- `save_steps=10` → adapter checkpoints push to HF every 10 grad updates
- `epochs=1` → 1 pass over 72 prompts at batch 4 = ~18 grad updates total (vs exp_015's 48)

**Important:** exp_019's training will be SHORTER than exp_015 in absolute steps because the curriculum is similar size but exp_015 did `4 epochs/2 epochs * 58 = 48 steps` — recheck epoch math against exp_015 once training kicks off.

## Curriculum details (data/curriculum_pass3.json)

Source: `data/difficulty_samples_pass2.jsonl` (1126 prompts × 4 samples, sampled from pass-2 GRPO model at T=1.0, max_new_tokens=6144)

| Slice | n | Notes |
|---|---:|---|
| Total sampled | 1126 | Full public.jsonl |
| 0 correct | 710 | Beyond model capability — excluded |
| 1 correct | 122 | Primary learning signal |
| 2 correct | 67 | (loose filter excludes — too risky for reward variance) |
| 3 correct | 57 | (loose filter excludes) |
| 4 correct | 170 | No gradient — excluded |
| **Strict filter** (1-3, no clip) | 92 | Discarded by 43% sample clip rate |
| **Loose filter** (==1, allow clip) | 122 (74 MCQ / 48 FF) | exp_015 recipe |
| **+ FF:MCQ≥2 cap** | **72 (24 MCQ / 48 FF)** | Final curriculum |

## Plan

### On DSMLP
1. SSH to dsmlp-login, `cd ~/151B_SP26_Competition && git pull origin main`
2. `kubectl get pods` — clean up any stale pod (mandatory; quota slot)
3. Launch via `scripts/launch_grpo_pass3.sh` (see scripts dir)
4. `kubectl logs -f <pod_name>` — watch for `reward_std` per step; first 2-3 steps must show > 0
5. If reward_std=0 for >3 consecutive steps, kill and investigate length_bonus saturation

### Watch points
- **Step 1 reward_std > 0** (length_bonus working)
- **No OOM** — exp_010 hit `entropy_from_logits` OOM at max_completion=6144; we're at 4096 so should be safe but watch for it
- **Reward trend** — by step 10-15 the mean correctness reward should drift up if learning is happening
- **Adapter checkpoints** — should push to HF every 10 grad updates (`save_steps=10`)

### After training (12h pod or earlier)
1. Merge adapter via `experiments/exp_015_grpo_pass2/merge_and_push.ipynb` — just swap `BASE`, `ADAPTR`, `TARGET` constants
2. Verify HF Hub upload of `TrevorDuong/qwen3-4b-thinking-grpo-pass3` (full merged, vLLM-loadable)
3. Scaffold `exp_020_pass3_stage1` (mirror of exp_017 for pass-2)
4. Kaggle run on `cse151b-notebook.ipynb` with `EXPERIMENT="exp_020_pass3_stage1"`
5. Score locally, compare vs exp_017's 56.84%

## Success / abort criteria

| Local (stage-1) vs exp_017 (56.84%) | Interpretation | Action |
|---|---|---|
| ≥ +0.5pp (≥ 57.34%) | Pass-3 holds the trend | Layer rescue → exp_021 → submit if ≥ 60.5% |
| 0.0 to +0.5pp | Flat — diminishing returns showing | Skip rescue submit; pivot to SFT v2 |
| < 0.0pp | Pass-3 regressed | Discard. Try earlier checkpoint (ckpt-10) before pivot |

## Results

_(to be filled after training + inference)_

## Conclusion

_(to be filled)_

## Next lever

- [ ] If exp_019/020/021 win: pass-4 with another fresh resample (only if lift ≥ +0.3pp)
- [ ] If pass-3 stalls: pivot to **SFT v2** — must start from `Qwen3-4B-Thinking-2507` + include MCQ data (exp_008 root-cause fix)
- [ ] Document the diminishing returns curve once we have 3 pass data points
