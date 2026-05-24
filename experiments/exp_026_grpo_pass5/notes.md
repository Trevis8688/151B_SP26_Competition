# Experiment: grpo_pass5

**Date:** 2026-05-22
**Baselines:**
- exp_024_pass4_stage1 — pass-4 stage-1 (local 60.75%, **leaderboard stage-1-only 0.600**) ← the model to beat
- exp_020_pass3_stage1 — pass-3 stage-1 (local 58.61%, leaderboard stage-1-only 0.586)
- exp_017_pass2_stage1 — pass-2 stage-1 (local 56.84%, leaderboard stage-1-only 0.586)

## Hypothesis

GRPO **pass 5** from the merged pass-4 model (`TrevorDuong/qwen3-4b-thinking-grpo-pass4`). Single intentional change from exp_022 (the pass-4 recipe): base model + curriculum advance one pass (pass-4 → pass-5). Optimizer recipe byte-identical (G=4, T=1.0, max_completion=5120, lr=2e-5, beta=0.01, length_bonus, matched-sampler curriculum).

**What pass-5 is actually testing.** Pass-4 was the FIRST pass to move the held-out board (0.586 → 0.600 stage-1-only). But that gain came from a *recipe fix* — the matched-sampler curriculum (top_k=−1, top_p=1.0, T=1.0, N=8, 5120 budget, allow-clipped) — not from "another pass" per se: pass-2→pass-3 (old top_k=20 curriculum) was 0pp board transfer. So the open question is whether the matched-sampler recipe **unlocked ongoing learning** (pass-5 keeps climbing) or just gave a **one-time step** off the converged plateau (pass-5 ties). Pass-4's +1.4pp was only ~0.6σ on the ~470-q board split — promising, not proven. Pass-5's own stage-1-only board test is the cleanest second data point on whether GRPO is still a live lever.

### Tempered prior

- The matched curriculum did NOT visibly raise the live-step fraction in pass-4's logs (~7% of late-epoch steps had `correctness_reward/std > 0`, vs the historical ~10%). The dead-step root cause is 4-bit policy peakedness (BnB), which pass-5 does not address. So a TIE is a real possibility.
- If the pass-5 curriculum (filtered from pass-4-policy samples) comes back **< 40 prompts**, that itself signals pass-4 has saturated this prompt set at the 5120 budget — a weak prior for a pass-5 gain. Note the size in Results.

## Change from baseline (exp_022)

**Single variable per stage:** `base_model` pass-3 → pass-4, and the curriculum is re-sampled from the pass-4 policy (everything else — filter recipe, optimizer, reward, sampler params — byte-identical to pass-4).

## Plan (training lifecycle, DSMLP)

This runs on DSMLP **in parallel with exp_025 rescue on Kaggle** — independent compute, no contention. exp_025 does not block pass-5: the curriculum is sampled from the pass-4 *policy weights* (which already exist), not from any rescue output.

0. **Delete the errored pass-4 pod first** (`kubectl get pods` → `kubectl delete pod <errored>`) — it holds the GPU slot.
1. SSH to dsmlp-login, `cd ~/151B_SP26_Competition && git pull origin main`.
2. **Curriculum resample** (long pole, ~4-8h, the SIGALRM→ProcessPool hang is fixed):
   `bash scripts/launch_difficulty_pass4.sh` (smoke first: `LIMIT=20 bash scripts/launch_difficulty_pass4.sh`).
   → produces `data/difficulty_samples_pass4.jsonl`.
3. **Filter into the curriculum:**
   ```
   python scripts/filter_curriculum_v2.py \
     --in data/difficulty_samples_pass4.jsonl \
     --out experiments/exp_026_grpo_pass5/curriculum_pass5.json \
     --min-correct 2 --max-correct 6 --allow-clipped --ff-mcq-ratio 2.0
   ```
   Commit `curriculum_pass5.json` to main (scp to Mac if DSMLP git push isn't set up, then commit here), so the training pod's `git reset --hard FETCH_HEAD` sees it.
4. **Train:** `bash scripts/launch_grpo_pass5.sh` (refuses to launch if curriculum missing). Resumes from HF every save_steps=10 if the 12h pod times out.
5. **Merge** the final checkpoint → `qwen3-4b-thinking-grpo-pass5` via `experiments/exp_026_grpo_pass5/merge_and_push.ipynb` on Kaggle (set `SUBFOLDER` to the last pushed checkpoint).
6. **Stage-1-only board test (decisive):** scaffold a pass-5 stage-1 inference exp (clone exp_024, swap `model_id` → pass-5), run full split on Kaggle, build a stage-1-only submission, submit. Judge on the BOARD, not local % (pass-3's +1.77pp local → 0pp board).

## Success / abort criteria (leaderboard — pre-committed)

Judge vs pass-4's **0.600** stage-1-only floor (split noise ~2.3pp = 1σ). Local % is diagnostic-only.

| Pass-5 stage-1 board (vs 0.600) | Interpretation | Action |
|---|---|---|
| ≥ ~0.610 (≥ +1pp) | Matched-sampler recipe is still climbing — GRPO is a live lever | Layer rescue (clone exp_025); consider pass-6 if time |
| ~0.600 (tie) | Matched-sampler was a one-time step; GRPO converged on the board | **STOP GRPO.** Gains stand (pass-4 best). Pivot remaining time to rescue-stage optimization (MCQ-only rescue, budget sweep) |
| < ~0.59 | Board regression (pass-5 hurt) | Discard; keep pass-4. Investigate earlier pass-5 checkpoints |

## Results

_(to be filled after curriculum build + training + merge + stage-1-only submission)_

- Curriculum size after filter: **156 (52 MCQ, 104 FF)** — well above the 40 floor; pass-4 has NOT saturated the prompt set. num_correct dist at N=8: 0=580, 1=131, 2=54, 3=47, 4=26, 5=30, 6=38, 7=59, 8=161 (bimodal; band [2,6]=195 pre-cap, 156 after FF:MCQ≤2.0). 697/1126 prompts had ≥1 clipped sample at 5120.
- Training: **completed — 130 grad updates** (checkpoints checkpoint-10 … checkpoint-130 in `…-pass5-ckpt`). The inline merge at end of `train_grpo.py` succeeded: merged model `TrevorDuong/qwen3-4b-thinking-grpo-pass5` = 8.04 GB single `model.safetensors`, last modified 2026-05-23 21:05Z. No Kaggle merge step needed — loads directly in vLLM.
- Stage-1-only board test scaffolded as **exp_029_pass5_stage1** (model_id swap pass-4→pass-5; original prompts; full split). Decisive vs the 0.600 pass-4 floor.

| | exp_017 pass-2 | exp_020 pass-3 | exp_024 pass-4 | exp_026 pass-5 |
|---|---:|---:|---:|---:|
| Leaderboard stage-1-only | 0.586 | 0.586 | 0.600 | TBD |
| Local public.jsonl | 56.84% | 58.61% | 60.75% | TBD |

## Conclusion

_(to be filled)_
