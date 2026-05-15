# Wrong free-form failures by domain — exp_014 (2026-05-15)

**Source:** `experiments/exp_014_rescue_v2_grpo/results.jsonl`
**Analyzer:** `scripts/analyze_wrong_freeform.py` (priority-ordered regex buckets)
**Scope:** 329 free-form questions that produced a `\boxed{}` but were judged wrong
(i.e. true reasoning failures — the 24 `missing_boxed` cases from exp_014 are
truncation, analyzed separately).

## Failure rate per math domain

| Bucket | Total in public | Wrong | Wrong rate | Share of all fails |
|---|---:|---:|---:|---:|
| **statistics** | 144 | **87** | **60.4%** | **26.4%** |
| applied_modeling | 62 | 37 | 59.7% | 11.2% |
| trigonometry | 61 | 35 | 57.4% | 10.6% |
| geometry | 58 | 32 | 55.2% | 9.7% |
| combinatorics_probability | 27 | 14 | 51.9% | 4.3% |
| algebra | 75 | 35 | 46.7% | 10.6% |
| (other) | 306 | 80 | 26.1% | 24.3% |
| sequences_series | 7 | 2 | 28.6% | 0.6% |
| linear_algebra | 4 | 3 | (too few) | 0.9% |
| number_theory | 6 | 3 | (too few) | 0.9% |
| calculus | 1 | 1 | (too few) | 0.3% |

751 free-form questions total. Numbers from priority-first keyword classifier;
"other" = no bucket pattern matched (mostly mixed word problems / function
manipulation / simplification — the model is **best** at these, ~74% correct).

## Headline reads

1. **Statistics is the single biggest lever.** 87 failures (26% of all
   free-form fails), 60% wrong rate. If we lift stats from 40% → 60% correct,
   that's ~29 questions recovered ≈ **+0.025–0.030 Kaggle** — bigger than any
   single experiment since exp_009.
2. **Three domains have ~60% wrong rate** (stats, applied modeling, trig). These
   are structural weaknesses, not random noise. Each is a candidate for
   targeted few-shots or domain-specific SFT data.
3. **Geometry and algebra are mid-band** (47–55% wrong). Less leverage per
   question but still real.
4. **The "other" bucket is healthy.** Lowest wrong rate (26%). Mixed word
   problems and function manipulation work fine — no need to spend effort here.
5. **Linear algebra / number theory / calculus are too rare to optimize for.**

## Implications for upcoming experiments

| Lever | Target | Expected lift | Cost |
|---|---|---|---|
| **Stats few-shots in MATH system prompt** | stats (87→~58 wrong) | +0.020–0.030 Kaggle | 1 Kaggle run (~80m) |
| **Applied modeling few-shots** | exp/decay/% (37→~22 wrong) | +0.010–0.015 | 1 Kaggle run |
| **Trig few-shots** | trig (35→~22 wrong) | +0.010–0.015 | 1 Kaggle run |
| **SFT v2 with domain-weighted data** | stats + applied + trig | bundle: +0.030–0.060 | 8–12h DSMLP |
| Best-of-N rescue (exp_011 scoped) | all 329 wrong_ff | +0.005–0.015 | 1 Kaggle run |

## Caveat: classifier noise

Keyword bucketing has known false positives/negatives. Three checks worth doing
before betting heavily on a domain:

1. **Sanity-check the stats bucket manually** — spot-check 10 random IDs from
   `reports/exp_014_wrong_ff_buckets.json` → `buckets["statistics"]`. If <80%
   are actually stats, the numbers above are inflated.
2. **The "other" bucket** still has 80 cases worth eyeballing — there may be
   another hidden cluster (e.g., function transformations).
3. **Domain overlap.** Some problems are both "applied modeling" and "stats"
   (e.g., a regression word problem). Priority-first matching forces a single
   bucket; the marginal benefit of multi-tagging is small but real.

## Next concrete action

**exp_016: stats-focused few-shot prompt augmentation.** Lowest cost, highest
expected lift. Add 2 worked-out statistics examples (one hypothesis test, one
std dev / sampling) to `FEWSHOT_MATH` in a new prompts.py. Run on the full
public set. If stats wrong-rate drops to <50% without regressing other domains,
keep it and submit. exp_005 lesson: do *not* include concrete numeric answers
in the few-shot examples — Qwen3-Thinking regurgitates them.

If it works, exp_017 = applied_modeling few-shots, exp_018 = trig few-shots.
Stack-able, each ~80 min on Kaggle. Hits the 60%-wrong-rate domains first.

## Artifacts

- `reports/exp_014_wrong_ff_buckets.json` — full ID → bucket mapping
- `scripts/analyze_wrong_freeform.py` — re-runnable classifier
