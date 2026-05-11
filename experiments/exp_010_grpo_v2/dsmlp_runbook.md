# DSMLP runbook — exp_010 GRPO v2

End-to-end playbook for running exp_010 on UCSD DSMLP. Assumes you've already
SSH'd to `dsmlp-login.ucsd.edu` and confirmed access.

## Phase 0 — sanity check (~30 min)

### 0.1 Get an interactive GPU pod

```bash
# from dsmlp-login.ucsd.edu
launch.sh -g 1 -m 32 -c 8
# wait until prompt changes to trduong@trduong-XXXXX:~$
```

If GPUs are all busy you'll see a "Pending" status — `kubectl get pods` to check.
Wait time is usually 5–10 min during off-peak. To request a specific GPU type
(once you know what's available), add `-v 2080ti` or `-v A5000`.

### 0.2 Inside the pod — clone repo, install deps

```bash
# inside the container
git clone https://github.com/Trevis8688/151B_SP26_Competition.git
cd 151B_SP26_Competition

# install GRPO deps on top of scipy-ml-notebook base
pip install -q -r experiments/exp_010_grpo_v2/requirements.txt
```

Expected install time: 3–5 min. If torch reinstall is slow (cu124 wheel), that's
normal — first time only.

### 0.3 Run pilot (5 GRPO steps)

```bash
# export HF token so push to hub works on the real run
export HF_TOKEN=hf_xxxxxx   # ← put your token here, do NOT commit it

# pilot: 5 steps, no HF push, no checkpoints saved
python experiments/exp_010_grpo_v2/pilot.py 2>&1 | tee pilot.log
```

**What to check:**
- Pilot finishes in 15–25 min (longer if no bf16)
- No CUDA OOM during step 5
- Reward column actually moves (don't expect it to climb — it just shouldn't be identical for all 4 generations every step)
- Final line: `✅ Pilot complete. Stack works.`

**If pilot fails:**
- `OOMKilled` on rollouts → reduce `max_completion_length` (try 1536, then 1024). Edit `pilot.py` line `max_completion_length=2048`.
- `OOMKilled` on optimizer step → reduce `num_generations` from 4 to 2 (cuts KV cache and grad buffer in half).
- bitsandbytes errors → `pip install --upgrade bitsandbytes` and re-run.
- TRL import errors mentioning vllm → the MagicMock patch should cover this. Re-check that `pilot.py` Cell 0 ran.

### 0.4 Exit and clean up

```bash
exit
# back on dsmlp-login
kubectl get pods   # confirm pod is gone, or delete with `kubectl delete pod <name>`
```

## Phase 1 — full training (~8 hours)

### 1.1 Background batch launch

```bash
# from dsmlp-login.ucsd.edu (NOT inside a pod)
# K8S_TIMEOUT_SECONDS=43200 bumps the 6hr default to 12hr (max allowed)
K8S_TIMEOUT_SECONDS=43200 launch.sh -g 1 -m 32 -c 8 -B \
  -- bash -c '
    set -e
    git clone https://github.com/Trevis8688/151B_SP26_Competition.git
    cd 151B_SP26_Competition
    pip install -q -r experiments/exp_010_grpo_v2/requirements.txt
    export HF_TOKEN=hf_xxxxxx
    python experiments/exp_010_grpo_v2/train_grpo_v2.py
  '
```

This returns immediately with a pod name. **Save the pod name** — you'll need
it for log streaming.

### 1.2 Monitor progress

```bash
# stream logs from the pod (Ctrl+C to detach without killing the pod)
kubectl logs -f <pod_name>

# or just dump everything that's accumulated so far
kubectl logs <pod_name> | tail -100

# pod state
kubectl get pod <pod_name>
kubectl describe pod <pod_name>   # if it's stuck in Pending or errored
```

Expected log signals:
- `Loaded curriculum: 196 sweet-spot prompts` (Step 1 in the script)
- `Loading tokenizer + 4-bit model: Qwen/Qwen3-4B-Thinking-2507` (Step 3)
- `Train set: 196 prompts (MCQ: X, FF: Y)` (Step 4)
- `Starting GRPO training` block with steps/epoch and tokens/epoch
- Per-step `[step N] loss=...  reward=...  kl=...  completion_length=...` lines
- Around step 98 (= 49/epoch × 2 epochs): `Final adapter saved`
- Then `Loading <BASE> in torch.float16 for merge ...`
- Finally `Pushing to https://huggingface.co/TrevorDuong/qwen3-4b-thinking-grpo-v2 ...`
- `ALL DONE.`

### 1.3 If pod dies mid-run

Symptoms in `kubectl get pod <name>`:

| Status | Cause | Recovery |
|---|---|---|
| `OOMKilled` | RAM overrun (CPU side, not GPU) | Bump `-m 32` to `-m 48`, re-launch |
| `DeadlineExceeded` | Hit 12h timeout | Reduce `num_train_epochs` to 1 in config.json, re-launch (re-running from scratch is unfortunately needed — adapter is only saved at end) |
| `Error` | Pull HF check `kubectl logs <name>` for the actual exception | Likely HF push failed or CUDA error — fix and re-launch |

**Mitigation for DeadlineExceeded:** if log shows training was on step 80+ of ~98, the adapter checkpoints from `save_steps=25` are inside the pod's local disk but the pod is gone. We can't recover those. Lesson for next iteration: add a per-25-step push-to-HF callback.

### 1.4 After completion

```bash
# verify the merged model is on HF
curl -s https://huggingface.co/api/models/TrevorDuong/qwen3-4b-thinking-grpo-v2 | head -50

# clean up the DSMLP pod
kubectl delete pod <pod_name>
```

## Phase 2 — Kaggle inference

Mirror exp_009's inference notebook:
1. Update `experiments/exp_010_grpo_v2/config.json` to add the Kaggle inference block
   (model_id pointing at `TrevorDuong/qwen3-4b-thinking-grpo-v2`, plus the vLLM block).
2. Push exp_010 folder to main; refresh the `151b-experiments` Kaggle dataset.
3. In `cse151b-notebook.ipynb` Cell 5, set `EXPERIMENT = "exp_010_grpo_v2"`.
4. Save & Run All (Commit) on Kaggle T4 x2.

## Cheat sheet

```bash
# list my pods
kubectl get pods

# kill a pod
kubectl delete pod <name>

# launch options recap (relevant ones for us)
launch.sh -g 1                 # 1 GPU, 4 CPU, 16 GB
launch.sh -g 1 -c 8 -m 32      # 1 GPU, 8 CPU, 32 GB — what we want
launch.sh -g 1 -v 2080ti       # request specific GPU
launch.sh -g 1 -P Always       # force re-pull image (we don't need this)
launch.sh -g 1 -B -- cmd       # batch mode (unattended); collect via kubectl logs
launch.sh -g 1 -b              # background interactive (6hr default, can re-attach via kubesh)

# raise timeout for -B
K8S_TIMEOUT_SECONDS=43200 launch.sh ...   # 12hr max
```
