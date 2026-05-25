"""
exp_034 — dev-probe gate evaluator.

Loads the pass-2 base in fp16 + the probe LoRA adapter, runs transformers
`generate` over data/splits/dev.jsonl (200q), scores MCQ (letter) + free-form
(judger), and checks the pre-committed probe gate from config.json.

WHY HF generate, not vLLM: vLLM-on-DSMLP is documented-brittle for Qwen3 (the
reason exp_010 abandoned vLLM for training). 200 questions is within CLAUDE.md's
"<=200 q spot-check is fine for HF generate" guidance, and keeping the gate off
vLLM means it runs in the SAME training venv — no second venv on the
safety-critical path. Cost: ~30-60 min vs ~10 min for vLLM. Acceptable: the
probe phase still completes inside the advisor's ~3h "catch forgetting early"
window.

Prompts are imported from experiments/exp_017_pass2_stage1/prompts.py so the
context is byte-identical to the exp_017 dev baseline the gate was calibrated
against. Generation is greedy-ish (T=0.6, top_p=0.95, top_k=20) to match.

Exit 0 = gate PASS. Exit 2 = gate FAIL. The launcher keys off this.
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
# A5000 (24GB) KV-cache budget: Qwen3-4B fp16 ~147KB/token. batch=8 @ ~5k tokens
# (~1k prompt + 4k gen) ≈ 6GB KV + 8GB weights = ~14GB. batch=16 @ 8192 gen would
# be ~27GB and OOM. Keep batch modest; this is a probe gate, not a throughput run.
BATCH_SIZE = 8


def _load_prompts(prompts_path: Path):
    spec = importlib.util.spec_from_file_location("exp017_prompts", str(prompts_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_prompt(question, options, P):
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return P.SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return P.SYSTEM_PROMPT_MATH, question


def _extract_letter(text):
    m = re.search(r"\\boxed\{([A-Za-z])\}", text or "")
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", (text or "").upper())
    return matches[-1] if matches else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.json"))
    ap.add_argument("--adapter_dir", required=True, help="probe adapter dir (peft save_pretrained)")
    ap.add_argument("--dev", default=str(REPO_ROOT / "data/splits/dev.jsonl"))
    ap.add_argument("--prompts", default=str(REPO_ROOT / "experiments/exp_017_pass2_stage1/prompts.py"))
    ap.add_argument("--judger_dir", default=str(REPO_ROOT))
    ap.add_argument("--out", default=str(Path(__file__).parent / "dev_probe_responses.jsonl"))
    # 4096 (not the 8192 inference budget): probe gate detects forgetting + gross
    # format breakage, where 4096 is ample. Keeps KV cache in A5000 budget at batch=8.
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    gate = cfg["probe_gate"]
    base_model = cfg["base_model_id"]

    # --- load base fp16 + adapter ---
    print(f"[load] base={base_model} (fp16) + adapter={args.adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generate
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    model.eval()

    # --- data + prompts ---
    P = _load_prompts(Path(args.prompts))
    dev = [json.loads(l) for l in open(args.dev)]
    n_mcq = sum(bool(d.get("options")) for d in dev)
    print(f"[data] dev={len(dev)} ({n_mcq} MCQ, {len(dev)-n_mcq} FF)")

    rendered = []
    for item in dev:
        system, user = _build_prompt(item["question"], item.get("options"), P)
        fewshot = P.FEWSHOT_MCQ if item.get("options") else P.FEWSHOT_MATH
        messages = [{"role": "system", "content": system}, *fewshot, {"role": "user", "content": user}]
        rendered.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        ))

    # --- batched generate ---
    responses = []
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
        pad_token_id=tokenizer.pad_token_id,
    )
    for bstart in range(0, len(rendered), BATCH_SIZE):
        batch = rendered[bstart:bstart + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        gen = out[:, enc["input_ids"].shape[1]:]
        responses.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        print(f"[infer] {min(bstart + BATCH_SIZE, len(rendered))}/{len(rendered)}", flush=True)

    responses = [r.strip() for r in responses]
    with open(args.out, "w") as f:
        for item, resp in zip(dev, responses):
            f.write(json.dumps({
                "id": item["id"], "is_mcq": bool(item.get("options")), "response": resp,
            }) + "\n")
    print(f"[infer] wrote {args.out}")

    # --- score ---
    if args.judger_dir not in sys.path:
        sys.path.insert(0, args.judger_dir)
    from judger import Judger
    judger = Judger(strict_extract=False)

    mcq_c = mcq = ff_c = ff = boxed_ok = 0
    for item, resp in zip(dev, responses):
        if BOXED_RE.search(resp or ""):
            boxed_ok += 1
        gold = item["answer"]
        if item.get("options"):
            mcq += 1
            mcq_c += int(_extract_letter(resp) == str(gold).strip().upper())
        else:
            ff += 1
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                ff_c += int(bool(judger.auto_judge(
                    pred=resp, gold=gold_list, options=[[]] * len(gold_list))))
            except Exception:
                pass

    mcq_pct = mcq_c / mcq * 100 if mcq else 0.0
    ff_pct = ff_c / ff * 100 if ff else 0.0
    extract_pct = boxed_ok / len(dev) * 100 if dev else 0.0

    print("=" * 52)
    print(f"  MCQ          {mcq_c:3d}/{mcq:3d}  {mcq_pct:6.2f}%   (gate >= {gate['mcq_dev_min_pct']})")
    print(f"  Free-form    {ff_c:3d}/{ff:3d}  {ff_pct:6.2f}%   (gate >= {gate['ff_dev_min_pct']})")
    print(f"  boxed rate   {boxed_ok:3d}/{len(dev):3d}  {extract_pct:6.2f}%   (gate >= {gate['boxed_extraction_min_pct']})")
    print("=" * 52)

    fails = []
    if mcq_pct < gate["mcq_dev_min_pct"]:
        fails.append(f"MCQ {mcq_pct:.2f} < {gate['mcq_dev_min_pct']}")
    if ff_pct < gate["ff_dev_min_pct"]:
        fails.append(f"FF {ff_pct:.2f} < {gate['ff_dev_min_pct']}")
    if extract_pct < gate["boxed_extraction_min_pct"]:
        fails.append(f"extraction {extract_pct:.2f} < {gate['boxed_extraction_min_pct']}")

    if not fails:
        print("PROBE GATE: PASS — proceed to full training.")
        sys.exit(0)
    print("PROBE GATE: FAIL — " + "; ".join(fails))
    print("ABORT full training. Likely MCQ forgetting or format drift — revise data plan.")
    sys.exit(2)


if __name__ == "__main__":
    main()
