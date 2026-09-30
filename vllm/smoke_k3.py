"""Kimi K3 on vLLM CPU: smoke test (dummy weights) or greedy check vs the C engine.

  python smoke_k3.py --layers 4                           # dummy weights
  python smoke_k3.py --model /scratch/.../k3model_l8 --load-format auto \
      --ids 19180 --gen 10 --ref /scratch/aheineck/k3work/l8.json
"""

import argparse
import json
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/scratch/aheineck/kimik3/k3model")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--load-format", default="dummy")
    ap.add_argument("--ids", default="19180")
    ap.add_argument("--gen", type=int, default=4)
    ap.add_argument("--ref", default=None, help="C engine --out json to compare")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--ep", action="store_true", help="expert parallel over the TP ranks")
    args = ap.parse_args()

    overrides = {}
    if args.layers is not None:
        overrides = {"text_config": {"num_hidden_layers": args.layers}}
    t0 = time.time()
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enable_expert_parallel=args.ep,
        load_format=args.load_format,
        hf_overrides=overrides,
        dtype=args.dtype,
        max_model_len=1024,
        max_num_seqs=1,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    t1 = time.time()
    prompt = TokensPrompt(prompt_token_ids=[int(x) for x in args.ids.split(",")])
    sp = SamplingParams(
        max_tokens=args.gen, temperature=0, logprobs=3, prompt_logprobs=3, ignore_eos=True
    )
    res = llm.generate([prompt], sp)[0]
    out = res.outputs[0]
    t2 = time.time()
    ids = list(out.token_ids)
    print(f"OUT {ids}")
    print(f"TIME load {t1 - t0:.1f} s, generate {t2 - t1:.2f} s ({args.gen} tokens)")
    for i, lp in enumerate(res.prompt_logprobs or []):
        if lp:
            top = sorted(lp.items(), key=lambda kv: -kv[1].logprob)
            print("PLP", i, [(t, round(v.logprob, 4)) for t, v in top])
    for i, lp in enumerate(out.logprobs or []):
        top = sorted(lp.items(), key=lambda kv: -kv[1].logprob)
        print("LP", i, [(t, round(v.logprob, 4)) for t, v in top])
    if args.ref:
        ref = json.load(open(args.ref))["generated_ids"][: len(ids)]
        n = next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), len(ref))
        print(f"REF {ref}\nMATCH {n}/{len(ref)} leading tokens")


if __name__ == "__main__":
    main()
