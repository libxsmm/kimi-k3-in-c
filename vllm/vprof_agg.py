"""Aggregate K3CPU_PROF per-rank JSONs (k3cpu/prof.py): python vprof_agg.py DIR"""

import glob
import json
import re
import sys

CATS = [
    ("TP/EP communication", r"^comm\."),
    ("routed experts: MXFP4 dequant", r"^moe\.experts\.dequant$"),
    ("routed experts: matmul+SiTU+index", r"^moe\.experts\.kernel$"),
    ("routed experts: FusedMoE glue", r"^moe\.experts$|^moe\.experts\._"),
    ("shared expert", r"^moe\.shared_experts"),
    ("router + MoE latent down", r"^moe\.(gate|routed_expert_down_proj|routed_expert_norm)"),
    ("MoE latent up", r"^moe\.(routed_expert_up_proj|routed_output)"),
    ("KDA projections", r"^kda\.(in_proj|f_b|g_b|f_a|g_a|b_proj|q_proj|k_proj|v_proj)"),
    ("KDA core (conv+gate+recurrence)", r"^kda\.core"),
    ("KDA output (o_norm+o_proj)", r"^kda\.(o_norm|o_proj)"),
    ("MLA (proj + core + out)", r"^mla"),
    ("dense MLP (layer 0)", r"^dense"),
    ("AttnRes + norms", r"^attnres$|norm$"),
    ("embedding", r"^embed_tokens$"),
    ("lm_head + logits", r"^logits$"),
    ("runner: sample", r"^runner\.sample$"),
    ("runner: execute_model overhead", r"^runner\.execute$"),
    ("python module glue", r"^(layer|model|moe|kda|mla|dense)$|."),
]


def main(d):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/prof_r*.json"))]
    runs.sort(key=lambda r: r["rank"])
    print(f"{len(runs)} ranks; steps per rank: {[len(r['steps']) for r in runs]}")
    for r in runs:
        s = r["steps"]
        busy = sum(x["busy"] for x in s) / len(s)
        gap = sum(x["gap"] for x in s) / len(s)
        exp = r["regions"].get("moe.experts.dequant", [0, 0, 0])
        print(f"  rank {r['rank']:2d} {r['host']}: wall {1e3 * (busy + gap):7.1f} ms "
              f"(idle {1e3 * gap:5.1f}), dequant calls/step {exp[0]:5.1f} "
              f"{1e3 * exp[2]:7.1f} ms, comm "
              f"{1e3 * sum(v[2] for k, v in r['regions'].items() if k.startswith('comm.')):7.1f} ms")
    names = sorted({k for r in runs for k in r["regions"]})
    mean = {k: sum(r["regions"].get(k, [0, 0, 0, 0])[2] for r in runs) / len(runs) for k in names}
    walls = [sum(x["busy"] + x["gap"] for x in r["steps"]) / len(r["steps"]) for r in runs]
    wall = sum(walls) / len(walls)
    print(f"\nmean wall/step {1e3 * wall:.1f} ms\n\nCategories (exclusive ms/step, mean over ranks; min..max):")
    left = set(names)
    total = 0.0
    for cat, pat in CATS:
        ks = [k for k in sorted(left) if re.search(pat, k)]
        left -= set(ks)
        per = [sum(r["regions"].get(k, [0, 0, 0, 0])[2] for k in ks) for r in runs]
        m = sum(per) / len(per)
        total += m
        print(f"  {cat:36s} {1e3 * m:8.1f}  ({1e3 * min(per):7.1f} .. {1e3 * max(per):7.1f})  "
              f"{100 * m / wall:5.1f}%")
    idle = wall - total
    print(f"  {'worker idle (between steps)':36s} {1e3 * idle:8.1f}  {100 * idle / wall:5.1f}%")
    print("\nTop regions (exclusive ms/step, mean; calls/step; KB/call):")
    for k in sorted(names, key=lambda k: -mean[k])[:40]:
        c = sum(r["regions"].get(k, [0] * 4)[0] for r in runs) / len(runs)
        b = sum(r["regions"].get(k, [0] * 4)[3] for r in runs) / len(runs)
        print(f"  {k:60s} {1e3 * mean[k]:8.2f} {c:7.1f} {b / max(c, 1e-9) / 1024:8.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
