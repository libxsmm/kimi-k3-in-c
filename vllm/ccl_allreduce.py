"""torch.distributed allreduce test: python ccl_allreduce.py [ccl|gloo] [bf16|fp32]; env from srun."""

import os
import sys
import time

import torch
import torch.distributed as dist

backend = sys.argv[1] if len(sys.argv) > 1 else "ccl"
dtname = sys.argv[2] if len(sys.argv) > 2 else "bf16"
dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtname]
if backend == "ccl":
    import oneccl_bindings_for_pytorch  # noqa: F401

rank = int(os.environ["SLURM_PROCID"])
world = int(os.environ["SLURM_NTASKS"])
os.environ.setdefault("MASTER_PORT", "29601")
dist.init_process_group(backend, rank=rank, world_size=world)

for n in (7168, 7168 * 16, 1 << 22):
    x = torch.full((n,), float(rank + 1), dtype=dtype)
    dist.all_reduce(x)
    assert float(x[0]) == world * (world + 1) / 2, float(x[0])
    iters = 200 if n < 1 << 20 else 20
    dist.barrier()
    t = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    dt = (time.perf_counter() - t) / iters
    if rank == 0:
        print(f"{backend} np={world} {dtname} n={n:>8}: {dt * 1e6:9.1f} us", flush=True)
dist.destroy_process_group()
