"""K3CPU_PROF=1: per-region wall-time profiler for Kimi K3 decode on vLLM CPU.

Regions nest; each keeps inclusive and exclusive time. Collectives are keyed by
their enclosing region (comm.all_reduce@kda.o_proj). Per request, steps after
the first are averaged and written to $K3CPU_PROF_DIR/prof_r{rank}.{txt,json}
after every step.
"""

import json
import os
import re
import socket
import time
from functools import wraps

import torch

_DIR = os.environ.get("K3CPU_PROF_DIR", "/scratch/aheineck/k3work/vprof")
_now = time.perf_counter

_stack: list[list] = []  # [name, start, child_time, record_function or None]
_cur: dict[str, list] = {}  # step: name -> [calls, incl, excl, bytes]
_acc: dict[str, list] = {}  # request: same, summed over counted steps
_steps: list[dict] = []
_st = {"req": 0, "start": None, "prev_end": None, "count": False, "ntok": 0}


def _enter(name):
    rf = None
    if torch.autograd._profiler_enabled():
        rf = torch.profiler.record_function(name)
        rf.__enter__()
    _stack.append([name, _now(), 0.0, rf])


def _exit(nbytes=0):
    name, t0, child, rf = _stack.pop()
    dt = _now() - t0
    if rf is not None:
        rf.__exit__(None, None, None)
    if _stack:
        _stack[-1][2] += dt
    e = _cur.setdefault(name, [0, 0.0, 0.0, 0])
    e[0] += 1
    e[1] += dt
    e[2] += dt - child
    e[3] += nbytes


def timed(fn, name):
    @wraps(fn)
    def w(*a, **k):
        _enter(name)
        try:
            return fn(*a, **k)
        finally:
            _exit()

    return w


def _comm(fn, op):
    @wraps(fn)
    def w(self, x, *a, **k):
        parent = _stack[-1][0] if _stack else "-"
        _enter(f"comm.{op}@{parent}")
        try:
            return fn(self, x, *a, **k)
        finally:
            _exit(x.numel() * x.element_size() if isinstance(x, torch.Tensor) else 0)

    return w


def _rank():
    try:
        return torch.distributed.get_rank()
    except Exception:
        return int(os.environ.get("RANK", 0))


def _write():
    n = len(_steps)
    if n == 0:
        return
    busy = sum(s["busy"] for s in _steps) / n
    gap = sum(s["gap"] for s in _steps) / n
    wall = busy + gap
    rows = sorted(_acc.items(), key=lambda kv: -kv[1][2])
    r = _rank()
    host = socket.gethostname().split(".")[0]
    os.makedirs(_DIR, exist_ok=True)
    with open(f"{_DIR}/prof_r{r}.txt", "w") as f:
        print(f"rank {r} host {host} request {_st['req']} decode steps {n} "
              f"(first step of the request excluded)", file=f)
        print(f"wall/step {wall * 1e3:.1f} ms = busy {busy * 1e3:.1f} + idle {gap * 1e3:.1f}",
              file=f)
        print("steps busy ms: " + " ".join(f"{s['busy'] * 1e3:.0f}" for s in _steps), file=f)
        print(f"{'region':58s} {'calls':>7s} {'excl ms':>9s} {'incl ms':>9s} {'excl%':>6s} "
              f"{'KB/call':>8s}", file=f)
        for k, (c, inc, exc, b) in rows:
            print(f"{k:58s} {c / n:7.1f} {exc / n * 1e3:9.2f} {inc / n * 1e3:9.2f} "
                  f"{100 * exc / n / wall:6.1f} {b / max(c, 1) / 1024:8.1f}", file=f)
    with open(f"{_DIR}/prof_r{r}.json", "w") as f:
        json.dump({"rank": r, "host": host, "request": _st["req"], "steps": _steps,
                   "regions": {k: [v[0] / n, v[1] / n, v[2] / n, v[3] / n]
                               for k, v in _acc.items()}}, f)


def _step_begin(sched):
    new = len(getattr(sched, "scheduled_new_reqs", ()) or ())
    ntok = int(getattr(sched, "total_num_scheduled_tokens", 0) or 0)
    if new:
        _st["req"] += 1
        _acc.clear()
        _steps.clear()
    _cur.clear()
    _st["count"] = new == 0 and ntok > 0
    _st["ntok"] = ntok
    _st["start"] = _now()


def _step_end():
    t = _now()
    if _st["start"] is None:
        return
    busy = t - _st["start"]
    gap = _st["start"] - _st["prev_end"] if _st["prev_end"] is not None else 0.0
    _st["prev_end"] = t
    _st["start"] = None
    if not _st["count"]:
        return
    _steps.append({"busy": busy, "gap": gap, "ntok": _st["ntok"]})
    for k, v in _cur.items():
        a = _acc.setdefault(k, [0, 0.0, 0.0, 0])
        for i in range(4):
            a[i] += v[i]
    _write()


def _layer_names(model) -> dict:
    """Module -> region name, with the layer index dropped and self_attn / mlp
    renamed by type (kda, mla, moe, dense)."""
    names = {}
    for path, mod in model.named_modules():
        m = re.match(r"(?:.*\.)?layers\.(\d+)(?:\.(.*))?$", path)
        if not m:
            if path in ("embed_tokens", "norm"):
                names[mod] = path
            continue
        layer = model.layers[int(m.group(1))]
        rest = m.group(2)
        if rest is None:
            names[mod] = "layer"
            continue
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.Identity)):
            continue
        parts = rest.split(".")
        if parts[0] == "self_attn":
            cls = type(layer.self_attn).__name__
            parts[0] = "kda" if "Delta" in cls else "mla" if "Latent" in cls else "attn"
        elif parts[0] in ("mlp", "block_sparse_moe"):
            parts[0] = "moe" if "MoE" in type(getattr(layer, parts[0])).__name__ else "dense"
        names[mod] = ".".join(parts)
    return names


def install() -> None:
    import vllm.models.kimi_k3.nvidia.mla as mla
    import vllm.models.kimi_k3.nvidia.model as model
    from vllm.distributed.device_communicators.cpu_communicator import CpuCommunicator
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    from . import ops

    model.attn_res = timed(model.attn_res, "attnres")
    KimiK3DeltaAttention._forward = timed(KimiK3DeltaAttention._forward, "kda.core")
    mla.MultiHeadLatentAttention._attention = timed(
        mla.MultiHeadLatentAttention._attention, "mla.core"
    )
    ops.moe_mxfp4_situ = timed(ops.moe_mxfp4_situ, "moe.experts.kernel")
    ops.mxfp4_dequant = timed(ops.mxfp4_dequant, "moe.experts.dequant")
    ops.kda_recurrence = timed(ops.kda_recurrence, "kda.core.recurrence")
    ops.kda_conv = timed(ops.kda_conv, "kda.core.conv")

    for op in ("all_reduce", "all_gather", "gather", "dispatch", "combine",
               "dispatch_router_logits", "reduce_scatter"):
        if hasattr(CpuCommunicator, op):
            setattr(CpuCommunicator, op, _comm(getattr(CpuCommunicator, op), op))

    cls = model.KimiLinearModel
    orig_init = cls.__init__

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        for mod, name in _layer_names(self).items():
            mod.forward = timed(mod.forward, name)

    cls.__init__ = init
    cls.forward = timed(cls.forward, "model")
    lm = model.KimiLinearForCausalLM
    lm.compute_logits = timed(lm.compute_logits, "logits")

    orig_exec = CPUModelRunner.execute_model
    orig_sample = CPUModelRunner.sample_tokens

    def execute_model(self, scheduler_output, *a, **k):
        _step_begin(scheduler_output)
        _enter("runner.execute")
        try:
            out = orig_exec(self, scheduler_output, *a, **k)
        finally:
            _exit()
        if out is not None:
            _step_end()
        return out

    def sample_tokens(self, *a, **k):
        _enter("runner.sample")
        try:
            return orig_sample(self, *a, **k)
        finally:
            _exit()
            _step_end()

    CPUModelRunner.execute_model = execute_model
    CPUModelRunner.sample_tokens = sample_tokens
