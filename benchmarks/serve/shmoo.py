#!/usr/bin/env python3
"""Prefill/decode shmoo against llama-server (k3 remote backend); driven by shmoo.sh.

Every request is an exact-length token prompt: the model's chat template around a unique
session tag, a context corpus (this repository's C sources or docs) truncated to fit, and a
closing question. cache_prompt is off, so nothing is reused, and generation is forced to
GEN tokens (ignore_eos). Each answer is saved next to the timings for inspection.

Workloads: code, chat (thinking on); code_direct, copy, count (thinking off, so the answer
starts at once; copy and count are as predictable as output gets).

Usage: shmoo.py PORT OUT.tsv LEN [LEN ...]   (env REPEATS=3 GEN=256 WORKLOADS=code,chat)
Writes OUT.tsv (one row per request) and OUT.text.jsonl (the generated text).
"""
import glob, json, os, re, sys, time, urllib.request

PORT, OUT, LENS = int(sys.argv[1]), sys.argv[2], [int(a) for a in sys.argv[3:]]
REPEATS, GEN = int(os.environ.get("REPEATS", 3)), int(os.environ.get("GEN", 256))
WORKLOADS = os.environ.get("WORKLOADS", "code,chat").split(",")
OPEN = urllib.request.build_opener(urllib.request.ProxyHandler({}))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def post(path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with OPEN.open(req, timeout=timeout) as r:
        return json.load(r)


def tok(text):
    return post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]


def read(paths):
    return [open(p, encoding="utf-8", errors="ignore").read() for p in sorted(paths)]


def template(**kw):
    s = post("/apply-template", {"messages": [{"role": "user", "content": "@@DOC@@"}], **kw})["prompt"]
    head, tail = s.split("@@DOC@@")
    return tok(head), tok(tail)


HEAD, TAIL = template()
HEAD_NT, TAIL_NT = template(chat_template_kwargs={"enable_thinking": False})

code_text = "\n\n".join(read(glob.glob(os.path.join(ROOT, "src", "*", "*.c"))))
chat_turns = []
for md in read(glob.glob(os.path.join(ROOT, "docs", "*.md")) + [os.path.join(ROOT, "README.md")]):
    for sec in re.split(r"\n(?=#{2,3} )", md):
        title, _, body = sec.partition("\n")
        title, body = title.strip("# ").strip(), body.strip()
        if title and len(body) > 200:
            chat_turns.append(f"User: Can you tell me about {title.lower()}?\n\nAssistant: {body}\n")

W = {
    "code": dict(
        doc=tok(code_text),
        intro="[{tag}] My C code:\n```c\n",
        outro="\n```\nAdd `int count_words(const char *path)`, -1 on error."),
    "chat": dict(
        doc=tok("\n".join(chat_turns)),
        intro="[{tag}] Our chat so far:\n",
        outro="\n\nNew topic: plan me a two-week spring trip to Japan, with tips."),
    "code_direct": dict(
        doc=tok(code_text), think=False,
        intro="[{tag}] My C code:\n```c\n",
        outro="\n```\nAdd `int count_words(const char *path)`, -1 on error."),
    "copy": dict(
        doc=tok(code_text), think=False, min_len=512,
        intro="[{tag}] My C code:\n```c\n",
        outro="\n```\nRepeat the code above verbatim from its first line. Output only code."),
    "count": dict(
        doc=tok("\n".join(chat_turns)), think=False,
        intro="[{tag}] Notes:\n",
        outro="\n\nList the integers 1 to 400, one per line, nothing else."),
}
for name in WORKLOADS:
    print(f"{name}: context corpus {len(W[name]['doc'])} tokens", flush=True)


def prompt(name, n, tag):
    w = W[name]
    head, tail = (HEAD, TAIL) if w.get("think", True) else (HEAD_NT, TAIL_NT)
    a, b = tok(w["intro"].format(tag=tag)), tok(w["outro"])
    k = n - len(head) - len(tail) - len(a) - len(b)
    if k < 0 or k > len(w["doc"]):
        raise SystemExit(f"cannot build a {n}-token {name} prompt")
    ids = head + a + w["doc"][:k] + b + tail
    assert len(ids) == n
    return ids


def wanted(name, n):
    return n >= W[name].get("min_len", 0)


def run(name, n, tag):
    r = post("/completion", {"prompt": prompt(name, n, tag), "n_predict": GEN, "temperature": 0,
                             "ignore_eos": True, "cache_prompt": False, "stream": False})
    t = r["timings"]
    return r.get("content", ""), (t["prompt_n"], t["prompt_ms"], t["prompt_per_second"], t["predicted_n"],
                                  t["predicted_ms"], t["predicted_per_token_ms"], t["predicted_per_second"])


for name in WORKLOADS:                         # every length must build before any timing
    for n in LENS:
        if wanted(name, n):
            prompt(name, n, "9-99999-999999999")
run(WORKLOADS[0], 256, "warmup")
with open(OUT, "w") as f, open(re.sub(r"\.tsv$", "", OUT) + ".text.jsonl", "w") as ft:
    f.write("workload\tlen\trep\tprompt_n\tprefill_ms\tprefill_tok_s\tgen_n\tdecode_ms\t"
            "decode_ms_tok\tdecode_tok_s\n")
    for rep in range(1, REPEATS + 1):
        for n in LENS:
            for name in WORKLOADS:
                if not wanted(name, n):
                    continue
                text, row = run(name, n, f"{rep}-{n}-{time.time_ns() % 10**9}")
                f.write("\t".join(str(x) for x in (name, n, rep) + row) + "\n")
                f.flush()
                ft.write(json.dumps({"workload": name, "len": n, "rep": rep, "text": text}) + "\n")
                ft.flush()
                print(f"{name:11s} len {n:6d} rep {rep}: prefill {row[2]:7.1f} tok/s ({row[1] / 1e3:6.1f} s)"
                      f" | decode {row[3]} tok {row[6]:6.2f} tok/s ({row[5]:6.1f} ms/tok)", flush=True)
