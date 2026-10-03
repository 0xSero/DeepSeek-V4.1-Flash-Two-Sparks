#!/usr/bin/env python3
"""Speed bench for the 2x Spark DeepSeek-V4.1-Flash server (OpenAI API, bearer key). No output caps anywhere.

  prefill  : fresh random-token prompts (unique per rep, no prefix-cache hits) via /v1/completions, stream;
             tok/s = prompt tokens / TTFT. The stream is closed after the first token (measures prefill only).
  decode   : C simultaneous chat requests (prose or code prompts), each runs to its natural stop.
             per-stream tok/s = (n-1)/(t_last - t_first); aggregate = sum(n) / (max t_last - min t_first).
  ctx      : real text (Python stdlib source) truncated to N tokens + a question; one request, natural stop;
             prefill = N / TTFT, decode as above.
Usage: bench.py --url http://127.0.0.1:8000 --out run.json [--prefill 8192 32768]
       [--conc 1 2 4] [--kinds prose code] [--ctx 8192 65536 262000] [--reps 3] [--dec-reps 2]
"""
import argparse, glob, json, os, random, statistics, sysconfig, threading, time, urllib.request

PROSE = ["Explain in detail how TCP congestion control works, including slow start and fast recovery.",
         "Describe the history of the printing press and its effects on European society.",
         "Explain how vaccines train the immune system, step by step.",
         "Describe how a jet engine produces thrust, covering each stage."]
CODE = ["Write a Python implementation of a thread-safe LRU cache with TTL expiry, with tests.",
        "Write a Rust function that parses an INI file into a nested HashMap, with error handling and tests.",
        "Implement Dijkstra's algorithm in TypeScript with a binary heap, plus example usage.",
        "Write a Go HTTP middleware that rate-limits per client IP using a token bucket, with tests."]


def req(url, path, key, payload, first_only=False, timeout=7200):
    r = urllib.request.Request(url + path, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time(); first = None; last = None; n = 0; usage = None; finish = None
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            d = json.loads(body)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("text") or (ch.get("delta") or {}).get("content") or \
                        (ch.get("delta") or {}).get("reasoning_content") or (ch.get("delta") or {}).get("reasoning")
                if delta:
                    now = time.time(); first = first or now; last = now; n += 1
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if first_only and first:
                break
    ct = (usage or {}).get("completion_tokens") or n
    pt = (usage or {}).get("prompt_tokens")
    return {"t0": t0, "ttft": (first - t0) if first else None, "t_first": first, "t_last": last,
            "completion_tokens": ct, "prompt_tokens": pt, "chunks": n, "finish": finish}


SAMPLING = {}   # chat: model generation defaults (greedy prose fell into a repetition loop in S010); --greedy sets temperature 0


def model_id(url, key):
    r = urllib.request.Request(url + "/v1/models", headers={"Authorization": f"Bearer {key}"})
    return json.load(urllib.request.urlopen(r, timeout=30))["data"][0]["id"]


def prefill(a, key, model, n, rep):
    rng = random.Random(hash((n, rep, time.time())))
    ids = [rng.randrange(1000, 100000) for _ in range(n)]
    r = req(a.url, "/v1/completions", key, {"model": model, "prompt": ids, "stream": True,
                                            "stream_options": {"include_usage": True}, "temperature": 0}, first_only=True)
    r["tok_s"] = n / r["ttft"]
    return r


def chat(a, key, model, text):
    return req(a.url, "/v1/chat/completions", key, {"model": model, "messages": [{"role": "user", "content": text}],
                                                    "stream": True, "stream_options": {"include_usage": True},
                                                    **SAMPLING})


def decode(a, key, model, C, kind):
    prompts = (PROSE if kind == "prose" else CODE)
    out = [None] * C
    def run(i):
        out[i] = chat(a, key, model, prompts[i % len(prompts)] + f" (variant {i}-{time.time():.0f})")
    th = [threading.Thread(target=run, args=(i,)) for i in range(C)]
    [t.start() for t in th]; [t.join() for t in th]
    per = [(o["completion_tokens"] - 1) / (o["t_last"] - o["t_first"]) for o in out if o["t_last"] and o["t_last"] > o["t_first"]]
    agg = sum(o["completion_tokens"] for o in out) / (max(o["t_last"] for o in out) - min(o["t_first"] for o in out))
    return {"C": C, "kind": kind, "per_stream": per, "per_stream_median": statistics.median(per) if per else None,
            "aggregate": agg, "reqs": out}


def ctx_text(n_tok):
    src = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob(sysconfig.get_paths()["stdlib"] + "/**/*.py", recursive=True))[:4000])
    return src[: int(n_tok * 3.2)]


def ctx(a, key, model, n):
    text = ctx_text(n) + "\n\nQuestion: name three functions defined in the code above and say what each does."
    o = chat(a, key, model, text)
    o["target_tokens"] = n
    o["prefill_tok_s"] = (o["prompt_tokens"] or n) / o["ttft"]
    o["decode_tok_s"] = (o["completion_tokens"] - 1) / (o["t_last"] - o["t_first"]) if o["t_last"] > o["t_first"] else None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--key-file", default=os.path.expanduser("~/.dsv41-two-sparks/api_key"),
                    help="ignored when DSV41_API_KEY is set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--prefill", type=int, nargs="*", default=[8192, 32768])
    ap.add_argument("--conc", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--kinds", nargs="*", default=["prose", "code"])
    ap.add_argument("--ctx", type=int, nargs="*", default=[])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dec-reps", type=int, default=2)
    ap.add_argument("--greedy", action="store_true", help="temperature 0 for chat requests")
    a = ap.parse_args()
    key = os.environ.get("DSV41_API_KEY") or open(a.key_file).read().strip()
    if a.greedy:
        SAMPLING["temperature"] = 0
    model = model_id(a.url, key)
    res = {"label": a.label, "model": model, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prefill": {}, "decode": [], "ctx": []}
    def save():
        json.dump(res, open(a.out, "w"), indent=1)
    chat(a, key, model, "Say hello.")  # warm-up, not recorded
    for n in a.prefill:
        rs = [prefill(a, key, model, n, i) for i in range(a.reps)]
        res["prefill"][n] = {"median": statistics.median(r["tok_s"] for r in rs), "runs": rs}
        print(f"prefill {n}: {res['prefill'][n]['median']:.1f} tok/s", flush=True); save()
    for kind in a.kinds:
        for C in a.conc:
            for rep in range(a.dec_reps):
                d = decode(a, key, model, C, kind); d["rep"] = rep; res["decode"].append(d)
                print(f"decode {kind} C{C} rep{rep}: per-stream {d['per_stream_median']:.2f} agg {d['aggregate']:.2f} tok/s "
                      f"(tokens {[o['completion_tokens'] for o in d['reqs']]}, finish {[o['finish'] for o in d['reqs']]})", flush=True)
                save()
    for n in a.ctx:
        o = ctx(a, key, model, n); res["ctx"].append(o)
        print(f"ctx {n}: prompt {o['prompt_tokens']} prefill {o['prefill_tok_s']:.1f} tok/s ttft {o['ttft']:.1f}s "
              f"decode {o['decode_tok_s']} tok/s, {o['completion_tokens']} tokens, finish {o['finish']}", flush=True)
        save()
    save()


if __name__ == "__main__":
    main()
