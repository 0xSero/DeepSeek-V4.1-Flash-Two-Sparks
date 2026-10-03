# DeepSeek-V4.1-Flash on two DGX Sparks

Scripts to serve [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) on two NVIDIA DGX
Spark nodes with vLLM (tensor parallel 2 over the Sparks' QSFP/RoCE link), and to use it from the
[Pi coding agent](https://github.com/earendil-works/pi).

The checkpoint is [`0xSero/DeepSeek-V4.1-Flash-Spark`](https://huggingface.co/0xSero/DeepSeek-V4.1-Flash-Spark):
every backbone routed expert (layers 0-39, 15,360 experts) is re-encoded as EXL3 trellis (MUL1 codebook) at a
per-expert K2 / K3 / K5 mix, **2.77 bpw** average. Everything else (attention, shared experts, routers, Engram
tables, DSpark/MTP draft layers, vision tower, embeddings, head) is the upstream checkpoint, byte for byte. The
server image is `ghcr.io/0xsero/deepseek-v4.1-flash-spark` (vLLM with the EXL3 MoE path and the B12X Spark kernels).

## Measured

Two DGX Spark (GB10), TP2 over RoCE, FP8 KV cache, DSpark speculative decoding (7 draft tokens), Engram tables read
from disk, CUDA graphs `FULL_DECODE_ONLY`, 2 concurrent sequences. All decode runs used the model's default
sampling and stopped naturally (no output caps).

| Metric | Result |
|---|---|
| Prefill, 8k prompt | 2,050 tok/s |
| Prefill, 32k prompt | 2,061 tok/s |
| Code decode, 1 stream | ~39-41 tok/s |
| Code decode, 2 streams | 60 tok/s aggregate |
| Prose decode, 1 stream | ~29 tok/s |
| KV cache pool | 2,023,717 tokens |
| Max context per request | 262,144 |
| Vision, tool calls, reasoning | yes (smoke-tested) |
| Long context | 175k-token context recall passed (local-ai-registry lab gates, 6/6 pass) |
| Load time | ~11-13 min first boot |

Quality, full-vocabulary token-wise KL divergence against the native checkpoint (64 windows per panel, bootstrap
95% CI):

| Panel | Mean KLD (nats) | Top-1 agreement |
|---|---:|---:|
| v3.1 | 0.0742 [0.0644, 0.0865] | 0.917 |
| legacy | 0.0449 | 0.926 |

## Requirements

- 2x DGX Spark (DGX OS), one acting as **head** (serves the API on port 8000) and one as **worker**.
- A QSFP cable between them with RoCE up: each node needs at least one RDMA port in state `ACTIVE`
  (`rdma link`) whose netdev has an IPv4 address on a shared subnet (for example 10.10.10.13 and 10.10.10.11).
- Passwordless ssh from the head to the worker (key auth; use the worker's fabric address).
- Docker with the NVIDIA runtime on both nodes, your user in the `docker` group.
- ~700 GB free NVMe on the head (462 GB of weights + ~197 GB while the Engram parts are reassembled), ~470 GB on
  the worker.
- `python3`, `rsync`, `curl` on both. Node.js + npm on whatever machine runs Pi.

## Quick start

All commands run on the head unless noted. `WORKER` is the ssh target of the second Spark.

```bash
git clone https://github.com/0xSero/DeepSeek-V4.1-Flash-Two-Sparks && cd DeepSeek-V4.1-Flash-Two-Sparks
export WORKER=user@10.10.10.11          # the worker's fabric address
```

1. **Download the weights** (~462 GB) to `~/models/DeepSeek-V4.1-Flash-Spark`:

   ```bash
   scripts/download.sh
   ```

2. **Reassemble the Engram shards.** `download.sh` does this: it runs the repo's `reassemble.sh` (two Engram
   shards are stored as `.partNN` files because Hugging Face caps single files at 50 GB), checks sha256 of all six
   native shards against `sha256-manifest.txt`, then deletes the parts (`KEEP_PARTS=1` keeps them).

3. **Copy to the worker** over the fabric and verify there:

   ```bash
   scripts/copy-to-peer.sh
   ```

   (Or run `scripts/download.sh` on the worker too.)

4. **Pull the image** on both nodes. `launch.sh` pulls it if missing; to do it ahead of time:

   ```bash
   docker pull ghcr.io/0xsero/deepseek-v4.1-flash-spark@sha256:5668e35e5ee021da4b9ce88a1964caf0e082e8ca01d81d1d64b6213a55c6add5
   ```

5. **Launch** both ranks. The script detects the fabric interfaces, writes an API key to
   `~/.dsv41-two-sparks/api_key`, starts the memory guards, starts rank 1 on the worker and rank 0 here, and waits
   until the API answers (first boot ~12 min):

   ```bash
   scripts/launch.sh            # also: scripts/launch.sh status | logs [r0|r1] | stop
   ```

6. **Smoke test** (text, tool call, vision; each prints PASS/FAIL):

   ```bash
   python3 scripts/smoke.py http://127.0.0.1:8000
   python3 scripts/bench.py --out bench.json --conc 1 2   # optional speed run
   ```

7. **Connect Pi** (on any machine that can reach the head):

   ```bash
   pi/install.sh http://<head-ip>:8000/v1
   export DSV41_API_KEY=$(ssh <head> cat ~/.dsv41-two-sparks/api_key)
   pi --model dsv41-sparks/deepseek-v4.1-flash
   ```

The endpoint is OpenAI-compatible: `http://<head-ip>:8000/v1`, model `deepseek-v4.1-flash`, bearer auth with the
key above.

## Pi

`pi/install.sh` installs Pi (`npm install -g @earendil-works/pi-coding-agent`) if `pi` is not on PATH, then:

- adds or replaces only the `dsv41-sparks` provider in `~/.pi/agent/models.json` (`$PI_CODING_AGENT_DIR` if set);
  other providers are untouched and a timestamped backup is written first. See `pi/models.fragment.json`.
- appends `dsv41-sparks/deepseek-v4.1-flash` to `enabledModels` in `settings.json` only if you use that list;
  `SET_DEFAULT=1` also makes it the startup model.
- installs `pi/dsv41-sparks.ts` as a Pi extension. Pi always sends an output-token limit; the extension removes it
  for this provider so answers run to their natural end.

The model entry: reasoning on, text + image input, 262,144 context, tools via the server's `deepseek_v41` parser.
Pi thinking levels map to the server's `reasoning_effort` as minimal/low -> `low`, medium/high -> `high`,
xhigh/max -> `max`. The key is read from `$DSV41_API_KEY` at request time and is never written to Pi's config.

## Configuration

`scripts/launch.sh` reproduces the measured configuration. Main settings (override via environment):

| Setting | Value |
|---|---|
| KV cache | fp8, `--kv-cache-memory-bytes 2700000000` (KV_BYTES), block 256, SWA block 128 |
| Context / batch | `--max-model-len 262144` (MAX_LEN), `--max-num-seqs 2` (MAX_SEQS), `--max-num-batched-tokens 2048` (BATCH) |
| Speculative decoding | DSpark, 7 draft tokens (NSPEC), adaptive verification |
| Engram | tables on disk (`table_memory: disk`) |
| Kernels | B12X attention / linear / MoE, EXL3 routed experts (`ST_EXL3_PREFILL=st`) |
| CUDA graphs | `FULL_DECODE_ONLY`, `VLLM_USE_BREAKABLE_CUDAGRAPH=0` |
| All-reduce | B12X RoCE one-shot (`VLLM_ENABLE_ROCE_ALLREDUCE=1`), NCCL over RoCE for the rest |
| Images | `--limit-mm-per-prompt {"image":1}` |
| Parsers | `deepseek_v41` reasoning and tool-call parsers |
| Host | `MALLOC_ARENA_MAX=2`, memory guard floor 1 GiB (MEMGUARD_GIB) |

Paths: `MODEL_DIR` (default `~/models/DeepSeek-V4.1-Flash-Spark`, same path on the worker unless
`WORKER_MODEL_DIR`), `STATE_DIR` (default `~/.dsv41-two-sparks` on both nodes: API key, plan copy, kernel caches,
memory guard log). The HF directory is mounted read-only at `/model`; the EXL3 allocation plan
(`exl3/plan/J268-ho-v31-K5n.json`) is copied to `STATE_DIR/plans` with its bank paths rewritten to
`/model/exl3/{qn,q31,q31s}`. The repo carries no native routed-expert tensors; the EXL3 path does not read them.

## Troubleshooting

- **Memory guard.** A GB10 Spark shares one 128 GB pool between CPU and GPU. If the model drives `MemAvailable` to
  zero the host does not OOM-kill cleanly; it wedges until the hardware watchdog reboots it. `scripts/memguard.sh`
  runs on both nodes and kills the `dsv41-r*` containers when `MemAvailable` falls below 1 GiB. Log:
  `~/.dsv41-two-sparks/memguard.log`. If it fires, lower `KV_BYTES` or `GPU_UTIL` before raising the floor (steady
  state with the default config is 2-3 GiB available).
- **Fabric pinning.** vLLM, NCCL and Gloo must bind to the QSFP addresses, not WiFi, Ethernet or Tailscale. The
  launcher sets `VLLM_HOST_IP`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` and `NCCL_IB_HCA` from the ACTIVE RoCE
  ports it finds. If detection picks the wrong port, set `HEAD_IP`, `WORKER_IP`, `HEAD_IFNAME`, `WORKER_IFNAME`,
  `HEAD_HCAS`, `WORKER_HCAS` (see `ip -br a` and `rdma link`).
- **WiFi-only nodes.** If a Spark's only internet path is WiFi, downloads are slow; download once on the node with
  the best uplink and use `copy-to-peer.sh` over the fabric. Clients (Pi) can reach the head over any network; only
  the rank-to-rank traffic needs the fabric.
- **First boot takes ~12 minutes**: weights load, EXL3 banks (~15 s per layer), B12X kernel compile (cached in
  `STATE_DIR/cache` after the first run) and CUDA graph capture. `launch.sh` waits up to 45 minutes and prints the
  last log lines of both ranks if one exits. `scripts/launch.sh logs r1` follows the worker.
- **`missing .../model-00005-of-00006.safetensors`**: the Engram shards were not reassembled; re-run
  `scripts/download.sh` (it resumes and skips finished steps).
- **Pi shows no model**: `DSV41_API_KEY` must be set in the shell that starts Pi; run `/model` to reload.

## Repository layout

| Path | What it does |
|---|---|
| `scripts/download.sh` | HF download, Engram reassembly, sha256 verification |
| `scripts/copy-to-peer.sh` | rsync the checkpoint to the worker over the fabric, verify there |
| `scripts/launch.sh` | start / stop / status / logs for both ranks |
| `scripts/memguard.sh` | host memory guard (started by `launch.sh`) |
| `scripts/smoke.py` | text, tool-call and vision checks |
| `scripts/bench.py` | prefill / decode / long-context speed bench, no output caps |
| `pi/install.sh` | install Pi and register the provider |
| `pi/models.fragment.json`, `pi/dsv41-sparks.ts` | Pi provider entry and request adapter |

## Credits and licences

- DeepSeek-V4.1-Flash: DeepSeek, MIT. The quantized checkpoint carries the same licence.
- vLLM: Apache-2.0. Spark serving stack (vLLM fork, B12X kernels): Local Inference Lab, Apache-2.0.
- ExLlamaV3 / EXL3 trellis encoder and kernels: turboderp, MIT.
- Pi coding agent: Earendil Works, MIT.
- The scripts in this repository: MIT (see `LICENSE`).
