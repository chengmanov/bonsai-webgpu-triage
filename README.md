# sysf.io on-device demo — Bonsai-1.7B on WebGPU

Two sysf.io-branded single-file pages built from the working engine by `_brand.py`
(run `python _brand.py` after changing `index.html` or the branding):

- **`sysf-triage.html`** (deployed as the root page) — support-message triage
- **`sysf-chat.html`** (deployed as `chat.html`) — free multi-turn chat

`index.html` in this folder is the unbranded working base (engine source of truth).

# Base demo — on-device message triage (WebGPU)

A single-file, self-contained web app that runs the **smallest 1-bit Bonsai model**
(`Bonsai-1.7B`, Q1_0, ~237 MB) **100% in the browser on WebGPU** — no server. Paste a
support message and the model returns a triage: **Category · Urgency · Summary**. The
model is downloaded once from the Hugging Face CDN, cached, and executed locally with
custom WebGPU (WGSL) kernels for the 1-bit format.

## Run it

Serve the folder over HTTP (not `file://`) and open it in a **WebGPU-capable browser**
(Chrome/Edge 121+ on desktop, with a working GPU):

```bash
cd web-demo
python -m http.server 8000
# open http://localhost:8000  in Chrome/Edge on a machine with a GPU
```

First run downloads `Bonsai-1.7B-Q1_0.gguf` (~237 MB) and caches it; later runs are
instant and offline. Requires WebGPU **and** WebGL (the landing animation uses WebGL).

> Note: a headless/GPU-less environment (e.g. a CI runner or a VM without GPU
> passthrough) has no WebGPU adapter and will not run this — test on real GPU hardware.

## How it works

- **Runtime:** a bespoke WebGPU LLM engine (Qwen3-style architecture with custom WGSL
  kernels for the 1-bit quant), which parses the GGUF and runs inference on the GPU.
- **Model:** `prism-ml/Bonsai-1.7B-gguf` → `Bonsai-1.7B-Q1_0.gguf`, streamed from the HF
  CDN and cached in-browser.
- **Triage:** each pasted message is wrapped with an instruction asking the model to
  reply with exactly `Category: … / Urgency: … / Summary: …`.

## Provenance & how this was built

This page is **adapted** from the open **"Bonsai WebGPU Kernels"** demo by
webml-community / Xenova (MIT):
<https://huggingface.co/spaces/webml-community/bonsai-webgpu-kernels>, which was built
for the 27B `qwen35` hybrid architecture. Because the smaller Bonsai models use the
plain **`qwen3`** architecture, the engine was **ported**: the GGUF config path reads
qwen3 metadata (all layers full-attention, rotary_dim synthesized), the per-layer norm
tensor name maps to `ffn_norm.weight`, the attention output gate was removed from the
decode QKV kernel + manifest (plain `[q;k;v]` packing) and the scalar attention
fallback, and each message is triaged independently (no chat-history accumulation).
Measured on a desktop GPU: ~114 tok/s, TTFT ~300 ms.

### Earlier CPU-WASM exploration (superseded)

Before choosing WebGPU, we proved the ternary `Q2_0` quant of `Ternary-Bonsai-1.7B` can
be decoded in the browser via a WASM build of the **PrismML-Eng/llama.cpp** prism fork
(compiled with Emscripten, `-DLLAMA_WASM_MEM64=OFF`, wllama compat runtime). It loads and
decodes correctly but single-threaded CPU inference is too slow for a good demo, so we
moved to WebGPU. That build lives outside this repo (`~/wllama-build`).

## Credits & license

- **Model:** Bonsai-1.7B (Q1_0) — © **Prism ML**, **Apache-2.0**
  ([model card](https://huggingface.co/prism-ml/Bonsai-1.7B-gguf) ·
  [prismml.com](https://prismml.com)).
- **Engine/UI:** adapted from **webml-community / Xenova**, **MIT**.
- **This adaptation:** Apache-2.0 (see repository `LICENSE`).

```
@techreport{bonsai,
  title  = {Bonsai: 1-bit Language Models},
  author = {Prism ML}, year = {2026}, url = {https://prismml.com} }
```
