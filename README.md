# Bonsai web demo — on-device message triage (WebGPU)

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
<https://huggingface.co/spaces/webml-community/bonsai-webgpu-kernels>. The only changes
are: repointed from `Bonsai-27B-gguf` to the smallest **`Bonsai-1.7B-gguf`**, relabeled,
and framed for support-message triage (a triage instruction is prepended to each user
message). The inference engine is unchanged.

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
