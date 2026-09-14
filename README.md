# RenderBench

A benchmark and evaluation harness for **LLM-generated GPU rendering kernels** — testing whether coding agents that succeed at ML-operator kernels (GEMM, attention) transfer to a fundamentally different class of GPU work: memory-bound, order-dependent, perceptually-correct rendering.

## Why this exists

Kernel-generation benchmarks (KernelBench, CudaForge, GEAK) target neural-network operators exclusively. None target rendering — despite rendering being the bottleneck in an increasingly common pipeline: **visual-text compression**, where documents are rasterized to images and read by a vision-language model to cut token costs. Glyph, the reference VTC implementation, flags its own rendering step as having "significant room for acceleration." Nobody has pointed a kernel-writing agent at it.

Rendering also breaks the correctness assumptions kernel-generation benchmarks rely on. ML operators have one correct numerical answer, checked by exact or tolerance match. Rendering does not — anti-aliased output legitimately differs between valid implementations, so correctness has to be *perceptual*. That turns out to be harder than it looks.

## Key finding: the standard perceptual metric fails silently

SSIM is the default choice for "do these images look the same." Calibrated against five known-good/known-bad implementations, it gets the wrong answer on the case that matters most:

| Implementation | SSIM | Verdict |
|---|---|---|
| Valid supersampled render | 0.99928 | should pass |
| Renders with one glyph silently deleted | **0.99930** | should fail |

**The renderer that drops content scores higher than the one that doesn't.** For a VTC pipeline, silently losing text is the single worst failure mode — the text is the entire payload. This isn't a synthetic artifact: the identical pattern reproduced live, three separate times, across three independent code generators (a hand-written Triton kernel, Claude Opus 5, and a NIM-hosted model), each producing a variant of the same box-boundary or phase-discarding error that plain SSIM would have accepted.

The oracle used throughout is a conjunction — an SSIM floor **and** a bounded worst-pixel error **and** a bounded bad-pixel fraction — calibrated so all five probes separate correctly before any experiment runs.

## Results

**CPU track** (agent-written NumPy, offline replay): 6/6 tasks solved, geomean 3.2x over naive Python. Speedup collapses from >15x on elementwise tasks to ~1.6x on order-dependent glyph rendering — the gradient the perceptual-correctness thesis predicts.

**Live models, cross-provider**: 4 independent models (DeepSeek, Kimi, Nemotron, Meta) tested via NVIDIA NIM. 4/4 solved the elementwise task on the first attempt (15.8–17.6x); 0/4 solved glyph rendering — and each failure decomposed into a distinct, nameable cause matching a pre-registered calibration probe (phase quantization, discarded sub-pixel position, and the silent-content-loss failure above).

**GPU track** (real Triton kernels, compiled and executed): agent-written Triton beat agent-written NumPy by 14x–842x kernel-only, ~31x end-to-end on the same task and model. One kernel's first attempt failed to compile with a genuine shape-broadcasting error; given the verbatim compiler error, the model fixed the exact line in one round and the corrected kernel ran bit-exact on hardware.

Full writeup: [`results/FINDINGS.md`](results/FINDINGS.md).

## Repository layout

```
renderbench/
  core.py         correctness oracles (incl. SSIM), timing methodology
  tasks.py        six rendering task specifications + reference implementations
  judge.py        sandboxed judge: source-screens, executes, scores, times
  pool.py         offline candidate pool (runs with no API key)
  agent/
    llm.py        provider abstraction (OpenAI-compatible + Anthropic native)
    nim.py        NVIDIA NIM free-endpoint registry, shared rate limiting
    loop.py        coder -> judge -> advisor -> coder controller
    parallel.py    concurrent multi-provider dispatch

run_bench.py        CPU-track runner
gpu_track.py         Triton kernels, kernel-only vs. end-to-end timing
triton_judge.py      compiles and scores a candidate on the GPU
triton_bench.py      live generation -> GPU judging -> compiler-error feedback
cross_model.py       multi-model concurrent evaluation
calibrate_thresholds.py   oracle calibration gate (blocks the runner on failure)

results/
  FINDINGS.md        full results writeup
  figures/           calibration probe comparison, reference render
  triton_track/       agent-written Triton kernels and GPU timings
docs/
  citations.md        sourced claims for the paper
  PREREGISTRATION.md   open methodological decisions
```

## Running it

```bash
pip install -r requirements.txt
python calibrate_thresholds.py   # gate: verifies oracle thresholds separate the probes
python run_bench.py              # CPU track, no API key required
python gpu_track.py              # Triton kernels (requires CUDA + torch + triton)
```

Live-model evaluation needs an API key (`RB_API_KEY` in `.env`); see `renderbench/agent/llm.py` for supported backends.

## Honest limitations

- Live results are single-seed, single-machine; run-to-run variance is real (documented in `FINDINGS.md`) and not yet resolved with multi-seed aggregation.
- No expert-ceiling comparison yet (e.g. against a hand-optimized GPU rasterizer).
- The VTC cost-savings claim (does faster rendering translate to real pipeline savings) requires profiling a full VTC system end-to-end — not yet done.
- GPU timings are kernel-only unless explicitly marked end-to-end; the two differ by up to 25x on cheap operations and only the end-to-end number should be quoted.
