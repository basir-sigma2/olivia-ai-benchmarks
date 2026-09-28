# Olivia AI benchmarks

Inference benchmarks for large language models on [Olivia](https://documentation.sigma2.no/hpc_machines/olivia.html),
Norway's national supercomputer (Sigma2). Olivia's GPU nodes have four NVIDIA GH200 superchips each,
connected by HPE Slingshot-11.

Each run is a folder under [`results/`](results) with two files:

- `run.yaml` records the exact model revision, engine version, container and server flags.
- `result.tsv` holds the numbers from one fixed benchmark suite.

The suite is the one CSCS used for GLM-5.3 on Alps, which has the same GH200 and Slingshot hardware, plus a
128-user point. That keeps runs comparable across models, engines and sites.

**Live table:** https://basir-sigma2.github.io/olivia-ai-benchmarks/ ·
**all runs as JSON:** https://basir-sigma2.github.io/olivia-ai-benchmarks/results.json

## September 2026 baseline

Measured with suite `olivia-v3`: each server runs the suite twice, and only the second pass is recorded. Each prompt
is 1,024 tokens and each response 128. Throughput is output tokens per second across all requests, at 1–128
concurrent users. TPOT is the time per output token at one user, so one user's decode speed is 1000 / TPOT. Peak /
GPU is the best throughput at any concurrency, divided by the run's GPU count.

| Model | Engine | GPUs | 1 | 16 | 64 | 128 | TPOT ms | Peak / GPU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [Apertus-v1.5-8B](results/2026-09-27_apertus-1.5-8b_vllm_1gpu) | vLLM 0.23.1rc1.dev (Swiss AI) | 1 | 170 | 1,753 | 3,778 | 4,476 | 5.8 | 4,476 |
| [borealis2-26b-a4b-preview](results/2026-09-27_borealis2-26b-a4b_sglang_1gpu) | SGLang 0.5.18 | 1 | 178 | 1,070 | 1,806 | 1,886 | 5.2 | 1,886 |
| [borealis-27b](results/2026-09-27_borealis-27b_vllm_1gpu) | vLLM 0.27.1 | 1 | 78 | 678 | 1,014 | 1,034 | 12.3 | 1,125 |
| [Qwen3.8-27B-FP8](results/2026-09-27_qwen3.8-27b_sglang_1gpu) | SGLang 0.5.18 | 1 | 87 | 764 | 911 | 911 | 11.2 | 985 |
| [Qwen3.8-27B-FP8](results/2026-09-27_qwen3.8-27b_sglang_2gpu) | SGLang 0.5.18 | 2 | 127 | 1,093 | 2,267 | 1,710 | 7.4 | 1,134 |
| [Qwen3.6-35B-A3B-FP8](results/2026-09-27_qwen3.6-35b-a3b_sglang_1gpu) | SGLang 0.5.18 | 1 | 252 | 1,833 | 3,485 | 3,224 | 3.7 | 3,485 |
| [Apertus-v1.5-70B](results/2026-09-27_apertus-1.5-70b_vllm_2gpu_eager) (eager) | vLLM 0.23.1rc1.dev (Swiss AI) | 2 | 11 | 144 | 479 | 721 | 93.1 | 361 |
| [Qwen3.8-Flash-Next-FP8](results/2026-09-27_qwen3.8-flash-next_sglang_4gpu_ep4) | SGLang 0.5.20 | 4 | 132 | 1,073 | 2,413 | 2,761 | 6.1 | 690 |
| [DeepSeek-V4-Flash-FP8](results/2026-09-27_deepseek-v4-flash_sglang_4gpu) | SGLang 0.5.17 | 4 | 126 | 980 | 1,802 | 2,224 | 6.6 | 556 |
| [DeepSeek-V4-Flash-FP8](results/2026-09-27_deepseek-v4-flash_sglang_8gpu_tp4-pp2) (tp4 × pp2) | SGLang 0.5.17 | 8 (2 nodes) | 102 | 959 | 2,645 | 3,902 | 8.4 | 488 |
| [GLM-5.2-FP8](results/2026-09-27_glm-5.2_vllm_12gpu_dp3-tp4-ep12) (dp3 × tp4, ep12) | vLLM 0.27.1 (LAIFS) | 12 (3 nodes) | 27 | 317 | 704 | 972 | 35.8 | 81 |
| [GLM-5.2-FP8](results/2026-09-27_glm-5.2_vllm_12gpu_dp3-tp4-ep12-deepepll) (dp3 × tp4, ep12, DeepEP) | vLLM 0.27.1 (LAIFS) | 12 (3 nodes) | 15 | 144 | 377 | 403 | 59.6 | 34 |
| [GLM-5.2-FP8](results/2026-09-27_glm-5.2_vllm_12gpu_tp4-pp3) (tp4 × pp3) | vLLM 0.27.1 (LAIFS) | 12 (3 nodes) | 74 | 444 | 979 | 1,255 | 13.1 | 104 |
| [DeepSeek-V4-Pro](results/2026-09-27_deepseek-v4-pro_sglang_12gpu_tp4-pp3) (tp4 × pp3) | SGLang 0.5.17 | 12 (3 nodes) | 60 | 427 | 1,073 | 1,380 | 14.9 | 115 |

The Apertus-v1.5-70B run is eager: CUDA-graph capture crashes in that image. Compare it only with other eager runs.
Prefill, decode and total-throughput numbers for every run are in its `result.tsv` and in the live table.
Multi-node runs are one server over 2–3 nodes, with NCCL over Slingshot and GPUDirect RDMA. Tensor parallel inside
each node with one pipeline stage per node (tp × pp) is the fastest layout; the GLM-5.2 wide expert-parallel
layouts (dp × tp, ep) are slower at every concurrency up to 128. DeepSeek-V4-Pro on 4 nodes is queued.
The first September runs, measured cold (`olivia-v1`), are kept as `*_cold` and hidden in the live table by default.

## Adding a result

Anyone who runs the suite on Olivia can add a result by pull request; the details are in
[CONTRIBUTING.md](CONTRIBUTING.md). The steps:

1. Run the suite described in [METHOD.md](METHOD.md).
2. Add `results/<date>_<model>_<engine>_<N>gpu/` containing `run.yaml` and `result.tsv`.
3. Run `python scripts/build.py`.
4. Open a pull request.

CI checks every run with the same script. The live table rebuilds when a pull request is merged.

## Layout

```
results/<run>/run.yaml     model, engine, image, flags, layout, validation count
results/<run>/result.tsv   the suite's 12 points
schema/run.schema.json     every run.yaml field, documented
scripts/build.py           validates all runs, writes results.json
site/index.html            the live table (GitHub Pages)
```

## Credits

The benchmark scenarios come from the GLM-5.3 inference work on CSCS Alps,
[andresnowak/glm-5.3-inference](https://github.com/andresnowak/glm-5.3-inference) (`scripts/bench_suite.sh`).
Their published numbers can be compared line by line with runs here.

## License

The results, run metadata and documentation are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The scripts are licensed under MIT. See [LICENSE](LICENSE).
