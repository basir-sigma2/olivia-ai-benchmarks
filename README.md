# Olivia AI benchmarks

Inference benchmarks for large language models on [Olivia](https://documentation.sigma2.no/hpc_machines/olivia.html),
Norway's national supercomputer (Sigma2 / NRIS). Olivia's GPU nodes have four NVIDIA GH200 superchips each,
connected by HPE Slingshot-11.

Each run is a folder under [`results/`](results) with two files:

- `run.yaml` records the exact model revision, engine version, container and server flags.
- `result.tsv` holds the numbers from one fixed benchmark suite.

The suite is the one CSCS used for GLM-5.3 on Alps, which has the same GH200 and Slingshot hardware, plus a
128-user point. That keeps runs comparable across models, engines and sites.

**Live table:** https://basir-sigma2.github.io/olivia-ai-benchmarks/ ·
**all runs as JSON:** https://basir-sigma2.github.io/olivia-ai-benchmarks/results.json

## September 2026 baseline

Each prompt is 1,024 tokens and each response 128. Throughput is output tokens per second across all requests,
at 1–128 concurrent users. TPOT is the time per output token at one user. Peak / GPU is the best throughput
at any concurrency, divided by the run's GPU count. The test is prompt-heavy and counts output tokens only, so one
user's figure sits well below the decode speed, which is 1000 / TPOT (77 tokens/s for GLM-5.2, for example).

| Model | Engine | GPUs | 1 | 16 | 64 | 128 | TPOT ms | Peak / GPU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| [Apertus-v1.5-8B](results/2026-09-27_apertus-1.5-8b_vllm_1gpu) | vLLM 0.23.1rc1.dev (Swiss AI) | 1 | 163 | 1,749 | 3,754 | 4,453 | 5.8 | 4,453 |
| [borealis2-26b-a4b-preview](results/2026-09-27_borealis2-26b-a4b_sglang_1gpu) | SGLang 0.5.18 | 1 | 159 | 1,073 | 1,821 | 1,897 | 5.2 | 1,897 |
| [borealis-27b](results/2026-09-27_borealis-27b_vllm_1gpu) | vLLM 0.27.1 | 1 | 70 | 683 | 1,015 | 1,033 | 12.3 | 1,128 |
| [Qwen3.8-27B-FP8](results/2026-09-27_qwen3.8-27b_sglang_1gpu) | SGLang 0.5.18 | 1 | 83 | 725 | 891 | 890 | 11.0 | 1,000 |
| [Qwen3.8-27B-FP8](results/2026-09-27_qwen3.8-27b_sglang_2gpu) | SGLang 0.5.18 | 2 | 122 | 1,073 | 2,260 | 1,721 | 7.5 | 1,130 |
| [Qwen3.6-35B-A3B-FP8](results/2026-09-27_qwen3.6-35b-a3b_sglang_1gpu) | SGLang 0.5.18 | 1 | 235 | 1,792 | 3,385 | 3,099 | 3.7 | 3,385 |
| [Apertus-v1.5-70B](results/2026-09-27_apertus-1.5-70b_vllm_2gpu_eager) (eager) | vLLM 0.23.1rc1.dev (Swiss AI) | 2 | 10 | 142 | 452 | 673 | 96.4 | 337 |
| [DeepSeek-V4-Flash-FP8](results/2026-09-27_deepseek-v4-flash_sglang_4gpu) | SGLang 0.5.17 | 4 | 101 | 928 | 1,810 | 2,226 | 6.6 | 557 |
| [GLM-5.2-FP8](results/2026-09-27_glm-5.2_vllm_12gpu_tp4-pp3) | vLLM 0.27.1 (LAIFS) | 12 (3 nodes) | 38 | 347 | 925 | 1,250 | 13.0 | 104 |
| [DeepSeek-V4-Pro](results/2026-09-27_deepseek-v4-pro_sglang_12gpu_tp4-pp3) | SGLang 0.5.17 | 12 (3 nodes) | 41 | 390 | 1,058 | 1,450 | 15.3 | 121 |

The Apertus-v1.5-70B run is eager: CUDA-graph capture crashes in that image. Compare it only with other eager
runs. GLM-5.2 and DeepSeek-V4-Pro each run as one server over three nodes: tensor parallel inside each node, one
pipeline stage per node. Their 1-user figures include first-request costs, since the suite does no warm-up. Prefill, decode and total-throughput numbers for every run are in its `result.tsv` and in the live table.

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
