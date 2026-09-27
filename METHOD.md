# Method: suite `olivia-v3`

## Setup

- **Server and client.** Each run has one OpenAI-compatible server (SGLang, vLLM or similar) and one benchmark
  client on the same node. They talk over loopback on `/v1/chat/completions`. For multi-node runs, the client
  runs on the node that serves the API.
- **Prompts.** Prompts are random token ids of fixed length. EOS is ignored, so every request generates exactly
  `out_len` tokens.
- **Warm-up.** The suite runs twice on the same server. The first pass is untimed and discarded; only the second
  pass is recorded. Each point also starts with one untimed wave: as many requests as its concurrency, with its own
  prompt and answer lengths. By the timed pass, every kernel and batch shape the suite uses has already run, so the
  server is measured hot, the way a running service is.
- **Server configuration.** Serve the model the way you would in production, without reasoning or tool-call
  parsers. Those parsers move tokens out of the counted output.

## The 12 points (1,076 measured requests, 2,738 in total)

| Scenario | Input tokens | Output tokens | Concurrency | Prompts |
|---|---|---|---|---|
| conc-scaling | 1,024 | 128 | 1, 4, 16, 32, 64, 128 | 4 × concurrency |
| prefill | 1,024 / 4,096 / 16,384 | 32 | 8 | 16 |
| decode | 256 | 128 / 512 / 1,024 | 8 | 16 |

The scenarios are CSCS's [`bench_suite.sh`](https://github.com/andresnowak/glm-5.3-inference/blob/main/scripts/bench_suite.sh);
the 128-user point is added here. A point is skipped only when input + output + 512 exceeds the server's
context length (`max_context` in `run.yaml`).

## Client command

Run all 12 points once and discard the results (warm pass), then run them again and record them (timed pass). Run
one command per point, using vLLM's client (`$SNAPSHOT` is the local model snapshot; the client only reads
its tokenizer):

```bash
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --base-url "http://127.0.0.1:$PORT" --model "$SNAPSHOT" --served-model-name "$SERVED" \
  --tokenizer "$SNAPSHOT" --trust-remote-code --ignore-eos \
  --dataset-name random --random-input-len "$IN" --random-output-len "$OUT" \
  --num-warmups "$CONC" --num-prompts "$PROMPTS" --max-concurrency "$CONC"
```

Some new architectures have a tokenizer or config that the client's transformers version cannot load. For
those, use SGLang's client. `random-ids` works offline, unlike `random`, which downloads ShareGPT. A range
ratio of 1 fixes the lengths.

```bash
python3 -m sglang.benchmark.serving --backend sglang-oai-chat --base-url "http://127.0.0.1:$PORT" \
  --model "$SERVED" --tokenizer "$SNAPSHOT" --dataset-name random-ids \
  --random-input-len "$IN" --random-output-len "$OUT" --random-range-ratio 1 \
  --warmup-requests "$CONC" --num-prompts "$PROMPTS" --max-concurrency "$CONC"
```

## `result.tsv`

The file is tab-separated, with one header line and one row per point:

```
label  scenario  in_len  out_len  prompts  conc  req_s  out_tok_s  total_tok_s  ttft_ms  tpot_ms  p99_tpot_ms
```

The last six values come from the client's summary lines, in order: Request throughput, Output token
throughput, Total token throughput, Mean TTFT, Mean TPOT and P99 TPOT. `total_tok_s` counts the chat
template's tokens as input. The columns match CSCS's `bench_suite.tsv`.

## Validity

Olivia's GPU nodes are shared between jobs, and a fixed port once let four jobs benchmark each other's servers.
To keep runs valid:

- **Use your own port.** Start the server on a port unique to your job, for example 20000 + (job id mod
  10000). Before the suite starts, check that `/v1/models` lists your served model name.
- **Count the requests.** After the suite, count the chat-completion requests in the server's own log and put
  the number in `validation.server_chat_requests`.
  - It must equal the suite's total including untimed requests: 2,738 when all 12 points run (two passes of 1,076
    prompts plus 293 warm-up requests each). Some SGLang versions log one more request of their own at start-up.
  - SGLang 0.5.20 and later log "compile after serving started" when a Triton kernel compiles while serving. Count
    those lines after the timed pass starts and record the number in `validation.timed_pass_compile_warnings`. It
    should be 0.
  - Any other number means another job reached your server, or your client reached another server. Rerun.
  - `scripts/build.py` enforces this.
- **No failed requests.** A failed request in any point invalidates the run.

## Olivia platform notes (September 2026)

- **Exclusive_Process GPUs.** The GPUs run in Exclusive_Process compute mode. SGLang multimodal models need
  `--disable-fast-image-processor`, because the fast processor uses the GPU from a second process.
- **No FlashAttention-3 on arm64.** arm64 SGLang builds ship without FlashAttention-3 kernels. Use
  `--attention-backend flashinfer`, or `triton`.
- **FP8 MoE layouts.** FP8 block-quantised MoE models need `moe_intermediate_size / tp` to be divisible by 128.
  If it is not, use expert parallel (`--ep-size`).
- **CUDA graphs.** Record whether graphs were on (`cuda_graphs`). Eager runs are much slower at low
  concurrency, and they are labelled in the tables.

## Suite versions

- **`olivia-v3`** (current, from 27 September 2026): the `olivia-v2` suite twice on the same server, with only the
  second pass recorded.
- **`olivia-v2`** (27 September 2026): the 12 points, each preceded by one untimed warm-up wave. That was not
  enough: SGLang 0.5.20 still compiled Triton kernels during timed points, and one 1-user request took 11.5 s.
- **`olivia-v1`** (27 September 2026): the same 12 points measured cold, like CSCS's `bench_suite.sh`. On
  multi-node servers the first point then includes one-time costs: time to first token at 1 user was about twice
  the 4-user value.

Each run keeps the suite it was measured with. Tables show the latest suite.
