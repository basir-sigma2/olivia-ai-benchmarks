## Run

<!-- Model, engine, GPUs, and anything unusual about this run. -->

## Checklist

- [ ] One folder per run: `results/<YYYY-MM-DD>_<model>_<engine>_<N>gpu[_<variant>]/` with `run.yaml` and `result.tsv`
- [ ] `python scripts/build.py` passes
- [ ] `model.revision` is the full commit hash of the snapshot that was served
- [ ] `validation.server_chat_requests` was counted from the server's own log
- [ ] `cuda_graphs` is correct, and workarounds are in `notes`
