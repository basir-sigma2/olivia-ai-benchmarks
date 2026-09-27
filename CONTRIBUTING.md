# Adding a result

1. **Run the suite.** Follow [METHOD.md](METHOD.md). One run means one server configuration.
2. **Make the folder.** Create `results/<YYYY-MM-DD>_<model>_<engine>_<N>gpu[_<variant>]/` in lower case,
   using the date the suite ran. Put two files in it:
   - `result.tsv`: the suite output.
   - `run.yaml`: copy one from another run and update every field. Each field is described in
     [`schema/run.schema.json`](schema/run.schema.json).
3. **Check it:**
   ```bash
   pip install -r scripts/requirements.txt
   python scripts/build.py
   ```
4. **Open a pull request.** CI runs the same check. A reviewer then reads the flags and notes against the
   numbers.

## run.yaml fields that need care

- **`model.revision`:** the full commit hash of the snapshot you served (`snapshots/<hash>` in the Hugging Face
  cache). A branch name is not accepted.
- **`engine.version`:** the package version inside the image (`pip show sglang`, `pip show vllm`).
- **`engine.image`:** where the container came from. If the tag moves (for example `latest`), add the pull date.
- **`engine.flags`:** exactly the flags you passed, without the model path, host and port.
- **`engine.env`:** environment variables that change behaviour.
- **`parallelism`:** the full layout (`tp`, `pp`, `ep`, `dp`). For multi-node runs, also set `nodes` and `gpus`.
- **`cuda_graphs`:** set to `false` for eager runs.
- **`validation.server_chat_requests`:** the chat-completion requests counted in the server's own log. See
  METHOD.md.
- **`notes`:** anything unusual, such as a workaround or a known issue that affects the numbers.

## Rules

- Give a rerun its own folder; never overwrite a published run.
- A run is not accepted if it has failed requests, is missing a point (other than one skipped for context
  length), or has a request count that does not match.
- Only results go here. Keep job scripts, project accounts and file-system paths out of `run.yaml`.
