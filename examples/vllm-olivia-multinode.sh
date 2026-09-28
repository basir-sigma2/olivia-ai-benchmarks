#!/bin/bash
#
# vLLM on Olivia across one or more GH200 nodes, without Ray.
#
# Set up as an example for GLM-5.2-FP8 (zai-org, 753B MoE, FP8) on 3 nodes: one server
# over 12 GH200, tensor parallel 4 x pipeline parallel 3. These are the launch settings of
# the measured Olivia run: 74 tokens/s for one user, 1,255 tokens/s at 128 users
# (https://github.com/basir-sigma2/olivia-ai-benchmarks). For another model, change
# MODEL, SERVED_NAME and --nodes.
#
# One vLLM process per node, started with srun; NCCL runs over Slingshot through
# aws-ofi-nccl + libfabric (cxi) with GPUDirect RDMA via dma-buf. The image is
# Dragana's LAIFS vLLM container (Sigma2 build of the LUMI AI Factory container,
# vLLM 0.27.1).
#
# The image lives in /cluster/projects/nn9997k, which only nn9997k members can
# open: "Permission denied" on IMAGE means you need access from Dragana.
#
# Usage: set --account and CACHE_DIR (and MODEL if needed), then   sbatch vllm-olivia-multinode.sh
# Output: vllm-<job>.out (this script) and vllm-server-<job>.log (the server).
#
#SBATCH --job-name=vllm-glm5.2
#SBATCH --account=nnXXXXk
#SBATCH --partition=accel
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=vllm-%j.out

set -euo pipefail

# ---------------------------------------------------------------- SETTINGS --
IMAGE=/cluster/projects/nn9997k/laifs_olivia_gh200_full/gh200_full_0903.sif
# GLM-5.2-FP8 at revision f33c6dc (704 GiB). This copy is readable by nn11132k members only;
# otherwise download your own:  hf download zai-org/GLM-5.2-FP8 --revision f33c6dc501ee5a2c7e35155653b1b1abbc320951
MODEL=/cluster/work/projects/nn11132k/based/models/hub/models--zai-org--GLM-5.2-FP8/snapshots/f33c6dc501ee5a2c7e35155653b1b1abbc320951
SERVED_NAME=zai-org/GLM-5.2-FP8                        # the name clients use
PORT=8000
TP=4                                                   # GPUs per node
PP=${SLURM_JOB_NUM_NODES}                              # one pipeline stage per node
MAX_MODEL_LEN=32768
CACHE_DIR=/cluster/work/projects/nnXXXXk/${USER}/vllm-cache   # compile caches (not $HOME: quota)
EXTRA_ARGS="--trust-remote-code"                       # more vllm serve flags if needed
KEEP_SERVING=0                                         # 1 = keep the server up until walltime
# ---------------------------------------------------------------------------

mkdir -p "${CACHE_DIR}"
HEAD=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
LOG="vllm-server-${SLURM_JOB_ID}.log"

# The Slingshot NIC only gets a network ID (VNI) inside an srun step; a single-node
# step needs single_node_vni. Without it NCCL fails with "RC: -38".
if [ "${SLURM_JOB_NUM_NODES}" -gt 1 ]; then NET=disable_rdzv_get; else NET=single_node_vni,disable_rdzv_get; fi

# vLLM ranks meet on the head node's Slingshot address (hsn0).
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD}" --overlap \
  bash -c "ip -4 -o addr show hsn0 | awk '{print \$4}' | cut -d/ -f1 | head -1")
echo "job ${SLURM_JOB_ID}: ${SLURM_JOB_NUM_NODES} node(s) ${SLURM_JOB_NODELIST}, head ${HEAD} (${HEAD_IP}), tp ${TP} x pp ${PP}"

export IMAGE MODEL SERVED_NAME PORT TP PP MAX_MODEL_LEN CACHE_DIR EXTRA_ARGS HEAD_IP
# shellcheck disable=SC2016  # the per-node script below expands its own variables.
srun --network="${NET}" --nodes="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  --gres=gpu:h200:4 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  bash -c '
    RANK=${SLURM_NODEID}
    MYIP=$(ip -4 -o addr show hsn0 | awk "{print \$4}" | cut -d/ -f1 | head -1)
    MULTI=""
    if [ "${SLURM_JOB_NUM_NODES}" -gt 1 ]; then
      MULTI="--nnodes ${SLURM_JOB_NUM_NODES} --node-rank ${RANK} --master-addr ${HEAD_IP} --master-port 29555"
      [ "${RANK}" != 0 ] && MULTI="${MULTI} --headless"   # rank 0 serves the API
    fi
    # shellcheck disable=SC2086  # MULTI and EXTRA_ARGS are flag lists.
    exec apptainer exec --nv \
      --bind /cluster/projects --bind /cluster/work \
      --env HF_HUB_OFFLINE=1 \
      --env XDG_CACHE_HOME="${CACHE_DIR}" --env TRITON_CACHE_DIR="${CACHE_DIR}/triton" \
      --env VLLM_HOST_IP="${MYIP}" \
      --env NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 --env GLOO_SOCKET_IFNAME=hsn0 \
      --env FI_HMEM_CUDA_USE_DMABUF=1 \
      --env FI_CXI_DEFAULT_TX_SIZE=16384 --env FI_CXI_RX_MATCH_MODE=software \
      --env NCCL_CROSS_NIC=0 --env NCCL_PXN_DISABLE=1 \
      --env NCCL_DEBUG=INFO --env NCCL_DEBUG_SUBSYS=INIT,NET \
      --env VLLM_USE_FLASHINFER_SAMPLER=0 \
      --env CPATH=/usr/local/cuda-13.0/targets/sbsa-linux/include/cccl \
      --env PYTHONNOUSERSITE=1 \
      "${IMAGE}" vllm serve "${MODEL}" --served-model-name "${SERVED_NAME}" \
        --host 0.0.0.0 --port "${PORT}" \
        --tensor-parallel-size "${TP}" --pipeline-parallel-size "${PP}" \
        --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization 0.88 \
        ${MULTI} ${EXTRA_ARGS}
  ' > "${LOG}" 2>&1 < /dev/null &
SRV=$!

echo "waiting for the server (big models take 5-10 min to load)..."
until curl -sf --noproxy '*' "http://${HEAD}:${PORT}/v1/models" > /dev/null; do
  kill -0 "${SRV}" 2>/dev/null || { echo "server exited before it was ready; see ${LOG}"; exit 1; }
  sleep 15
done
echo "server ready: http://${HEAD}:${PORT}/v1"

echo "--- test request"
curl -s --noproxy '*' "http://${HEAD}:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\": \"${SERVED_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello from Olivia in one sentence.\"}], \"max_tokens\": 64}"
echo

echo "--- network check (from ${LOG})"
grep -m1 -oE "NCCL version [^ ]+" "${LOG}" || true
grep -m1 -oE "Using network [A-Za-z ]+" "${LOG}" || echo "WARNING: no 'Using network' line: NCCL may be on TCP sockets"
echo "GPUDirect RDMA channels: $(grep -c 'GDRDMA' "${LOG}" || true) (should be > 0 on more than one node)"

if [ "${KEEP_SERVING}" = 1 ]; then
  echo "serving until walltime"; wait "${SRV}"
else
  kill "${SRV}" 2>/dev/null || true; wait "${SRV}" 2>/dev/null || true; echo "done"
fi
