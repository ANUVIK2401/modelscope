#!/usr/bin/env bash
# Run from Google Cloud Shell on the credited trial account.
set -euo pipefail

project="${1:?Pass the existing Google Cloud project ID}"
region="${2:-europe-west1}"
runner="${3:-benchmarks/qwen_serving.py}"
mode="${4:-main}"
task_timeout=30m
case "$mode" in
    main) job=modelscope-qwen-serving; task_timeout=60m ;;
    prefix) job=modelscope-qwen-prefix ;;
    retry) job=modelscope-qwen-retry ;;
    quick) job=modelscope-qwen-paired; task_timeout=35m ;;
    *) echo "Expected mode main, prefix, retry, or quick." >&2; exit 2 ;;
esac

if [[ ! -r "$runner" ]]; then
    echo "Cannot read the Qwen serving benchmark: $runner" >&2
    exit 2
fi

source_b64="$(gzip -c "$runner" | base64 | tr -d '\n')"
launcher="$(cat <<'PY'
import base64, gzip, io, os, pathlib, subprocess, sys, tarfile
path = pathlib.Path('/tmp/qwen_serving.py')
path.write_bytes(gzip.decompress(base64.b64decode(os.environ['MODELSCOPE_SOURCE_B64'])))
output = pathlib.Path('/tmp/modelscope-results')
command = [sys.executable, '-u', str(path), '--output', str(output)]
mode = os.environ['MODELSCOPE_MODE']
if mode != 'prefix':
    command.append('--discard-model-cache')
if mode == 'prefix':
    command.append('--prefix')
elif mode in ('retry', 'quick'):
    command.extend(['--requests', '16' if mode == 'quick' else '32'])
    if mode == 'retry':
        os.environ['HF_HUB_DISABLE_XET'] = '1'
result = subprocess.run(command)
if output.is_dir():
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as archive:
        archive.add(output, arcname='modelscope-results')
    encoded = base64.b64encode(data.getvalue()).decode()
    for index, offset in enumerate(range(0, len(encoded), 16000)):
        print(f'MODELSCOPE_ARTIFACT {index:05d} {encoded[offset:offset+16000]}', flush=True)
raise SystemExit(result.returncode)
PY
)"
gcloud run jobs create "$job" \
    --project="$project" --region="$region" \
    --image=docker.io/vllm/vllm-openai:v0.11.2 \
    --gpu=1 --gpu-type=nvidia-l4 --no-gpu-zonal-redundancy \
    --cpu=8 --memory=32Gi --tasks=1 --parallelism=1 \
    --max-retries=0 --task-timeout="$task_timeout" \
    --command=python3 --args="^@^-u@-c@$launcher" \
    --set-env-vars="MODELSCOPE_SOURCE_B64=$source_b64,MODELSCOPE_MODE=$mode,HF_HOME=/tmp/modelscope-hf,HF_XET_CHUNK_CACHE_SIZE_BYTES=0,HF_HUB_DOWNLOAD_TIMEOUT=60" \
    --execute-now
