#!/bin/bash -l
#SBATCH --job-name=modelscope-llama-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# Export MODELSCOPE_VENV and HF_HUB_CACHE before submitting.
# sbatch --account=ACCOUNT --partition=PARTITION --gres=gpu:TYPE:1 \
#     --chdir=/path/to/modelscope /path/to/modelscope/scripts/slurm/llama_smoke.sbatch
# Optional: MODELSCOPE_SERVER_LOG and MODELSCOPE_STARTUP_TIMEOUT (seconds).
set -euo pipefail

: "${MODELSCOPE_VENV:?Set MODELSCOPE_VENV to the ModelScope virtualenv directory}"
: "${HF_HUB_CACHE:?Set HF_HUB_CACHE to the Hugging Face Hub cache directory}"
export HF_HUB_CACHE

if [[ ! -f serve/api.py ]]; then
    echo "Use sbatch --chdir to run from the ModelScope checkout." >&2
    exit 1
fi

module load gcc/13.3.0 python/3.11.9
venv_python="$MODELSCOPE_VENV/bin/python"
if [[ ! -x "$venv_python" ]]; then
    echo "ModelScope Python is missing or not executable at $venv_python" >&2
    exit 1
fi

# Release any CUDA state from the preflight before starting the serving process.
"$venv_python" -u - <<'PY'
import subprocess

import torch

print("PyTorch", torch.__version__)
print("PyTorch CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the ModelScope virtualenv.")
if torch.cuda.device_count() != 1:
    raise SystemExit("Expected one visible CUDA device.")

gpu = torch.cuda.get_device_properties(0)
print("GPU model", gpu.name)
print(f"GPU VRAM {gpu.total_memory / 1024**2:.0f} MiB")
driver = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    text=True,
    timeout=10,
).splitlines()
if not driver:
    raise SystemExit("nvidia-smi reported no NVIDIA driver version.")
print("NVIDIA driver", ", ".join(sorted(set(driver))))
PY

server_log="${MODELSCOPE_SERVER_LOG:-slurm-${SLURM_JOB_ID:-$$}-llama-smoke.server.log}"
exec "$venv_python" -u - "$server_log" <<'PY'
import json
import os
import signal
import socket
import subprocess
import sys
import time
from http.client import HTTPConnection, HTTPException
from pathlib import Path

HOST = "127.0.0.1"
VARIANT = "llama_nf4"


def request_json(port, path, timeout, payload=None):
    connection = HTTPConnection(HOST, port, timeout=timeout)
    try:
        if payload is None:
            connection.request("GET", path)
        else:
            connection.request(
                "POST", path, json.dumps(payload), {"Content-Type": "application/json"}
            )
        with connection.getresponse() as response:
            if response.status != 200:
                raise RuntimeError(f"{path} returned HTTP {response.status}")
            result = json.load(response)
        if not isinstance(result, dict):
            raise RuntimeError(f"{path} returned a non-object JSON response")
        return result
    finally:
        connection.close()


def wait_for_health(server, port, timeout):
    deadline = time.monotonic() + timeout
    while (remaining := deadline - time.monotonic()) > 0:
        if (status := server.poll()) is not None:
            raise RuntimeError(f"Uvicorn exited with status {status} during startup")
        try:
            health = request_json(port, "/health", min(1.0, remaining))
        except (OSError, HTTPException):
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
            continue
        if time.monotonic() < deadline:
            return health
    raise RuntimeError(f"/health did not respond before the {timeout}s startup deadline")


def stop_server(server):
    # Reap the child before handling any further cancellation signals.
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def interrupted(signum, _frame):
    print(f"ModelScope smoke test interrupted by signal {signum}.", file=sys.stderr)
    raise SystemExit(128 + signum)


def main(log_path):
    timeout = int(os.environ.get("MODELSCOPE_STARTUP_TIMEOUT", "300"))
    if timeout <= 0:
        raise ValueError("MODELSCOPE_STARTUP_TIMEOUT must be a positive number of seconds")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    server = None
    try:
        # Pass a reserved loopback socket so another job cannot answer this probe.
        with socket.socket() as listener, log_path.open("a") as log:
            listener.bind((HOST, 0))
            port = listener.getsockname()[1]
            fd = listener.fileno()
            print(f"Starting Uvicorn on {HOST}:{port}. Server log at {log_path}")
            server = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn", "serve.api:app",
                    "--fd", str(fd), "--workers", "1", "--loop", "asyncio",
                    "--lifespan", "on",
                ],
                pass_fds=(fd,),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        health = wait_for_health(server, port, timeout)
        if health.get("status") != "ok" or health.get("model") != VARIANT:
            raise RuntimeError(f"/health did not report a healthy {VARIANT}: {health!r}")
        result = request_json(
            port, "/generate", 60,
            {"prompt": "The capital of France is", "max_tokens": 16, "variant": VARIANT},
        )
        text = result.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("/generate did not return nonempty text")
    finally:
        if server is not None:
            stop_server(server)

    print(f"Smoke test passed for {VARIANT}.")


signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGTERM, interrupted)
log_path = Path(sys.argv[1])
try:
    main(log_path)
except (OSError, ValueError, RuntimeError, HTTPException, subprocess.SubprocessError) as exc:
    raise SystemExit(f"ModelScope smoke test failed: {exc}. Server log at {log_path}") from exc
PY
