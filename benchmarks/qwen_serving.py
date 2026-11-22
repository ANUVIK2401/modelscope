"""Run a one-GPU, paired Qwen3-8B streaming benchmark."""

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

MODELS = {
    "bf16": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "awq": ("Qwen/Qwen3-8B-AWQ", "4da05a8edb55c6046cce958586c33b61da07bb79"),
}
SLO_MS = {"short": 500, "long": 1500}
CONCURRENCIES = (1, 4, 16)
PORT = 8765
ENDPOINT = f"http://127.0.0.1:{PORT}"

SHORT_TOPICS = (
    "DNS caching", "HTTP keep-alive", "database indexing", "write-ahead logs",
    "binary search", "TCP congestion control", "connection pools", "LRU eviction",
    "schema migrations", "hash partitions", "idempotency", "rate limiting",
    "backpressure", "thread pools", "reverse proxies", "quorum reads",
)


def long_document(index):
    facts = (f"Harbor-{100 + index}", str(160 + index), str(20 + index))
    sections = [
        f"Record {i:02d}. The maintenance panel logged survey {index:03d}-{i:02d}. "
        f"The onsite team reviewed service tickets from zone {i % 7}. "
        "These preliminary observations concern equipment inventory, "
        "scheduling dependencies and unrelated background events. "
        "Every observation below is provisional unless included in the final decision."
        for i in range(23)
    ]
    sections.insert(index % 19, (
        f"Final decision. The selected site is {facts[0]}. The approved budget is "
        f"{facts[1]} thousand dollars. The implementation window is {facts[2]} days."
    ))
    return "\n".join(sections), facts


def prompt_for(workload, index):
    if workload == "short":
        return (
            f"Case {index:03d}. Explain {SHORT_TOPICS[index % len(SHORT_TOPICS)]} "
            "in two concise sentences and include one concrete example."
        )
    document, _ = long_document(index)
    return (
        "Summarize the final project decision in three bullets. Include the selected "
        "site, budget in thousands of dollars, and implementation window in days. "
        f"Ignore provisional notes.\n\n{document}"
    )


def quality_cases():
    for index in range(100, 120):
        left, right = index + 23, index % 17 + 8
        yield f"Compute {left} × {right}. Return just the integer.", str(left * right)
    for index in range(120, 140):
        document, facts = long_document(index)
        yield (
            f"{document}\n\nReturn only the selected site, budget in thousands, "
            "and window in days, separated by commas, in that order.",
            ", ".join(facts),
        )


async def request(client, prompt, max_tokens, *, fixed_tokens=False):
    body = {
        "model": "modelscope",
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": fixed_tokens,
    }
    first = None
    parts = []
    usage = {}
    done = False
    reason = None
    error = None
    start = time.perf_counter()
    try:
        async with client.stream("POST", f"{ENDPOINT}/v1/chat/completions", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise ValueError(str(event["error"]))
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    reason = choice.get("finish_reason") or reason
                    content = choice.get("delta", {}).get("content")
                    if content:
                        if first is None and content.strip():
                            first = time.perf_counter()
                        parts.append(content)
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    if error is None and not (first is not None and done and reason and usage):
        error = "Incomplete stream, missing generated text, finish reason, or usage"
    return {
        "ttft_ms": (first - start) * 1000 if first is not None else None,
        "e2e_ms": (time.perf_counter() - start) * 1000,
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "finish_reason": reason,
        "text": "".join(parts),
        "error": error,
    }


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def save_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


async def condition(client, folder, workload, concurrency, count, repeat, prompts=None):
    if prompts is None:
        prompts = [prompt_for(workload, i) for i in range(count)]
    semaphore = asyncio.Semaphore(concurrency)

    async def send(index, prompt):
        async with semaphore:
            row = await request(client, prompt, 64, fixed_tokens=True)
        return {"request": index, **row}

    start = time.perf_counter()
    rows = await asyncio.gather(*(send(i, prompt) for i, prompt in enumerate(prompts)))
    elapsed = time.perf_counter() - start
    ok = [row for row in rows if row["error"] is None]
    target = SLO_MS.get(workload, SLO_MS["long"])
    summary = {
        "workload": workload, "concurrency": concurrency, "repeat": repeat,
        "requests": len(rows), "successful": len(ok), "errors": len(rows) - len(ok),
        "elapsed_s": elapsed, "ttft_target_ms": target,
        "ttft_p50_ms": percentile([row["ttft_ms"] for row in ok], 0.50),
        "ttft_p95_ms": percentile([row["ttft_ms"] for row in ok], 0.95),
        "output_toks_s": sum(row["output_tokens"] for row in ok) / elapsed,
        "goodput_rps": sum(row["ttft_ms"] <= target for row in ok) / elapsed,
    }
    save_json(folder / f"{workload}-c{concurrency}-r{repeat}.json", {
        "summary": summary, "rows": rows,
    })
    print(json.dumps(summary), flush=True)
    return summary


def serve_command(variant, cache, cache_dir):
    model, revision = MODELS[variant]
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--model", model, "--revision", revision,
        "--tokenizer", MODELS["bf16"][0], "--tokenizer-revision", MODELS["bf16"][1],
        "--served-model-name", "modelscope", "--generation-config", "vllm",
        "--dtype", "bfloat16", "--kv-cache-dtype", "auto", "--max-model-len", "2048",
        "--max-num-seqs", "16", "--max-num-batched-tokens", "2048",
        "--gpu-memory-utilization", "0.94", "--cpu-offload-gb", "0",
        "--enforce-eager", "--disable-log-requests", "--download-dir", str(cache_dir),
        "--enable-prefix-caching" if cache else "--no-enable-prefix-caching",
    ]
    if variant == "awq":
        command += ["--quantization", "awq_marlin"]
    return command


async def run_server(variant, folder, cache_dir, args, *, prefix_cache=False):
    folder.mkdir(parents=True)
    command = serve_command(variant, prefix_cache, cache_dir)
    save_json(folder / "command.json", command)
    print(f"Starting {variant} on localhost with prefix caching {prefix_cache}", flush=True)
    log_path = folder / "server.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                last_progress = ""
                startup_deadline = time.monotonic() + 600
                attempt = 0
                while time.monotonic() < startup_deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f"vLLM exited ({server.returncode}); see {log_path}")
                    try:
                        health = await client.get(f"{ENDPOINT}/health", timeout=2)
                        if health.is_success:
                            break
                    except httpx.RequestError:
                        pass
                    if attempt % 15 == 0:
                        lines = log_path.read_text(errors="replace").splitlines()
                        progress = lines[-1].strip() if lines else ""
                        if progress and progress != last_progress:
                            print(f"{variant} startup: {progress}", flush=True)
                            last_progress = progress
                    attempt += 1
                    await asyncio.sleep(2)
                else:
                    raise TimeoutError("vLLM startup exceeded the deadline")
                smoke = await request(client, "State the color of grass in one word.", 8)
                save_json(folder / "smoke.json", smoke)
                if smoke["error"]:
                    raise RuntimeError(f"Streaming smoke failed: {smoke['error']}")
                print(f"{variant} streaming smoke passed", flush=True)
                if args.smoke:
                    return
                if args.prefix:
                    shared, _ = long_document(85)
                    shared_prompts = [f"{shared}\nSummarize record {i}." for i in range(args.requests)]
                    unique_prompts = [
                        f"{long_document(86 + i)[0]}\nSummarize record {i}."
                        for i in range(args.requests)
                    ]
                    for workload, prompts in (("shared_prefix", shared_prompts),
                                              ("unique_prefix", unique_prompts)):
                        for trial in range(2):
                            await condition(client, folder, workload, 4, len(prompts), trial, prompts)
                    return

                for workload in SLO_MS:
                    for concurrency in CONCURRENCIES:
                        warmup = await asyncio.gather(*(
                            request(client, prompt_for(workload, 60 + i), 8)
                            for i in range(concurrency)
                        ))
                        if any(row["error"] for row in warmup):
                            save_json(folder / f"{workload}-c{concurrency}-warmup.json", warmup)
                        for repeat in range(args.repeats):
                            await condition(client, folder, workload, concurrency, args.requests, repeat)
                quality = []
                for index, (prompt, expected) in enumerate(quality_cases()):
                    row = await request(client, prompt, 32)
                    actual = re.sub(r"\s*,\s*", ", ", row["text"].strip())
                    quality.append({
                        "id": index, "prompt": prompt, "expected": expected,
                        "correct": row["error"] is None and actual == expected, **row,
                    })
                save_json(folder / "quality.json", quality)
                print(f"{variant} exact-match quality: {sum(row['correct'] for row in quality)}/{len(quality)}")
        except (RuntimeError, TimeoutError):
            print("\n".join(log_path.read_text(errors="replace").splitlines()[-20:]), flush=True)
            raise
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prefix", action="store_true")
    parser.add_argument("--discard-model-cache", action="store_true",
                        help="Use an isolated throwaway model cache per variant.")
    args = parser.parse_args()
    if args.requests < 16 or args.repeats < 1 or (args.smoke and args.prefix):
        parser.error("Use at least 16 requests, one repeat, and at most one of --smoke or --prefix.")
    import torch
    import vllm

    if vllm.__version__ != "0.11.2":
        parser.error(f"Install vLLM 0.11.2. Found {vllm.__version__}.")
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0)[0] < 8:
        parser.error("Expose one Ampere-or-newer CUDA GPU with BF16 support.")
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "environment.json", {
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
                                        "--format=csv,noheader"], text=True).strip(),
        "torch": torch.__version__, "vllm": vllm.__version__, "cuda": torch.version.cuda,
        "python": sys.version, "models": MODELS, "slo_ms": SLO_MS,
        "client": "loopback, closed-loop concurrency, no client-side queue time",
    })
    modes = [("awq", False), ("awq", True)] if args.prefix else [(v, False) for v in MODELS]
    failed = []
    for variant, cache_enabled in modes:
        label = f"{variant}-prefix-{cache_enabled}" if args.prefix else variant
        cache_dir = (args.output / ".models" / label) if args.discard_model_cache else args.cache
        try:
            asyncio.run(run_server(variant, args.output / label, cache_dir, args,
                                   prefix_cache=cache_enabled))
        except (RuntimeError, TimeoutError) as exc:
            failed.append(label)
            save_json(args.output / label / "failure.json", {"error": str(exc)})
            print(f"{label}: {exc}", flush=True)
        finally:
            if args.discard_model_cache and cache_dir.exists():
                shutil.rmtree(cache_dir)
    if failed:
        raise SystemExit(f"Failed runs: {', '.join(failed)}")


if __name__ == "__main__":
    main()
