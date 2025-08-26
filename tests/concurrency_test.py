import argparse
import asyncio
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except Exception:
    print("Please install httpx: uv add httpx OR pip install httpx", file=sys.stderr)
    sys.exit(1)

# uv run tests/concurrency_test.py --host http://localhost:8080 --text "Привет! Это параллельный тест." --concurrency 4 --total 16 --response_format wav


async def wait_for_server(host: str, timeout_seconds: float = 120.0) -> None:
    start = time.time()
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=5.0)) as client:
        while True:
            try:
                r = await client.get(f"{host}/voices")
                if r.status_code == 200:
                    return
            except Exception:
                pass
            if time.time() - start > timeout_seconds:
                raise RuntimeError("Server did not become ready in time")
            await asyncio.sleep(0.5)


async def fetch_voices_and_model(host: str) -> Tuple[List[str], str]:
    async with httpx.AsyncClient(timeout=None) as client:
        vr = await client.get(f"{host}/voices")
        vr.raise_for_status()
        voices_json = vr.json()
        voices = [v["name"] for v in voices_json.get("voices", [])]

        mr = await client.get(f"{host}/models")
        mr.raise_for_status()
        models_json = mr.json()
        model = models_json.get("models", [{}])[0].get("name")
        return voices, model


async def one_request(
    client: httpx.AsyncClient,
    host: str,
    model: str,
    voice: str,
    text: str,
    response_format: str,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    r = await client.post(
        f"{host}/v1/audio/speech",
        json={
            "input": text,
            "voice": voice,
            "model": model,
            "response_format": response_format,
        },
    )
    dt = time.perf_counter() - t0
    try:
        r.raise_for_status()
        # Consume content to free connection pool slot
        _ = r.content
        return {"ok": True, "status": r.status_code, "latency_s": dt, "size": len(_)}
    except httpx.HTTPError as e:
        return {"ok": False, "status": r.status_code if r is not None else None, "latency_s": dt, "error": str(e)}


async def one_stream_request(
    host: str,
    model: str,
    voice: str,
    text: str,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream(
                "POST",
                f"{host}/v1/audio/speech/stream",
                json={
                    "input": text,
                    "voice": voice,
                    "model": model,
                    "response_format": "pcm16",
                },
            ) as r:
                r.raise_for_status()
                size = 0
                async for chunk in r.aiter_bytes():
                    size += len(chunk)
                dt = time.perf_counter() - t0
                return {"ok": True, "status": r.status_code, "latency_s": dt, "size": size}
        except httpx.HTTPError as e:
            dt = time.perf_counter() - t0
            return {"ok": False, "status": getattr(e.response, "status_code", None), "latency_s": dt, "error": str(e)}


async def run_load(
    host: str,
    text: str,
    voice: Optional[str],
    model: Optional[str],
    response_format: str,
    concurrency: int,
    total_requests: int,
    stream: bool,
) -> None:
    await wait_for_server(host)
    voices, model_name = await fetch_voices_and_model(host)
    if not voices:
        raise RuntimeError("No voices available on server")
    if voice is None:
        voice = voices[0]
    if model is None:
        model = model_name

    print(f"Using model={model}, voice={voice}, stream={stream}, concurrency={concurrency}, total={total_requests}")

    # Prepare tasks
    results: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(concurrency)

    async def bound_task(i: int) -> None:
        async with sem:
            if stream:
                res = await one_stream_request(host, model, voice, f"{text} (req {i})")
            else:
                async with httpx.AsyncClient(timeout=None) as client:
                    res = await one_request(client, host, model, voice, f"{text} (req {i})", response_format)
            results.append(res)

    # Issue tasks
    await asyncio.gather(*(bound_task(i) for i in range(total_requests)))

    # Aggregate
    oks = [r for r in results if r.get("ok")]
    errs = [r for r in results if not r.get("ok")]
    latencies = [r["latency_s"] for r in oks]
    total_bytes = sum(r.get("size", 0) for r in oks)
    p50 = sorted(latencies)[len(latencies) // 2] if latencies else None
    p95 = sorted(latencies)[int(len(latencies) * 0.95) - 1] if latencies else None
    print(json.dumps({
        "success": len(oks),
        "errors": len(errs),
        "p50_s": p50,
        "p95_s": p95,
        "avg_s": (sum(latencies) / len(latencies)) if latencies else None,
        "bytes": total_bytes,
    }, ensure_ascii=False, indent=2))
    if errs:
        print("Sample error:", errs[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000", help="Server host, e.g. http://localhost:8000")
    parser.add_argument("--text", default="Привет, это тест", help="Text to synthesize")
    parser.add_argument("--voice", default=None, help="Voice name (defaults to first)")
    parser.add_argument("--model", default=None, help="Model name (defaults to server current)")
    parser.add_argument("--response_format", default="wav", choices=["wav", "pcm16"], help="Response format (non-stream)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--total", type=int, default=8)
    parser.add_argument("--stream", action="store_true", help="Use streaming endpoint")
    args = parser.parse_args()

    asyncio.run(run_load(
        host=args.host,
        text=args.text,
        voice=args.voice,
        model=args.model,
        response_format=args.response_format,
        concurrency=args.concurrency,
        total_requests=args.total,
        stream=args.stream,
    ))


if __name__ == "__main__":
    main()


