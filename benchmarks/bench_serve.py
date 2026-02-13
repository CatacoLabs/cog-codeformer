#!/usr/bin/env python3
"""Production-accurate benchmark for CodeFormer via cog serve.

Starts `cog serve`, waits for setup(), then sends predictions via HTTP —
matching exactly how Cog models run in production.

Usage (from host, with container already built):
    python benchmarks/bench_serve.py --image cog-cf-h100 --profiles baseline fast max_speed
    python benchmarks/bench_serve.py --image cog-codeformer  # no profiles for original
"""
import argparse
import base64
import json
import os
import subprocess
import statistics
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def summarize_latencies(values_ms: List[float]) -> Dict[str, float]:
    if not values_ms:
        return {}
    return {
        "num_samples": len(values_ms),
        "avg_ms": round(statistics.mean(values_ms), 2),
        "p50_ms": round(percentile(values_ms, 0.50), 2),
        "p95_ms": round(percentile(values_ms, 0.95), 2),
        "p99_ms": round(percentile(values_ms, 0.99), 2),
        "min_ms": round(min(values_ms), 2),
        "max_ms": round(max(values_ms), 2),
    }


def image_sort_key(path: Path):
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def encode_image_base64(path: Path) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    suffix = path.suffix.lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        suffix, "image/jpeg"
    )
    return f"data:{mime};base64,{b64}"


def wait_for_ready(base_url: str, timeout: int = 600) -> bool:
    """Wait for cog serve to complete setup and be READY."""
    health_url = base_url.rstrip("/") + "/health-check"
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(health_url)
            resp = urllib.request.urlopen(req, timeout=5)
            body = json.loads(resp.read().decode("utf-8"))
            status = body.get("status", "")
            setup_status = body.get("setup", {}).get("status", "")
            if status == "READY" and setup_status == "succeeded":
                return True
            print(f"  status={status}, setup={setup_status}, waiting...")
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(2)
    return False


def send_prediction(
    base_url: str,
    image_data_uris: List[str],
    codeformer_fidelity: float = 0.5,
    background_enhance: bool = True,
    face_upsample: bool = True,
    upscale: int = 2,
    output_format: str = "jpg",
    runtime_profile: Optional[str] = None,
    timeout: int = 600,
    retries: int = 3,
) -> Dict:
    """Send prediction request with retry on 409 Conflict."""
    predict_url = base_url.rstrip("/") + "/predictions"
    input_data = {
        "images": image_data_uris,
        "codeformer_fidelity": codeformer_fidelity,
        "background_enhance": background_enhance,
        "face_upsample": face_upsample,
        "upscale": upscale,
        "output_format": output_format,
    }
    if runtime_profile is not None:
        input_data["runtime_profile"] = runtime_profile

    payload = json.dumps({"input": input_data}).encode("utf-8")

    for attempt in range(retries):
        req = urllib.request.Request(
            predict_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            t0 = time.perf_counter()
            resp = urllib.request.urlopen(req, timeout=timeout)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            body = json.loads(resp.read().decode("utf-8"))

            status = body.get("status", "unknown")
            if status == "failed":
                error = body.get("error", "unknown error")
                print(f"    Prediction failed: {error}")
                return {"elapsed_ms": elapsed_ms, "status": "failed", "error": error}

            return {
                "elapsed_ms": elapsed_ms,
                "status": status,
                "output_count": len(body.get("output", [])) if isinstance(body.get("output"), list) else 0,
                "metrics": body.get("metrics", {}),
            }
        except urllib.error.HTTPError as e:
            if e.code == 409 and attempt < retries - 1:
                print(f"    409 Conflict (server busy), retrying in 5s...")
                time.sleep(5)
                continue
            raise

    return {"elapsed_ms": 0, "status": "error", "error": "max retries exceeded"}


def chunked(items: list, size: int) -> list:
    return [items[i : i + size] for i in range(0, len(items), size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production benchmark for CodeFormer via cog serve"
    )
    parser.add_argument(
        "--image", required=True,
        help="Docker image name (e.g., cog-cf-h100 or cog-codeformer)",
    )
    parser.add_argument("--image-dir", default="/home/shadeform/Code/cog-cf-h100/data")
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--codeformer-fidelity", type=float, default=0.5)
    parser.add_argument("--background-enhance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--face-upsample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upscale", type=int, default=2)
    parser.add_argument("--output-format", default="jpg", choices=["png", "jpg"])
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--weights-dir", default="/home/shadeform/Code/cog-cf-h100/weights")
    return parser.parse_args()


def start_cog_serve(image: str, port: int, weights_dir: str) -> subprocess.Popen:
    cmd = [
        "docker", "run",
        "--gpus", "all",
        "--rm",
        "-p", f"{port}:5000",
        "-v", f"{weights_dir}:/src/weights:ro",
        image,
    ]
    print(f"Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return proc


def main() -> None:
    args = parse_args()

    # Discover and encode images
    image_dir = Path(args.image_dir)
    image_paths = sorted(image_dir.glob("*.jpg"), key=image_sort_key)
    if not image_paths:
        raise FileNotFoundError(f"No .jpg files in {image_dir}")
    image_paths = image_paths[: args.num_images]
    print(f"Encoding {len(image_paths)} images as base64...")
    image_data_uris = [encode_image_base64(p) for p in image_paths]
    image_batches = chunked(image_data_uris, args.batch_size)
    print(f"  {len(image_batches)} batches of up to {args.batch_size} images")

    # Determine profiles
    profiles = args.profiles
    if profiles is None:
        if "cog-codeformer" in args.image:
            profiles = [None]
        else:
            profiles = ["baseline", "fast", "max_speed"]

    # Start server
    base_url = f"http://localhost:{args.port}"
    proc = start_cog_serve(args.image, args.port, args.weights_dir)

    try:
        print(f"Waiting for server setup at {base_url}...")
        if not wait_for_ready(base_url, timeout=600):
            print("ERROR: Server did not become ready")
            sys.exit(1)
        print("Server ready!\n")

        metadata = {
            "image": args.image,
            "num_images": len(image_paths),
            "batch_size": args.batch_size,
            "num_batches": len(image_batches),
            "profiles": [p or "original" for p in profiles],
            "warmup": args.warmup,
            "repeats": args.repeats,
            "params": {
                "codeformer_fidelity": args.codeformer_fidelity,
                "background_enhance": args.background_enhance,
                "face_upsample": args.face_upsample,
                "upscale": args.upscale,
                "output_format": args.output_format,
            },
            "mode": "cog_serve_http",
        }

        all_results = {}
        started = time.time()

        for profile in profiles:
            profile_name = profile or "original"
            print(f"\n{'='*60}")
            print(f"Profile: {profile_name}")
            print(f"{'='*60}")

            # Warmup — torch.compile can be very slow on first call
            print(f"  Warmup ({args.warmup} rounds)...")
            for w in range(args.warmup):
                for batch in image_batches:
                    result = send_prediction(
                        base_url, batch,
                        codeformer_fidelity=args.codeformer_fidelity,
                        background_enhance=args.background_enhance,
                        face_upsample=args.face_upsample,
                        upscale=args.upscale,
                        output_format=args.output_format,
                        runtime_profile=profile,
                        timeout=600,
                    )
                    if result.get("status") == "failed":
                        print(f"    WARNING: warmup prediction failed")
                print(f"    warmup {w+1}/{args.warmup} done")

            # Measured rounds
            print(f"  Measured ({args.repeats} rounds)...")
            round_totals_ms: List[float] = []
            batch_latencies_ms: List[float] = []

            for r in range(args.repeats):
                round_total = 0.0
                for batch in image_batches:
                    result = send_prediction(
                        base_url, batch,
                        codeformer_fidelity=args.codeformer_fidelity,
                        background_enhance=args.background_enhance,
                        face_upsample=args.face_upsample,
                        upscale=args.upscale,
                        output_format=args.output_format,
                        runtime_profile=profile,
                        timeout=600,
                    )
                    round_total += result["elapsed_ms"]
                    batch_latencies_ms.append(result["elapsed_ms"])

                round_totals_ms.append(round_total)
                avg_per_img = round_total / len(image_paths)
                print(
                    f"    round {r+1}/{args.repeats}: "
                    f"total={round_total:.1f}ms, "
                    f"avg/img={avg_per_img:.1f}ms"
                )

            avg_round = statistics.mean(round_totals_ms)
            throughput = (len(image_paths) / avg_round) * 1000.0

            profile_result = {
                "round_totals": summarize_latencies(round_totals_ms),
                "per_batch": summarize_latencies(batch_latencies_ms),
                "per_image_avg_ms": round(avg_round / len(image_paths), 2),
                "throughput_img_per_s": round(throughput, 3),
            }
            all_results[profile_name] = profile_result

            print(f"\n  Summary for {profile_name}:")
            print(f"    Round total (20 imgs): avg={avg_round:.1f}ms")
            print(f"    Per-image avg:         {avg_round/len(image_paths):.1f}ms")
            print(f"    Per-batch avg:         {statistics.mean(batch_latencies_ms):.1f}ms")
            print(f"    Throughput:            {throughput:.2f} img/s")

        payload = {
            "metadata": {
                **metadata,
                "total_elapsed_s": round(time.time() - started, 2),
            },
            "results": all_results,
        }

        if args.output:
            out_path = Path(args.output)
        else:
            out_path = Path(f"benchmarks/last_bench_serve_{args.image}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        for pname, presult in all_results.items():
            rt = presult["round_totals"]
            print(
                f"  {pname:12s}: avg={rt['avg_ms']:.0f}ms  "
                f"p50={rt['p50_ms']:.0f}ms  "
                f"throughput={presult['throughput_img_per_s']:.2f} img/s"
            )
        print(f"\nSaved to {out_path}")

    finally:
        print("\nStopping server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("Server stopped.")


if __name__ == "__main__":
    main()
