#!/usr/bin/env python3
import argparse
import base64
import json
import socket
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Dict, List


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = (len(values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def summarize(values_ms: List[float]) -> Dict[str, float]:
    if not values_ms:
        return {}
    avg_ms = statistics.mean(values_ms)
    return {
        "num_samples": len(values_ms),
        "avg_ms": avg_ms,
        "p50_ms": percentile(values_ms, 0.50),
        "p95_ms": percentile(values_ms, 0.95),
        "p99_ms": percentile(values_ms, 0.99),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
    }


def chunked(items: List[Path], size: int) -> List[List[Path]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def image_sort_key(path: Path):
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def encode_image_base64(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        suffix, "image/jpeg"
    )
    return f"data:{mime};base64,{payload}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CodeFormer through `cog serve` HTTP predictions."
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Input images. If omitted, all jpg files in --image-dir are used.",
    )
    parser.add_argument(
        "--image-dir",
        default="data",
        help="Directory used for auto-discovery when --images is omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Images per prediction call (must be <= 4 for current predictor).",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["baseline", "fast", "max_speed"],
        choices=["baseline", "fast", "max_speed"],
        help="Runtime profiles to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup rounds.")
    parser.add_argument("--repeats", type=int, default=5, help="Measured rounds.")
    parser.add_argument(
        "--codeformer-fidelity",
        type=float,
        default=0.5,
        dest="codeformer_fidelity",
    )
    parser.add_argument(
        "--stability-mode",
        default="safe",
        choices=["safe", "optimized"],
        help="Predictor stability_mode input.",
    )
    parser.add_argument(
        "--compile-backend",
        default="none",
        choices=["none", "eager", "aot_eager", "inductor"],
        help="Predictor compile_backend input.",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port for cog serve.")
    parser.add_argument(
        "--output",
        default="benchmarks/last_bench.json",
        help="Output json file path.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=1800,
        help="Timeout per prediction request.",
    )
    return parser.parse_args()


def wait_for_ready(base_url: str, timeout_sec: int) -> bool:
    health_url = base_url.rstrip("/") + "/health-check"
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if body.get("status") == "READY" and body.get("setup", {}).get("status") == "succeeded":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def timed_call(
    base_url: str,
    image_data_uris: List[str],
    profile: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    payload = {
        "input": {
            "images": image_data_uris,
            "runtime_profile": profile,
            "codeformer_fidelity": args.codeformer_fidelity,
            "stability_mode": args.stability_mode,
            "compile_backend": args.compile_backend,
        }
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/predictions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        status = body.get("status", "unknown")
        if status != "succeeded":
            return {
                "elapsed_ms": elapsed_ms,
                "ok": False,
                "error": body.get("error", f"prediction status={status}"),
            }
        return {"elapsed_ms": elapsed_ms, "ok": True, "error": ""}
    except Exception as error:
        return {
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            "ok": False,
            "error": str(error),
        }


def start_cog_serve(port: int) -> subprocess.Popen:
    cmd = ["cog", "serve", "--port", str(port), "--progress", "quiet"]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.batch_size > 4:
        raise ValueError("--batch-size must be <= 4 (predictor limit)")

    if args.images:
        image_paths = [Path(p) for p in args.images]
    else:
        image_dir = Path(args.image_dir)
        image_paths = sorted(image_dir.glob("*.jpg"), key=image_sort_key)
        if not image_paths:
            raise FileNotFoundError(
                f"No .jpg files found under image directory: {image_dir}"
            )

    missing = [str(p) for p in image_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input images: {missing}")

    image_batches = chunked(image_paths, args.batch_size)
    image_data_batches = [
        [encode_image_base64(p) for p in batch]
        for batch in image_batches
    ]

    metadata = {
        "images": [str(p) for p in image_paths],
        "num_images": len(image_paths),
        "batch_size": args.batch_size,
        "num_batches_per_repeat": len(image_batches),
        "profiles": args.profiles,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "mode": "cog_serve_http_local",
        "stability_mode": args.stability_mode,
        "compile_backend": args.compile_backend,
    }

    if is_port_in_use(args.port):
        raise RuntimeError(
            f"Port {args.port} is already in use. Use --port with a free port to avoid stale-server benchmarks."
        )

    proc = start_cog_serve(args.port)
    base_url = f"http://127.0.0.1:{args.port}"

    try:
        if not wait_for_ready(base_url, timeout_sec=600):
            raise RuntimeError("cog serve did not become ready within 600s")

        all_results: Dict[str, Dict[str, object]] = {}
        started = time.time()

        for profile in args.profiles:
            print(f"\n=== Profile: {profile} ===")

            for _ in range(args.warmup):
                for batch in image_data_batches:
                    _ = timed_call(base_url, batch, profile, args)

            round_totals_ms: List[float] = []
            request_latencies_ms: List[float] = []
            per_image_latencies_ms: List[float] = []
            failures: List[Dict[str, object]] = []
            for rep in range(args.repeats):
                rep_total_ms = 0.0
                rep_failed = False
                for batch_idx, batch in enumerate(image_data_batches, start=1):
                    result = timed_call(base_url, batch, profile, args)
                    if not bool(result["ok"]):
                        rep_failed = True
                        failures.append(
                            {
                                "rep": rep + 1,
                                "batch_idx": batch_idx,
                                "batch_images": [p.name for p in image_batches[batch_idx - 1]],
                                "error": str(result["error"])[-1000:],
                            }
                        )
                        print(
                            f"rep={rep + 1} batch={batch_idx}/{len(image_batches)} failed"
                        )
                        break
                    elapsed_ms = float(result["elapsed_ms"])
                    batch_size = len(image_batches[batch_idx - 1])
                    rep_total_ms += elapsed_ms
                    request_latencies_ms.append(elapsed_ms)
                    # Weighted image-level approximation for mixed batch sizes.
                    per_image_ms = elapsed_ms / float(batch_size)
                    per_image_latencies_ms.extend([per_image_ms] * batch_size)

                if not rep_failed:
                    round_totals_ms.append(rep_total_ms)
                    print(
                        f"rep={rep + 1} images={len(image_paths)} "
                        f"elapsed_ms={rep_total_ms:.2f}"
                    )

            req_summary = summarize(request_latencies_ms)
            img_summary = summarize(per_image_latencies_ms)
            round_summary = summarize(round_totals_ms)
            if req_summary:
                req_summary["throughput_req_s"] = 1000.0 / req_summary["avg_ms"]
            if img_summary:
                img_summary["throughput_img_s"] = 1000.0 / img_summary["avg_ms"]

            all_results[profile] = {
                "round_totals": round_summary,
                "per_request": req_summary,
                "per_image": img_summary,
                "failures": failures,
            }

        payload = {
            "metadata": {
                **metadata,
                "elapsed_s": time.time() - started,
            },
            "results": all_results,
        }

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print("\n=== Final Summary ===")
        print(json.dumps(payload, indent=2))
        print(f"\nSaved benchmark report to {out_path}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    main()
