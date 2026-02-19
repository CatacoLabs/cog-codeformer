#!/usr/bin/env python3
import argparse
import base64
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-image latency across runtime settings."
    )
    parser.add_argument("--image", default="data/1.jpg", help="Input image path.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["baseline", "fast", "max_speed"],
        choices=["baseline", "fast", "max_speed"],
    )
    parser.add_argument(
        "--stability-modes",
        nargs="+",
        default=["safe", "optimized"],
        choices=["safe", "optimized"],
    )
    parser.add_argument(
        "--compile-backends",
        nargs="+",
        default=["none", "eager"],
        choices=["none", "eager", "aot_eager", "inductor"],
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup calls per config.")
    parser.add_argument("--repeats", type=int, default=5, help="Measured calls per config.")
    parser.add_argument("--codeformer-fidelity", type=float, default=0.5)
    parser.add_argument("--port-base", type=int, default=5400)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument(
        "--output",
        default="benchmarks/last_bench_single_image.json",
        help="Output json file path.",
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


def send_prediction(
    base_url: str,
    data_uri: str,
    cfg: Dict[str, str],
    codeformer_fidelity: float,
    timeout_sec: int,
) -> float:
    payload = {
        "input": {
            "images": [data_uri],
            "codeformer_fidelity": codeformer_fidelity,
            **cfg,
        }
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/predictions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if body.get("status") != "succeeded":
        raise RuntimeError(f"prediction failed: {body}")
    return elapsed_ms


def encode_data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        suffix, "image/jpeg"
    )
    return f"data:{mime};base64,{payload}"


def build_configs(args: argparse.Namespace) -> List[Dict[str, str]]:
    configs: List[Dict[str, str]] = []
    for profile in args.profiles:
        for stability_mode in args.stability_modes:
            backend_candidates = ["none"] if stability_mode == "safe" else args.compile_backends
            for compile_backend in backend_candidates:
                configs.append(
                    {
                        "runtime_profile": profile,
                        "stability_mode": stability_mode,
                        "compile_backend": compile_backend,
                    }
                )
    return configs


def main() -> None:
    args = parse_args()
    image = Path(args.image)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    data_uri = encode_data_uri(image)
    configs = build_configs(args)

    results = []
    for idx, cfg in enumerate(configs, start=1):
        port = args.port_base + idx
        base_url = f"http://127.0.0.1:{port}"
        proc = subprocess.Popen(
            ["cog", "serve", "--port", str(port), "--progress", "quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
        )
        label = f"{cfg['runtime_profile']}|{cfg['stability_mode']}|{cfg['compile_backend']}"
        print(f"\n=== [{idx}/{len(configs)}] {label} ===")
        try:
            if not wait_for_ready(base_url, timeout_sec=600):
                raise RuntimeError("cog serve did not become ready in time")

            cold_ms = send_prediction(
                base_url, data_uri, cfg, args.codeformer_fidelity, args.timeout_sec
            )
            for _ in range(args.warmup):
                _ = send_prediction(
                    base_url, data_uri, cfg, args.codeformer_fidelity, args.timeout_sec
                )
            warm_samples = [
                send_prediction(base_url, data_uri, cfg, args.codeformer_fidelity, args.timeout_sec)
                for _ in range(args.repeats)
            ]

            row = {
                "config": cfg,
                "cold_ms": round(cold_ms, 2),
                "warm_avg_ms": round(statistics.mean(warm_samples), 2),
                "warm_min_ms": round(min(warm_samples), 2),
                "warm_max_ms": round(max(warm_samples), 2),
                "warm_samples_ms": [round(v, 2) for v in warm_samples],
            }
            results.append(row)
            print(row)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    by_warm = sorted(results, key=lambda r: r["warm_avg_ms"])
    by_cold = sorted(results, key=lambda r: r["cold_ms"])
    payload = {
        "metadata": {
            "image": str(image),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "codeformer_fidelity": args.codeformer_fidelity,
            "num_configs": len(results),
        },
        "results": results,
        "best_warm": by_warm[0] if by_warm else None,
        "best_cold": by_cold[0] if by_cold else None,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print("\n=== Best warm config ===")
    print(json.dumps(payload["best_warm"], indent=2))
    print("\n=== Best cold config ===")
    print(json.dumps(payload["best_cold"], indent=2))
    print(f"\nSaved benchmark report to {out_path}")


if __name__ == "__main__":
    main()
