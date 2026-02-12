#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, List


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    idx = (len(values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def summarize(values_ms: List[float], images_per_call: float) -> Dict[str, float]:
    values_ms = sorted(values_ms)
    if not values_ms:
        return {}
    avg_ms = statistics.mean(values_ms)
    return {
        "num_samples": len(values_ms),
        "avg_total_ms": avg_ms,
        "p50_total_ms": percentile(values_ms, 0.50),
        "p95_total_ms": percentile(values_ms, 0.95),
        "p99_total_ms": percentile(values_ms, 0.99),
        "throughput_req_s": 1000.0 / avg_ms,
        "throughput_img_s": (1000.0 / avg_ms) * images_per_call,
    }


def chunked(items: List[Path], size: int) -> List[List[Path]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def image_sort_key(path: Path):
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CodeFormer by invoking `cog predict`."
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
        help="Images per cog predict call (must be <= 4 for current predictor).",
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
        "--background-enhance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable background enhancement.",
    )
    parser.add_argument(
        "--face-upsample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable face upsample.",
    )
    parser.add_argument("--upscale", type=int, default=2)
    parser.add_argument(
        "--output-format",
        default="png",
        choices=["png", "jpg"],
    )
    parser.add_argument(
        "--output",
        default="benchmarks/last_bench.json",
        help="Output json file path.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=1800,
        help="Timeout per `cog predict` call.",
    )
    return parser.parse_args()


def build_cmd(args: argparse.Namespace, images: List[Path], profile: str) -> List[str]:
    cmd = [
        "cog",
        "predict",
        "-i",
        f"runtime_profile={profile}",
        "-i",
        f"codeformer_fidelity={args.codeformer_fidelity}",
        "-i",
        f"background_enhance={'true' if args.background_enhance else 'false'}",
        "-i",
        f"face_upsample={'true' if args.face_upsample else 'false'}",
        "-i",
        f"upscale={args.upscale}",
        "-i",
        f"output_format={args.output_format}",
    ]
    for image in images:
        cmd.extend(["-i", f"images=@{image}"])
    return cmd


def timed_call(cmd: List[str], timeout_sec: int) -> Dict[str, object]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "elapsed_ms": elapsed_ms,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


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

    metadata = {
        "images": [str(p) for p in image_paths],
        "num_images": len(image_paths),
        "batch_size": args.batch_size,
        "num_batches_per_repeat": len(image_batches),
        "profiles": args.profiles,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "mode": "cog_predict_subprocess",
    }

    all_results: Dict[str, Dict[str, object]] = {}
    started = time.time()

    for profile in args.profiles:
        print(f"\n=== Profile: {profile} ===")

        for _ in range(args.warmup):
            for batch in image_batches:
                warmup_cmd = build_cmd(args, batch, profile)
                _ = timed_call(warmup_cmd, args.timeout_sec)

        latencies_ms: List[float] = []
        failures: List[Dict[str, object]] = []
        for rep in range(args.repeats):
            rep_total_ms = 0.0
            rep_failed = False
            for batch_idx, batch in enumerate(image_batches, start=1):
                cmd = build_cmd(args, batch, profile)
                result = timed_call(cmd, args.timeout_sec)
                if result["returncode"] != 0:
                    rep_failed = True
                    failures.append(
                        {
                            "rep": rep + 1,
                            "batch_idx": batch_idx,
                            "batch_images": [p.name for p in batch],
                            "returncode": result["returncode"],
                            "stderr_tail": str(result["stderr"])[-1000:],
                        }
                    )
                    print(
                        f"rep={rep + 1} batch={batch_idx}/{len(image_batches)} failed "
                        f"returncode={result['returncode']}"
                    )
                    break
                rep_total_ms += float(result["elapsed_ms"])

            if not rep_failed:
                latencies_ms.append(rep_total_ms)
                print(
                    f"rep={rep + 1} images={len(image_paths)} "
                    f"elapsed_ms={rep_total_ms:.2f}"
                )

        all_results[profile] = {
            "summary": summarize(latencies_ms, images_per_call=float(len(image_paths))),
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


if __name__ == "__main__":
    main()
