#!/usr/bin/env python3
"""In-process benchmark for cog-cf-h100 CodeFormer.

Imports Predictor directly (no subprocess overhead), runs warmup + measured
rounds, and collects per-stage timing from predictor.last_stats.

Usage:
    cd /home/shadeform/Code/cog-cf-h100
    python -m benchmarks.bench_inprocess [--profiles baseline fast max_speed]
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

# Ensure project root is on sys.path so local imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from predict import Predictor  # noqa: E402


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
        "avg_ms": statistics.mean(values_ms),
        "p50_ms": percentile(values_ms, 0.50),
        "p95_ms": percentile(values_ms, 0.95),
        "p99_ms": percentile(values_ms, 0.99),
        "min_ms": min(values_ms),
        "max_ms": max(values_ms),
    }


def aggregate_stage_times(all_stats: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-stage times across all runs."""
    if not all_stats:
        return {}
    stage_keys = [k for k in all_stats[0].keys() if k.endswith("_ms")]
    result = {}
    for key in stage_keys:
        values = [s[key] for s in all_stats if key in s]
        if values:
            result[key] = summarize_latencies(values)
    return result


def image_sort_key(path: Path):
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="In-process benchmark for cog-cf-h100")
    parser.add_argument(
        "--image-dir", default="data",
        help="Directory containing test images (default: data/)",
    )
    parser.add_argument(
        "--num-images", type=int, default=20,
        help="Number of images to use (default: 20)",
    )
    parser.add_argument(
        "--profiles", nargs="+",
        default=["baseline", "fast", "max_speed"],
        choices=["baseline", "fast", "max_speed"],
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup rounds")
    parser.add_argument("--repeats", type=int, default=5, help="Measured rounds")
    parser.add_argument("--codeformer-fidelity", type=float, default=0.5)
    parser.add_argument("--stability-mode", default="safe", choices=["safe", "optimized"])
    parser.add_argument(
        "--compile-backend",
        default="none",
        choices=["none", "eager", "aot_eager", "inductor"],
    )
    parser.add_argument(
        "--output", default="benchmarks/last_bench_inprocess.json",
        help="Output JSON file",
    )
    return parser.parse_args()


def run_benchmark(
    predictor: Predictor,
    image_paths: List[Path],
    profile: str,
    args: argparse.Namespace,
) -> List[Dict[str, float]]:
    """Process all images once, return list of per-image stats."""
    all_stats = []
    predictor.runtime_profile = profile
    predictor.stability_mode = args.stability_mode
    predictor.compile_backend = args.compile_backend
    if predictor.stability_mode == "optimized":
        predictor._ensure_optimized_runtime(compile_backend=args.compile_backend)

    for img_path in image_paths:
        import tempfile
        out_dir = Path(tempfile.mkdtemp())
        out_path = out_dir / "out.jpg"
        predictor._process_single_image(
            image=img_path,
            codeformer_fidelity=args.codeformer_fidelity,
            output_path=out_path,
            runtime_profile=profile,
        )
        all_stats.append(dict(predictor.last_stats))
    return all_stats


def main() -> None:
    args = parse_args()

    # Discover images
    image_dir = Path(args.image_dir)
    image_paths = sorted(image_dir.glob("*.jpg"), key=image_sort_key)
    if not image_paths:
        raise FileNotFoundError(f"No .jpg files in {image_dir}")
    image_paths = image_paths[: args.num_images]
    print(f"Using {len(image_paths)} images from {image_dir}")

    # Setup predictor once
    print("Setting up predictor...")
    predictor = Predictor()
    predictor.setup()
    print("Setup complete.\n")

    metadata = {
        "implementation": "cog-cf-h100",
        "images": [str(p) for p in image_paths],
        "num_images": len(image_paths),
        "profiles": args.profiles,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "params": {
            "codeformer_fidelity": args.codeformer_fidelity,
            "stability_mode": args.stability_mode,
            "compile_backend": args.compile_backend,
        },
        "mode": "in_process",
    }

    all_results: Dict[str, object] = {}
    started = time.time()

    for profile in args.profiles:
        print(f"\n{'='*60}")
        print(f"Profile: {profile}")
        print(f"{'='*60}")

        # Warmup
        print(f"  Warmup ({args.warmup} rounds)...")
        for w in range(args.warmup):
            run_benchmark(predictor, image_paths, profile, args)
            print(f"    warmup {w+1}/{args.warmup} done")

        # Measured runs
        print(f"  Measured ({args.repeats} rounds)...")
        round_totals_ms: List[float] = []
        all_round_stats: List[List[Dict[str, float]]] = []

        for r in range(args.repeats):
            t0 = time.perf_counter()
            round_stats = run_benchmark(predictor, image_paths, profile, args)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            round_totals_ms.append(elapsed_ms)
            all_round_stats.append(round_stats)

            avg_per_image = elapsed_ms / len(image_paths)
            print(
                f"    round {r+1}/{args.repeats}: "
                f"total={elapsed_ms:.1f}ms, "
                f"avg/img={avg_per_image:.1f}ms"
            )

        # Flatten all per-image stats from measured rounds
        flat_stats = [s for round_stats in all_round_stats for s in round_stats]

        # Per-image latency summary (from last_stats total_ms)
        per_image_totals = [s.get("total_ms", 0.0) for s in flat_stats]

        # Throughput
        avg_round_ms = statistics.mean(round_totals_ms)
        throughput_img_s = (len(image_paths) / avg_round_ms) * 1000.0

        profile_result = {
            "round_totals": summarize_latencies(round_totals_ms),
            "per_image": summarize_latencies(per_image_totals),
            "throughput_img_per_s": round(throughput_img_s, 3),
            "stage_breakdown": aggregate_stage_times(flat_stats),
        }
        all_results[profile] = profile_result

        print(f"\n  Summary for {profile}:")
        print(f"    Round total:  avg={avg_round_ms:.1f}ms")
        print(f"    Per-image:    avg={statistics.mean(per_image_totals):.1f}ms")
        print(f"    Throughput:   {throughput_img_s:.2f} img/s")

        # Stage breakdown
        stage_bk = profile_result["stage_breakdown"]
        if stage_bk:
            print(f"    Stage breakdown (avg ms):")
            for stage, vals in sorted(stage_bk.items()):
                print(f"      {stage:20s}: {vals['avg_ms']:8.1f}")

    payload = {
        "metadata": {
            **metadata,
            "total_elapsed_s": round(time.time() - started, 2),
        },
        "results": all_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n{'='*60}")
    print(f"Saved benchmark report to {out_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
