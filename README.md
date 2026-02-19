# CodeFormer - Robust Face Restoration

A Replicate Cog implementation of [CodeFormer](https://github.com/sczhou/CodeFormer), a robust face restoration model that leverages a codebook lookup transformer for high-quality face restoration.

## Model Description

CodeFormer is a state-of-the-art face restoration algorithm that can:
- Restore old/degraded face photos
- Enhance low-quality face images
- Fix AI-generated face artifacts
- Improve face quality without global image upscaling

The model uses a discrete codebook prior learned from high-quality faces combined with a transformer architecture to achieve natural and high-fidelity face restoration.

## Usage

### Basic Usage

```bash
cog predict --json '{"images":["data/1.jpg"]}'
```

### Multiple Images (up to 4)

```bash
cog predict --json '{"images":["data/1.jpg","data/2.jpg"]}'
```

### With Custom Parameters

```bash
cog predict \
  --json '{"images":["data/1.jpg"],"codeformer_fidelity":0.7,"runtime_profile":"fast","stability_mode":"optimized","compile_backend":"eager"}'
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | List[Path] | required | Input image(s) - up to 4 images |
| `codeformer_fidelity` | float | 0.5 | Balance between quality (0) and fidelity (1). Lower values produce higher quality but may alter facial features. |
| `runtime_profile` | string | "baseline" | Runtime profile: "baseline", "fast", or "max_speed" |
| `stability_mode` | string | "safe" | Execution mode: "safe" (recommended on Replicate) or "optimized" (highest throughput) |
| `compile_backend` | string | "eager" | `torch.compile` backend used only when `stability_mode=optimized` |

Notes:
- `compile_backend` is initialization-time for a warm predictor process in optimized mode. After optimized runtime is initialized, changing this value in later requests returns a validation error.

## Recommended Production Parameters

For sustained high-load inference (single image per request), use:

- `runtime_profile`: `max_speed`
- `stability_mode`: `optimized`
- `compile_backend`: `none`
- `codeformer_fidelity`: `0.5` (tune for quality preference; usually small speed impact)

Recommended JSON input:

```json
{
  "input": {
    "images": ["https://.../input.jpg"],
    "runtime_profile": "max_speed",
    "stability_mode": "optimized",
    "compile_backend": "none",
    "codeformer_fidelity": 0.5
  }
}
```

## Output

Returns a list of restored images corresponding to each input image.
If any input fails to load/process, the prediction fails explicitly with a detailed error (no silent partial-success output).
Outputs are always JPG. Background enhancement is disabled and upscale is fixed to `1`.

### `cog serve` HTTP Request

```bash
curl http://localhost:8080/predictions -X POST \
  -H 'Content-Type: application/json' \
  -d '{"input":{"images":["https://.../input.jpg"],"runtime_profile":"baseline","stability_mode":"safe","compile_backend":"eager"}}'
```

## Benchmarking

Use the benchmark runner to compare runtime profiles via repeated local `cog serve` HTTP calls:

```bash
python benchmarks/run_bench.py --image-dir data --batch-size 1 --profiles max_speed --stability-mode optimized --compile-backend none --warmup 1 --repeats 3 --port 6301
```

The script auto-discovers `data/*.jpg` when `--images` is not provided and executes prediction calls through a local `cog serve` instance. It now reports:
- `round_totals`: repeat-level totals (one sample per repeat)
- `per_request`: request-level latency distribution (p50/p95/p99)
- `per_image`: image-level latency distribution (same as request-level when `--batch-size 1`)

Important:
- Use a free `--port` each run. The benchmark now fails fast if the port is already in use to avoid stale-server contamination.

### Single-Image Benchmark

Find the fastest single-image config (cold and warm winners):

```bash
python benchmarks/bench_single_image.py --image data/1.jpg --warmup 1 --repeats 5
```

This writes `benchmarks/last_bench_single_image.json` and reports both:
- `best_warm`: best steady-state latency
- `best_cold`: best first-request latency

### Runtime Profiles

- `baseline`: original behavior with no additional speed-focused settings.
- `fast`: enables mixed precision and batched face restoration.
- `max_speed`: fastest profile; keeps batching and precision optimizations and skips face parsing during paste-back.

### Stability Modes

- `safe` (default): reliability-first mode for Replicate. Disables high-risk GPU runtime optimizations.
- `optimized`: enables `torch.compile` and channels-last for maximum throughput.

### Optimization Results (Current)

Measured on all `data/*.jpg` (100 images), `batch_size=1`, `runtime_profile=max_speed`, `stability_mode=optimized`, `warmup=1`, `repeats=3`, request-level stats:

| compile_backend | Avg / request (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (img/s) |
|----------------|-------------------:|---------:|---------:|---------:|-------------------:|
| `none` | 126.04 | 122.69 | 146.76 | 151.77 | 7.93 |
| `eager` | 135.26 | 132.34 | 153.41 | 158.25 | 7.39 |

Main takeaways:
- For sustained load in this implementation, `compile_backend=none` outperformed `eager` on avg, p50, and p95.
- Recommended sustained-load settings: `runtime_profile=max_speed`, `stability_mode=optimized`, `compile_backend=none`.
- Replicate wall-clock can be higher than model stage totals (platform/network overhead). Stage logs around `~160ms` can still correspond to `~0.3s` end-to-end requests.

### Full Dataset Benchmark (100 images)

Run all images in `data/` (auto-discovered) with single-image request latency tracking:

```bash
python benchmarks/run_bench.py --image-dir data --batch-size 1 --profiles max_speed --stability-mode optimized --compile-backend none --warmup 1 --repeats 3 --port 6301 --output benchmarks/last_bench_100.json
```

## Model Weights

Model weights are automatically downloaded on first run:
- CodeFormer face restoration model
- RetinaFace face detection model
- ParseNet face parsing model

## References

- [CodeFormer Paper](https://arxiv.org/abs/2206.11253) - "Towards Robust Blind Face Restoration with Codebook Lookup Transformer"
- [Official Repository](https://github.com/sczhou/CodeFormer)

## Citation

```bibtex
@inproceedings{zhou2022codeformer,
  title={Towards Robust Blind Face Restoration with Codebook Lookup Transformer},
  author={Zhou, Shangchen and Chan, Kelvin C.K. and Li, Chongyi and Loy, Chen Change},
  booktitle={NeurIPS},
  year={2022}
}
```

## License

This implementation follows the license of the original CodeFormer repository. Please refer to the [official repository](https://github.com/sczhou/CodeFormer) for license details.
