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
cog predict -i images=@input.jpg
```

### Multiple Images (up to 4)

```bash
cog predict -i images=@photo1.jpg -i images=@photo2.jpg
```

### With Custom Parameters

```bash
cog predict \
  -i images=@input.jpg \
  -i codeformer_fidelity=0.7 \
  -i runtime_profile=fast \
  -i stability_mode=optimized \
  -i compile_backend=eager
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | List[Path] | required | Input image(s) - up to 4 images |
| `codeformer_fidelity` | float | 0.5 | Balance between quality (0) and fidelity (1). Lower values produce higher quality but may alter facial features. |
| `runtime_profile` | string | "baseline" | Runtime profile: "baseline", "fast", or "max_speed" |
| `stability_mode` | string | "safe" | Execution mode: "safe" (recommended on Replicate) or "optimized" (highest throughput) |
| `compile_backend` | string | "eager" | `torch.compile` backend used only when `stability_mode=optimized` |

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

Use the benchmark runner to compare runtime profiles via repeated `cog predict` calls:

```bash
python benchmarks/run_bench.py --image-dir data --batch-size 4 --profiles baseline fast max_speed --repeats 3
```

The script auto-discovers `data/*.jpg` when `--images` is not provided and executes batched `cog predict` calls (default 4 images per request to match the predictor limit). It then saves per-profile latency percentiles and throughput.

### Runtime Profiles

- `baseline`: original behavior with no additional speed-focused settings.
- `fast`: enables mixed precision and batched face restoration.
- `max_speed`: fastest profile; keeps batching and precision optimizations and skips face parsing during paste-back.

### Stability Modes

- `safe` (default): reliability-first mode for Replicate. Disables high-risk GPU runtime optimizations.
- `optimized`: enables `torch.compile` and channels-last for maximum throughput.

### Optimization Results (Current)

Measured with `data/1.jpg` and `data/2.jpg`, `warmup=1`, `repeats=3`:

| Profile | Avg Latency (ms) | P95 (ms) | Throughput (img/s) | Delta vs Baseline |
|---------|------------------:|---------:|-------------------:|------------------:|
| `baseline` | 13425.92 | 13559.31 | 0.14897 | - |
| `fast` | 13250.94 | 13430.49 | 0.15093 | +1.30% |
| `max_speed` | 12985.76 | 13080.51 | 0.15401 | +3.28% |

Main takeaways:
- End-to-end speedup is real but modest in `cog predict` mode.
- Biggest gains came from face batching and mixed precision.
- `max_speed` is currently the fastest profile.
- `fast` is a middle-ground profile when you want speedups with more conservative quality behavior.

### Full Dataset Benchmark (100 images)

Run all images in `data/` (auto-discovered) with request chunking:

```bash
python benchmarks/run_bench.py --image-dir data --batch-size 4 --profiles baseline fast max_speed --warmup 1 --repeats 3 --output benchmarks/last_bench_100.json
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
