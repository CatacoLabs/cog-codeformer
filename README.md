# CodeFormer - Robust Face Restoration

A Replicate Cog implementation of [CodeFormer](https://github.com/sczhou/CodeFormer), a robust face restoration model that leverages a codebook lookup transformer for high-quality face restoration.

## Model Description

CodeFormer is a state-of-the-art face restoration algorithm that can:
- Restore old/degraded face photos
- Enhance low-quality face images
- Fix AI-generated face artifacts
- Upscale face images with background enhancement

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
  -i background_enhance=true \
  -i face_upsample=true \
  -i upscale=2 \
  -i output_format=png
```

## Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | List[Path] | required | Input image(s) - up to 4 images |
| `codeformer_fidelity` | float | 0.5 | Balance between quality (0) and fidelity (1). Lower values produce higher quality but may alter facial features. |
| `background_enhance` | bool | true | Enhance background with Real-ESRGAN |
| `face_upsample` | bool | true | Upsample restored faces for high-resolution output |
| `upscale` | int | 2 | Final upsampling scale of the image |
| `output_format` | string | "png" | Output format: "png" or "jpg" |

## Output

Returns a list of restored images corresponding to each input image.

## Model Weights

Model weights are automatically downloaded on first run (~610MB total):
- CodeFormer face restoration model
- RetinaFace face detection model
- ParseNet face parsing model
- Real-ESRGAN background upsampler

## References

- [CodeFormer Paper](https://arxiv.org/abs/2206.11253) - "Towards Robust Blind Face Restoration with Codebook Lookup Transformer"
- [Official Repository](https://github.com/sczhou/CodeFormer)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Used for background enhancement

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
