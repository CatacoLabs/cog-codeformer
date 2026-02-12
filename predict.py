import os
import tempfile
import time
import cv2
import torch
from typing import Dict, List
from torch.hub import download_url_to_file
from torchvision.transforms.functional import normalize
from cog import BasePredictor, Input, Path

from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.img_util import tensor2img_fast
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.utils.face_restoration_helper import FaceRestoreHelper

# Model weight URLs - downloaded during setup if not present
WEIGHT_URLS = {
    "weights/CodeFormer/codeformer.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "weights/realesrgan/RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "weights/facelib/detection_Resnet50_Final.pth": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "weights/facelib/parsing_parsenet.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
}


class Predictor(BasePredictor):
    def _download_weights(self):
        """Download model weights if they don't exist."""
        for weight_path, url in WEIGHT_URLS.items():
            if not os.path.exists(weight_path):
                print(f"Downloading {weight_path}...")
                os.makedirs(os.path.dirname(weight_path), exist_ok=True)
                download_url_to_file(url, weight_path, progress=True)

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self._download_weights()
        self.device = "cuda:0"
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        self.amp_dtype = (
            torch.bfloat16
            if self.cuda_available and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.upsampler_tiled = set_realesrgan(tile=400)
        self.upsampler_notile = set_realesrgan(tile=0, share_model=self.upsampler_tiled.model if self.upsampler_tiled else None)
        self.net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)
        ckpt_path = "weights/CodeFormer/codeformer.pth"
        checkpoint = torch.load(ckpt_path, weights_only=False)[
            "params_ema"
        ]  # update file permission if cannot load
        self.net.load_state_dict(checkpoint)
        self.net.eval()

        # Apply channels_last memory format for better GPU utilization
        if self.cuda_available:
            self.net = self.net.to(memory_format=torch.channels_last)
            self.upsampler_tiled.model = self.upsampler_tiled.model.to(
                memory_format=torch.channels_last
            )
            self.upsampler_notile.model = self.upsampler_notile.model.to(
                memory_format=torch.channels_last
            )

        # torch.compile models for fast/max_speed profiles
        if self.cuda_available:
            self.net_compiled = torch.compile(self.net, mode="max-autotune")
            self.upsampler_notile.model = torch.compile(
                self.upsampler_notile.model, mode="max-autotune"
            )
            self.upsampler_tiled.model = torch.compile(
                self.upsampler_tiled.model, mode="reduce-overhead"
            )
        else:
            self.net_compiled = self.net

        self.face_helper = None
        self.face_helper_config = None
        self.runtime_profile = "baseline"
        self.last_stats = {}

    def _get_net(self):
        """Return compiled net for fast/max_speed, uncompiled for baseline."""
        if self.runtime_profile == "baseline":
            return self.net
        return self.net_compiled

    def _maybe_sync_cuda(self):
        if self.cuda_available:
            torch.cuda.synchronize()

    def _build_or_get_face_helper(
        self, upscale: int, output_format: str, use_parse: bool
    ) -> FaceRestoreHelper:
        config = (upscale, output_format, use_parse)
        if self.face_helper is None or self.face_helper_config != config:
            self.face_helper = FaceRestoreHelper(
                upscale,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext=output_format,
                use_parse=use_parse,
                device=self.device,
            )
            self.face_helper_config = config
        else:
            self.face_helper.set_upscale_factor(upscale)
            self.face_helper.save_ext = output_format
            self.face_helper.use_parse = use_parse
            self.face_helper.clean_all()
        return self.face_helper

    def _restore_faces_batched(
        self, cropped_faces: List, codeformer_fidelity: float, use_amp: bool
    ) -> List:
        if not cropped_faces:
            return []

        face_tensors = []
        for cropped_face in cropped_faces:
            face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            face_tensors.append(face_t)

        batched_face_t = torch.stack(face_tensors, dim=0).to(
            self.device, memory_format=torch.channels_last
        )

        try:
            net = self._get_net()
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=self.amp_dtype,
                    enabled=use_amp and self.cuda_available,
                ):
                    outputs = net(
                        batched_face_t, w=codeformer_fidelity, adain=True
                    )[0]
            restored_faces = [
                tensor2img_fast(outputs[i : i + 1], rgb2bgr=True, min_max=(-1, 1))
                for i in range(outputs.shape[0])
            ]
            return restored_faces
        except Exception as error:
            print(f"\tFailed batched inference for CodeFormer: {error}")
            return [
                tensor2img(face_t.unsqueeze(0), rgb2bgr=True, min_max=(-1, 1))
                for face_t in face_tensors
            ]

    def _process_single_image(
        self,
        image: Path,
        codeformer_fidelity: float,
        background_enhance: bool,
        face_upsample: bool,
        upscale: int,
        output_format: str,
        output_path: Path,
        runtime_profile: str,
    ) -> bool:
        """Process a single image. Returns True on success, False on failure."""
        # take the default setting for the demo
        has_aligned = False
        only_center_face = False
        draw_box = False
        use_parse = runtime_profile != "max_speed"
        use_amp = runtime_profile in {"fast", "max_speed"}
        use_batch_faces = runtime_profile in {"fast", "max_speed"}
        self.face_helper = self._build_or_get_face_helper(
            upscale=upscale,
            output_format=output_format,
            use_parse=use_parse,
        )
        self.face_helper.clean_all()

        upsampler = self.upsampler_tiled if runtime_profile == "baseline" else self.upsampler_notile
        bg_upsampler = upsampler if background_enhance else None
        face_upsampler = upsampler if face_upsample else None
        stage_times_ms: Dict[str, float] = {}

        t0 = time.perf_counter()
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        stage_times_ms["read_ms"] = (time.perf_counter() - t0) * 1000

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            self.face_helper.cropped_faces = [img]
        else:
            t0 = time.perf_counter()
            self.face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            self._maybe_sync_cuda()
            stage_times_ms["detect_ms"] = (time.perf_counter() - t0) * 1000
            print(f"\tdetect {num_det_faces} faces")
            # align and warp each face
            t0 = time.perf_counter()
            self.face_helper.align_warp_face()
            stage_times_ms["align_ms"] = (time.perf_counter() - t0) * 1000

        # For fast/max_speed: overlap bg upsampling (CUDA stream) with face restoration
        use_streams = (
            runtime_profile in {"fast", "max_speed"}
            and self.cuda_available
            and bg_upsampler is not None
            and not has_aligned
        )

        bg_img = None
        bg_stream = None

        if use_streams:
            # Launch background upsampling on a separate CUDA stream
            bg_stream = torch.cuda.Stream()
            t0_bg = time.perf_counter()
            with torch.cuda.stream(bg_stream):
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]

        # face restoration for each cropped face (runs on default stream)
        t0 = time.perf_counter()
        if use_batch_faces:
            restored_faces = self._restore_faces_batched(
                self.face_helper.cropped_faces,
                codeformer_fidelity=codeformer_fidelity,
                use_amp=use_amp,
            )
            for restored_face in restored_faces:
                self.face_helper.add_restored_face(restored_face.astype("uint8"))
        else:
            for cropped_face in self.face_helper.cropped_faces:
                cropped_face_t = img2tensor(
                    cropped_face / 255.0, bgr2rgb=True, float32=True
                )
                normalize(
                    cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
                )
                cropped_face_t = cropped_face_t.unsqueeze(0).to(
                    self.device, memory_format=torch.channels_last
                )

                try:
                    net = self._get_net()
                    with torch.inference_mode():
                        output = net(
                            cropped_face_t, w=codeformer_fidelity, adain=True
                        )[0]
                        restored_face = tensor2img(
                            output, rgb2bgr=True, min_max=(-1, 1)
                        )
                    del output
                except Exception as error:
                    print(f"\tFailed inference for CodeFormer: {error}")
                    restored_face = tensor2img(
                        cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                    )

                self.face_helper.add_restored_face(restored_face.astype("uint8"))
        self._maybe_sync_cuda()
        stage_times_ms["codeformer_ms"] = (time.perf_counter() - t0) * 1000

        # paste_back
        if not has_aligned:
            # Wait for bg upsampling stream or run it sequentially
            if use_streams:
                bg_stream.synchronize()
                stage_times_ms["bg_upsample_ms"] = (time.perf_counter() - t0_bg) * 1000
            elif bg_upsampler is not None:
                t0 = time.perf_counter()
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
                self._maybe_sync_cuda()
                stage_times_ms["bg_upsample_ms"] = (time.perf_counter() - t0) * 1000
            else:
                stage_times_ms["bg_upsample_ms"] = 0.0
            t0 = time.perf_counter()
            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None:
                restored_img = self.face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            else:
                restored_img = self.face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, draw_box=draw_box
                )
            self._maybe_sync_cuda()
            stage_times_ms["paste_ms"] = (time.perf_counter() - t0) * 1000
            stage_times_ms["parse_ms"] = getattr(
                self.face_helper, "parse_time_ms", 0.0
            )

        # save restored img with appropriate format
        t0 = time.perf_counter()
        if output_format == "jpg":
            cv2.imwrite(str(output_path), restored_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            imwrite(restored_img, str(output_path))
        stage_times_ms["write_ms"] = (time.perf_counter() - t0) * 1000
        stage_times_ms["total_ms"] = sum(stage_times_ms.values())
        stage_times_ms["num_faces"] = float(len(self.face_helper.cropped_faces))
        self.last_stats = stage_times_ms

        return True

    def predict(
        self,
        images: List[Path] = Input(description="Input image(s) - up to 4 images"),
        codeformer_fidelity: float = Input(
            default=0.5,
            ge=0,
            le=1,
            description="Balance the quality (lower number) and fidelity (higher number).",
        ),
        background_enhance: bool = Input(
            description="Enhance background image with Real-ESRGAN", default=True
        ),
        face_upsample: bool = Input(
            description="Upsample restored faces for high-resolution AI-created images",
            default=True,
        ),
        upscale: int = Input(
            description="The final upsampling scale of the image",
            default=2,
        ),
        output_format: str = Input(
            description="Output image format",
            default="png",
            choices=["png", "jpg"],
        ),
        runtime_profile: str = Input(
            description="Runtime profile",
            default="baseline",
            choices=["baseline", "fast", "max_speed"],
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        # Validate number of images
        if len(images) > 4:
            raise ValueError("Maximum of 4 images allowed")
        self.runtime_profile = runtime_profile

        # Create output directory
        out_dir = Path(tempfile.mkdtemp())
        output_paths = []
        all_stats = []

        # Process each image
        for idx, img_path in enumerate(images, start=1):
            out_path = out_dir / f"output_{idx}.{output_format}"
            try:
                print(f"Processing image {idx}...")
                success = self._process_single_image(
                    img_path,
                    codeformer_fidelity,
                    background_enhance,
                    face_upsample,
                    upscale,
                    output_format,
                    out_path,
                    runtime_profile,
                )
                if success:
                    output_paths.append(out_path)
                    all_stats.append(self.last_stats)
            except Exception as e:
                print(f"Failed to process image {idx}: {e}")
                # Continue to next image
        if all_stats:
            avg_total = sum(s["total_ms"] for s in all_stats) / len(all_stats)
            print(
                f"Runtime profile={runtime_profile}, avg_total_ms={avg_total:.2f}, "
                f"images={len(all_stats)}"
            )

        return output_paths


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def set_realesrgan(tile=400, share_model=None):
    if not torch.cuda.is_available():  # CPU
        import warnings

        warnings.warn(
            "The unoptimized RealESRGAN is slow on CPU. We do not use it. "
            "If you really want to use it, please modify the corresponding codes.",
            category=RuntimeWarning,
        )
        upsampler = None
    else:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        upsampler = RealESRGANer(
            scale=2,
            model_path="./weights/realesrgan/RealESRGAN_x2plus.pth",
            model=model,
            tile=tile,
            tile_pad=40 if tile > 0 else 0,
            pre_pad=0,
            half=True,
        )
        # Share model weights if provided (avoids double load + double VRAM)
        if share_model is not None:
            upsampler.model = share_model
    return upsampler
