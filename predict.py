import faulthandler
import os
import tempfile
import time
import traceback
from pathlib import Path as FSPath
from typing import Dict, List

import cv2
import torch
from cog import BasePredictor, Input, Path as CogPath
from torch.hub import download_url_to_file
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.img_util import tensor2img_fast
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.utils.face_restoration_helper import FaceRestoreHelper

faulthandler.enable()

# Model weight URLs - downloaded during setup if not present
WEIGHT_URLS = {
    "weights/CodeFormer/codeformer.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "weights/facelib/detection_Resnet50_Final.pth": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "weights/facelib/parsing_parsenet.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
}


class Predictor(BasePredictor):
    def _log(self, message: str) -> None:
        print(message, flush=True)

    def _download_weights(self) -> None:
        """Download model weights if they don't exist and verify they are readable."""
        for weight_path, url in WEIGHT_URLS.items():
            if not os.path.exists(weight_path):
                print(f"Downloading {weight_path}...")
                os.makedirs(os.path.dirname(weight_path), exist_ok=True)
                download_url_to_file(url, weight_path, progress=True)

            if not os.path.isfile(weight_path):
                raise RuntimeError(f"Weight path is not a file: {weight_path}")
            if os.path.getsize(weight_path) == 0:
                raise RuntimeError(f"Weight file is empty: {weight_path}")

    def _load_codeformer_checkpoint(self, ckpt_path: str) -> Dict:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing CodeFormer checkpoint: {ckpt_path}")

        # Prefer secure loading mode, with compatibility fallback.
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=self.device)

        if "params_ema" not in checkpoint:
            raise RuntimeError(
                f"Invalid CodeFormer checkpoint format at {ckpt_path}: missing 'params_ema'"
            )
        return checkpoint["params_ema"]

    def setup(self) -> None:
        """Load model resources once at startup."""
        self._log("[setup] starting setup")
        self._download_weights()

        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            raise RuntimeError("CUDA is required for this model deployment")

        self.device = "cuda:0"
        torch.backends.cudnn.benchmark = True
        self._log(f"[setup] cuda_available={self.cuda_available}, device={self.device}")

        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)

        ckpt_path = "weights/CodeFormer/codeformer.pth"
        checkpoint = self._load_codeformer_checkpoint(ckpt_path)
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        self._log("[setup] loaded CodeFormer checkpoint")

        self.net_compiled = None
        self.optimized_runtime_enabled = False

        self.face_helper = None
        self.face_helper_config = None
        self.runtime_profile = "baseline"
        self.stability_mode = "safe"
        self.last_stats = {}
        self.compile_backend = "none"
        self._log("[setup] setup complete")

    def _ensure_optimized_runtime(self, compile_backend: str = "inductor") -> None:
        if self.optimized_runtime_enabled:
            self._log("[opt] optimized runtime already enabled")
            return

        self.compile_backend = compile_backend
        self._log(f"[opt] enabling optimized runtime, compile_backend={compile_backend}")

        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception as error:
            self._log(f"Warning: failed enabling CUDA SDP optimizations: {error}")

        # channels_last can improve throughput on H100, but only enable in optimized mode.
        self.net = self.net.to(memory_format=torch.channels_last)

        if compile_backend != "none":
            try:
                compile_kwargs = {"backend": compile_backend}
                if compile_backend == "inductor":
                    compile_kwargs["mode"] = "max-autotune"
                self.net_compiled = torch.compile(self.net, **compile_kwargs)
                self._log(f"[opt] enabled torch.compile backend={compile_backend}")
            except Exception as error:
                # Keep model functional even if compile fails.
                self._log(
                    f"Warning: torch.compile failed for backend={compile_backend}, continuing uncompiled: {error}"
                )
                self.net_compiled = None
        else:
            self._log("[opt] compile_backend=none, skipping torch.compile")
            self.net_compiled = None

        self.optimized_runtime_enabled = True

    def _get_net(self):
        if self.runtime_profile == "baseline":
            return self.net
        if self.stability_mode == "optimized" and self.net_compiled is not None:
            return self.net_compiled
        return self.net

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.stability_mode == "optimized" and self.optimized_runtime_enabled:
            return tensor.to(self.device, memory_format=torch.channels_last)
        return tensor.to(self.device)

    def _build_or_get_face_helper(self, use_parse: bool, use_optimized: bool) -> FaceRestoreHelper:
        config = (1, "jpg", use_parse, use_optimized)
        if self.face_helper is None or self.face_helper_config != config:
            det_half = use_optimized
            self.face_helper = FaceRestoreHelper(
                1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model="retinaface_resnet50",
                save_ext="jpg",
                use_parse=use_parse,
                det_half=det_half,
                compile_models=False,
                device=self.device,
            )
            self.face_helper_config = config
            self._log(
                f"[face_helper] created use_parse={use_parse}, det_half={det_half}, compile_models=False"
            )
        else:
            self.face_helper.set_upscale_factor(1)
            self.face_helper.save_ext = "jpg"
            self.face_helper.use_parse = use_parse
            self.face_helper.clean_all()
            self._log("[face_helper] reused existing helper")
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

        batched_face_t = self._to_device(torch.stack(face_tensors, dim=0))

        try:
            net = self._get_net()
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=self.amp_dtype,
                    enabled=use_amp and self.cuda_available,
                ):
                    outputs = net(batched_face_t, w=codeformer_fidelity, adain=True)[0]
            return [
                tensor2img_fast(outputs[i : i + 1], rgb2bgr=True, min_max=(-1, 1))
                for i in range(outputs.shape[0])
            ]
        except Exception as error:
            self._log(f"Failed batched inference for CodeFormer: {error}")
            return [
                tensor2img(face_t.unsqueeze(0), rgb2bgr=True, min_max=(-1, 1))
                for face_t in face_tensors
            ]

    def _process_single_image(
        self,
        image: FSPath,
        codeformer_fidelity: float,
        output_path: FSPath,
        runtime_profile: str,
    ) -> None:
        has_aligned = False
        only_center_face = False
        draw_box = False

        use_optimized = self.stability_mode == "optimized"
        use_parse = runtime_profile != "max_speed"
        use_amp = use_optimized and runtime_profile in {"fast", "max_speed"}
        use_batch_faces = use_optimized and runtime_profile in {"fast", "max_speed"}

        self.face_helper = self._build_or_get_face_helper(
            use_parse=use_parse,
            use_optimized=use_optimized,
        )
        self.face_helper.clean_all()

        stage_times_ms: Dict[str, float] = {}

        self._log("[stage:start] read_image")
        t0 = time.perf_counter()
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        stage_times_ms["read_ms"] = (time.perf_counter() - t0) * 1000
        if img is None:
            raise ValueError(f"Failed to decode input image: {image}")
        self._log(f"[stage:done] read_image ms={stage_times_ms['read_ms']:.2f}")

        if has_aligned:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            self.face_helper.cropped_faces = [img]
        else:
            self._log("[stage:start] face_landmarks")
            t0 = time.perf_counter()
            self.face_helper.read_image(img)
            num_det_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            stage_times_ms["detect_ms"] = (time.perf_counter() - t0) * 1000
            self._log(
                f"[stage:done] face_landmarks ms={stage_times_ms['detect_ms']:.2f}, detect={num_det_faces}"
            )

            self._log("[stage:start] align_warp_face")
            t0 = time.perf_counter()
            self.face_helper.align_warp_face()
            stage_times_ms["align_ms"] = (time.perf_counter() - t0) * 1000
            self._log(f"[stage:done] align_warp_face ms={stage_times_ms['align_ms']:.2f}")

        self._log("[stage:start] codeformer_restore")
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
                cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = self._to_device(cropped_face_t.unsqueeze(0))

                try:
                    net = self._get_net()
                    with torch.inference_mode():
                        with torch.autocast(
                            device_type="cuda",
                            dtype=self.amp_dtype,
                            enabled=use_amp and self.cuda_available,
                        ):
                            output = net(cropped_face_t, w=codeformer_fidelity, adain=True)[0]
                        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    del output
                except Exception as error:
                    self._log(f"Failed inference for CodeFormer: {error}")
                    restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                self.face_helper.add_restored_face(restored_face.astype("uint8"))

        stage_times_ms["codeformer_ms"] = (time.perf_counter() - t0) * 1000
        self._log(f"[stage:done] codeformer_restore ms={stage_times_ms['codeformer_ms']:.2f}")

        if not has_aligned:
            self._log("[stage:start] inverse_affine_and_paste")
            t0 = time.perf_counter()
            self.face_helper.get_inverse_affine(None)
            restored_img = self.face_helper.paste_faces_to_input_image(
                upsample_img=None,
                draw_box=draw_box,
            )
            stage_times_ms["paste_ms"] = (time.perf_counter() - t0) * 1000
            stage_times_ms["parse_ms"] = getattr(self.face_helper, "parse_time_ms", 0.0)
            self._log(
                f"[stage:done] inverse_affine_and_paste ms={stage_times_ms['paste_ms']:.2f}, parse_ms={stage_times_ms['parse_ms']:.2f}"
            )

        self._log("[stage:start] write_output")
        t0 = time.perf_counter()
        ok = cv2.imwrite(str(output_path), restored_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError(f"Failed to write JPG output: {output_path}")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Output file missing or empty: {output_path}")

        stage_times_ms["write_ms"] = (time.perf_counter() - t0) * 1000
        self._log(f"[stage:done] write_output ms={stage_times_ms['write_ms']:.2f}, path={output_path}")
        stage_times_ms["total_ms"] = sum(stage_times_ms.values())
        stage_times_ms["num_faces"] = float(len(self.face_helper.cropped_faces))
        self.last_stats = stage_times_ms

    def predict(
        self,
        images: List[CogPath] = Input(description="Input image(s) - up to 4 images"),
        codeformer_fidelity: float = Input(
            default=0.5,
            ge=0,
            le=1,
            description="Balance the quality (lower number) and fidelity (higher number).",
        ),
        runtime_profile: str = Input(
            description="Runtime profile",
            default="baseline",
            choices=["baseline", "fast", "max_speed"],
        ),
        stability_mode: str = Input(
            description="Execution mode: safe for maximum reliability on Replicate, optimized for maximum speed",
            default="safe",
            choices=["safe", "optimized"],
        ),
        compile_backend: str = Input(
            description="torch.compile backend used only when stability_mode=optimized",
            default="eager",
            choices=["none", "eager", "aot_eager", "inductor"],
        ),
    ) -> List[CogPath]:
        """Run a single prediction on the model."""
        if not images:
            raise ValueError("At least one image is required")
        if len(images) > 4:
            raise ValueError("Maximum of 4 images allowed")

        self.runtime_profile = runtime_profile
        self.stability_mode = stability_mode
        self.compile_backend = compile_backend
        self._log(
            f"[predict] runtime_profile={runtime_profile}, stability_mode={stability_mode}, compile_backend={compile_backend}, images={len(images)}"
        )
        if self.stability_mode == "optimized":
            self._ensure_optimized_runtime(compile_backend=compile_backend)

        input_paths: List[FSPath] = []
        for image in images:
            img_path = FSPath(str(image))
            if not img_path.exists():
                raise ValueError(f"Input image does not exist: {img_path}")
            if not img_path.is_file():
                raise ValueError(f"Input path is not a file: {img_path}")
            input_paths.append(img_path)

        out_dir = FSPath(tempfile.mkdtemp(prefix="codeformer-"))
        output_paths: List[CogPath] = []
        all_stats = []

        for idx, img_path in enumerate(input_paths, start=1):
            out_path = out_dir / f"output_{idx}.jpg"
            self._log(f"Processing image {idx}/{len(input_paths)}: {img_path.name}")
            try:
                self._process_single_image(
                    img_path,
                    codeformer_fidelity,
                    out_path,
                    runtime_profile,
                )
            except Exception as error:
                tb = traceback.format_exc()
                raise RuntimeError(
                    f"Image processing failed for index={idx}, path={img_path}: {error}\n{tb}"
                ) from error

            if not out_path.exists() or out_path.stat().st_size == 0:
                raise RuntimeError(f"Output validation failed for image index={idx}: {out_path}")

            output_paths.append(CogPath(str(out_path)))
            all_stats.append(self.last_stats)

        if all_stats:
            avg_total = sum(s["total_ms"] for s in all_stats) / len(all_stats)
            self._log(
                f"Runtime profile={runtime_profile}, stability_mode={stability_mode}, "
                f"avg_total_ms={avg_total:.2f}, images={len(all_stats)}"
            )

        if not output_paths:
            raise RuntimeError("No output images generated")

        return output_paths
