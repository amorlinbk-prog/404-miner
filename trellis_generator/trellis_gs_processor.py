import gc
import os
import random
from io import BytesIO
from typing import Sequence

import ray
import torch
import torch.distributed as dist
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

from loguru import logger
from trellis_generator.pipelines import TrellisImageTo3DPipeline
from background_remover.ray_bg_remover import RayBGRemoverProcessor
from background_remover.bg_removers.ben2_bg_remover import Ben2BGRemover
from background_remover.bg_removers.birefnet_bg_remover import BiRefNetBGRemover
from background_remover.image_selector import ImageSelector
from background_remover.utils.rand_utils import secure_randint, set_random_seed


class GaussianProcessor:
    """Generates 3d models and videos"""

    def __init__(self, image_shape: tuple[int, int, int], vllm_flash_attn_backend: str = "FLASHINFER") -> None:
        logger.info(f"VLLM FLASH ATTENTION backend: {vllm_flash_attn_backend}")
        logger.info(f"TRELLIS ATTENTION backend: {os.environ['ATTN_BACKEND']}")

        self._bg_removers_workers: list[RayBGRemoverProcessor] = []
        self._vlm_image_selector = ImageSelector(3, image_shape, vllm_flash_attn_backend)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._image_to_3d_pipeline: TrellisImageTo3DPipeline | None = None
        self._qwen_pipe: QwenImageEditPlusPipeline | None = None
        self.gaussians: torch.Tensor | None = None

    def load_models(self, model_name: str = "microsoft/TRELLIS-image-large") -> None:
        """ Function for preloading all essential models for image -> 3D pipeline """

        self._image_to_3d_pipeline = TrellisImageTo3DPipeline.from_pretrained(model_name)
        self._image_to_3d_pipeline.to(self._device)

        self._bg_removers_workers: list[RayBGRemoverProcessor] = [
            RayBGRemoverProcessor.remote(Ben2BGRemover),
            RayBGRemoverProcessor.remote(BiRefNetBGRemover),
        ]

        # Load Qwen image edit pipeline for multi-view generation
        logger.info("Loading Qwen/Qwen-Image-Edit-2511 for view generation...")
        qwen_dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        # If there are multiple GPUs, let diffusers/accelerate shard the model across them
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info(
                f"Detected {torch.cuda.device_count()} CUDA devices; "
                "loading Qwen with device_map='auto' to spread it across GPUs."
            )
            self._qwen_pipe = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2511",
                torch_dtype=qwen_dtype,
                device_map="balanced",
            )
        else:
            # Single-GPU or CPU: keep the whole pipeline on self._device
            self._qwen_pipe = QwenImageEditPlusPipeline.from_pretrained(
                "Qwen/Qwen-Image-Edit-2511",
                torch_dtype=qwen_dtype,
                device_map="cuda",
            ).to(self._device)
        logger.info("Qwen image edit pipeline loaded.")

        torch.cuda.empty_cache()
        self._vlm_image_selector.load_model()

    def unload_models(self) -> None:
        """  Function for unloading all models for image -> 3D pipeline """

        for worker in self._bg_removers_workers:
            worker.unload_model.remote()
        dist.destroy_process_group()

        del self._image_to_3d_pipeline
        del self.gaussians

        self._image_to_3d_pipeline = None
        self.gaussians = None

        gc.collect()
        torch.cuda.empty_cache()

    def _generate_views(self, image: Image.Image, seed: int) -> list[Image.Image]:
        """
        Generate three edited views (left, right, back) of the object using Qwen.
        Prompts are aligned with the congenial-adventure pipeline.
        """
        if self._qwen_pipe is None:
            raise RuntimeError("Qwen image edit pipeline is not loaded. Call load_models() first.")

        base_seed = seed if seed >= 0 else secure_randint(0, 10_000)

        # Prepare base image once
        base_image = image.convert("RGB")
        base_image = base_image.resize((1024, 1024), Image.Resampling.LANCZOS)

        left_prompt = (
            "Show this object in left three-quarters view and make sure it is fully visible. "
            "Turn background neutral solid color contrasting with an object. "
            "Delete background details. Delete watermarks. Keep object colors. Sharpen image details"
        )
        right_prompt = (
            "Show this object in right three-quarters view and make sure it is fully visible. "
            "Turn background neutral solid color contrasting with an object. "
            "Delete background details. Delete watermarks. Keep object colors. Sharpen image details"
        )
        back_prompt = (
            "Show this object in back view and make sure it is fully visible. "
            "Turn background neutral solid color contrasting with an object. "
            "Delete background details. Delete watermarks. Keep object colors. Sharpen image details"
        )

        images = [base_image, base_image, base_image]
        prompts = [left_prompt, right_prompt, back_prompt]

        # Use different generators for each view to decorrelate randomness while staying reproducible
        generators = [
            torch.Generator(device=self._device).manual_seed(base_seed + i)
            for i in range(len(images))
        ]

        result = self._qwen_pipe(
            image=images,
            prompt=prompts,
            generator=generators,
            num_inference_steps=4,
        )

        left_view, right_view, back_view = result.images

        return [left_view, right_view, back_view]

    def warmup_generator(self):
        """ Function for warming up the generator. """

        dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
        self.get_model_from_image_as_ply_obj(image=dummy, seed=0)

    @staticmethod
    def _get_random_index_cycler(list_size: int):
        """
        Creates a generator that yields random indices without repetition.
        When all indices are exhausted, it reshuffles and continues.
        """

        while True:
            indices = list(range(list_size))
            random.shuffle(indices)
            for idx in indices:
                yield idx

    def _remove_background(self, image: Image.Image, seed: int) -> Image.Image:
        """ Function for removing background from the image. """

        futurs = [worker.run.remote(image) for worker in self._bg_removers_workers]
        results = ray.get(futurs)
        image1 = results[0]
        image2 = results[1]
        output_image = self._vlm_image_selector.select_with_image_selector(image1, image2, image, seed)
        return output_image

    def _generate_3d_object(self, images_no_bg: Sequence[Image.Image], seed: int) -> BytesIO:
        """ Function for generating a 3D object using one or more input images without background. """

        if seed < 0:
            set_seed = secure_randint(0, 10000)
            set_random_seed(set_seed)
        else:
            set_random_seed(seed)

        outputs = self._image_to_3d_pipeline.run_multi_image(
            list(images_no_bg),
        )
        self.gaussians = outputs["gaussian"][0]

        buffer = BytesIO()
        self.gaussians.save_ply(buffer)
        buffer.seek(0)

        return buffer

    def get_model_from_image_as_ply_obj(self, image: Image.Image, seed: int = -1) -> tuple[BytesIO, Image.Image]:
        """
        Generate 3D model using an image as input.

        Steps:
        1. Use Qwen/Qwen-Image-Edit-2511 to synthesize three edited views of the object
           (left, right, back).
        2. Run background removal on each edited view.
        3. Feed the three background-free views into the Trellis image-to-3D pipeline.
        """

        # 1. Generate three edited views with Qwen
        try:
            edited_views = self._generate_views(image, seed)
        except Exception as err:
            logger.error(f"Failed to generate edited views with Qwen; falling back to single-view pipeline. Error: {err}")
            edited_views = [image]

        # 2. Remove background for each view
        views_no_bg: list[Image.Image] = []
        for idx, view in enumerate(edited_views):
            view_seed = seed + idx if seed >= 0 else -1
            has_alpha = view.mode in ("LA", "RGBA", "PA")
            if not has_alpha:
                cleaned = self._remove_background(view, view_seed)
            else:
                cleaned = view
            views_no_bg.append(cleaned)

        # 3. Generate 3D object from all background-free views
        buffer = self._generate_3d_object(views_no_bg, seed)

        # Return buffer and the first cleaned view for preview/debug purposes
        return buffer, views_no_bg[0]
