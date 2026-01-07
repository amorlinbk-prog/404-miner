import gc
import argparse
import asyncio
import os
from io import BytesIO
from time import time
from PIL import Image
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
import uvicorn
from loguru import logger
from fastapi import FastAPI,  UploadFile, File, APIRouter, Form
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import State

from trellis_generator.trellis_gs_processor import GaussianProcessor


# Setting up default attention backend for trellis generator: can be 'flash-attn' or 'xformers'
os.environ['ATTN_BACKEND'] = 'flash-attn'


def get_args() -> argparse.Namespace:
    """ Function for getting arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10006)
    return parser.parse_args()


executor = ThreadPoolExecutor(max_workers=1)


class MyFastAPI(FastAPI):
    state: State
    router: APIRouter
    version: str


@asynccontextmanager
async def lifespan(app: MyFastAPI) -> AsyncIterator[None]:
    """ Function that loading all models and warming up the generation."""
    major, minor = torch.cuda.get_device_capability(0)

    if major == 9:
        vllm_flash_attn_backend = "FLASH_ATTN"
    else:
        vllm_flash_attn_backend = "FLASHINFER"

    try:
        logger.info("Loading Trellis generator model...")
        app.state.trellis_generator = GaussianProcessor((int(1024 / 2), int(1024 / 2), 3), vllm_flash_attn_backend)
        app.state.trellis_generator.load_models()
        clean_vram()
        logger.info("Model loading is complete.")
    except Exception as e:
        logger.exception(f"Exception during model loading: {e}")
        raise SystemExit("Model failed to load → exiting server")

    try:
        logger.info("Warming up Trellis generator...")
        app.state.trellis_generator.warmup_generator()
        clean_vram()
        logger.info("Warm-up is complete. Server is ready.")

    except Exception as e:
        logger.exception(f"Exception during warming up the generator: {e}")
        raise SystemExit("Warm-up failed → exiting server")

    yield


app = MyFastAPI(title="404 Base Miner Service", version="0.0.0")
app.router.lifespan_context = lifespan


def clean_vram() -> None:
    """ Function for cleaning VRAM. """
    gc.collect()
    torch.cuda.empty_cache()


def generation_block(prompt_image: Image.Image, seed: int = -1) -> BytesIO:
    """ Function for 3D data generation using provided image"""

    t_start = time()
    buffer, _ = app.state.trellis_generator.get_model_from_image_as_ply_obj(image=prompt_image, seed=seed)

    t_get_model = time()
    logger.debug(f"Model Generation took: {(t_get_model - t_start)} secs.")

    clean_vram()

    t_gc= time()
    logger.debug(f"Garbage Collection took: {(t_gc - t_get_model)} secs")

    return buffer


@app.post("/generate")
async def generate_model(prompt_image_file: UploadFile = File(...), seed: int = Form(-1)) -> Response:
    """ Generates a 3D model as a PLY buffer """

    logger.info("Task received. Prompt-Image")

    contents = await prompt_image_file.read()
    prompt_image = Image.open(BytesIO(contents))

    loop = asyncio.get_running_loop()
    buffer = await loop.run_in_executor(executor, generation_block, prompt_image, seed)
    logger.info(f"Task completed.")

    return StreamingResponse(buffer, media_type="application/octet-stream")


@app.get("/version", response_model=str)
async def version() -> str:
    """ Returns current endpoint version."""
    return app.version


@app.get("/health")
def health_check() -> dict[str, str]:
    """ Return if the server is alive """
    return {"status": "healthy"}


if __name__ == "__main__":
    args: argparse.Namespace  = get_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
