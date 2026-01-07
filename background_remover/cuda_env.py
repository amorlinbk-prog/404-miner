import os
from loguru import logger


def set_cuda_arch_env() -> None:
    """
    Detect available CUDA devices and set env vars that control torch/nvcc archs.

    - TORCH_CUDA_ARCH_LIST: space-separated list like "8.9 9.0"
    - CUDAARCHS: semicolon-separated list with "-real" suffix like "89-real;90-real"

    Safe to call multiple times. No-op if CUDA isn't available.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("CUDA not available; keeping existing arch env vars unchanged.")
            return

        device_caps: list[tuple[int, int]] = []
        for device_index in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(device_index)
            name = torch.cuda.get_device_name(device_index)
            device_caps.append((major, minor))
            logger.info(f"Detected CUDA device {device_index}: {name} (compute capability {major}.{minor})")

        if not device_caps:
            logger.info("No CUDA devices detected; keeping existing arch env vars unchanged.")
            return

        unique_caps = sorted({(maj, minr) for maj, minr in device_caps})
        torch_arch_list = " ".join([f"{maj}.{minr}" for maj, minr in unique_caps])
        cuda_archs_list = ";".join([f"{maj}{minr}-real" for maj, minr in unique_caps])

        prev_torch_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        prev_nvcc_archs = os.environ.get("CUDAARCHS")

        os.environ["TORCH_CUDA_ARCH_LIST"] = torch_arch_list
        os.environ["CUDAARCHS"] = cuda_archs_list

        logger.info(
            f"Setting TORCH_CUDA_ARCH_LIST={torch_arch_list} (was {prev_torch_arch if prev_torch_arch is not None else 'unset'})"
        )
        logger.info(
            f"Setting CUDAARCHS={cuda_archs_list} (was {prev_nvcc_archs if prev_nvcc_archs is not None else 'unset'})"
        )
    except Exception as e:
        logger.warning(f"Failed to configure CUDA arch env vars: {e}")


