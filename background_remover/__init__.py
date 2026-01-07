from background_remover.bg_removers.ben2_bg_remover import Ben2BGRemover
from background_remover.bg_removers.birefnet_bg_remover import BiRefNetBGRemover
from background_remover.ray_bg_remover import RayBGRemoverProcessor
from background_remover.image_selector import ImageSelector
from background_remover.config import (VLMSettings,
                                       load_vlm_image_selector_settings_from_yaml)
try:
    from background_remover.cuda_env import set_cuda_arch_env
    set_cuda_arch_env()
except Exception:
    # Avoid import-time crashes in library usage
    pass
