from PIL import Image

import ray
import torch
from background_remover.bg_removers.base_bg_remover import BaseBGRemover


@ray.remote(num_gpus=0.05)
class RayBGRemoverProcessor:
    def __init__(self,  model: BaseBGRemover):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._bg_remover = model(device)
        self._bg_remover.load_model()

    def unload_model(self):
       self._bg_remover.unload_model()
       del self._bg_remover
       self._bg_remover = None
       ray.shutdown()

    def run(self, image: Image.Image) -> Image:
        result_image = self._bg_remover.remove_bg(image)
        return result_image
