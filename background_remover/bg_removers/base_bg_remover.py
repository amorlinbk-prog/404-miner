from abc import ABC, abstractmethod
from PIL import Image

import numpy as np
from background_remover.utils.rand_utils import set_random_seed


class BaseBGRemover(ABC):
    def __init__(self) -> None:
        self._image_threshold_check = 500

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def unload_model(self) -> None:
        pass

    @abstractmethod
    def remove_bg(self, image: Image) -> tuple[Image, bool]:
        pass

    @staticmethod
    def set_seed(seed: int):
        set_random_seed(seed)

    def is_image_valid(self, image: Image) -> bool:
        """ Function that checks if the image after background removal is empty or barely filled in with image data. """

        # Convert the image to grayscale
        img_gray = image.convert('L')

        # Convert the grayscale image to a NumPy array
        img_array = np.array(img_gray)

        # Calculate the variance of pixel values
        variance = np.var(img_array)

        if variance > self._image_threshold_check:
            return True
        else:
            return False

    def filter_semi_transparent_pixels(self, image: Image.Image) -> Image.Image:
        # Convert to numpy array
        img_np = np.array(image)

        # Extract alpha channel (4th channel)
        alpha = img_np[:, :, 3] / 255.0
        mask = alpha >= 0.72

        # Apply mask to all 4 channels
        filtered_np = img_np.copy()
        filtered_np[~mask] = [0, 0, 0, 0]  # Set fully transparent or black

        # Convert back to PIL image
        filtered_img = Image.fromarray(filtered_np, mode="RGBA")
        return filtered_img
