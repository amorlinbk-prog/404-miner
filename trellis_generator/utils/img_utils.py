from PIL import Image
import numpy as np


def recenter_image(image: Image.Image, square_size: int, padding: float, min_size_ratio: float = 0.6) -> Image.Image:
    """ Recenter object in the image and resize it to a specified square size. """

    image_np = np.asarray(image)

    if image_np.shape[-1] < 4:
        raise ValueError("Image must have alpha channel (RGBA)")

    mask = image_np[..., -1] > 0
    if not np.any(mask):
        empty_image = np.zeros((square_size, square_size, 4), dtype=np.uint8)
        return Image.fromarray(empty_image)

    # Get bounding box
    coords = np.where(mask)
    y_min, y_max = coords[0].min(), coords[0].max() + 1
    x_min, x_max = coords[1].min(), coords[1].max() + 1

    # Extract the object
    cropped = image_np[y_min:y_max, x_min:x_max]
    h, w = cropped.shape[:2]

    # Calculate size constraints
    max_size = int(square_size * (1 - padding))
    min_size = int(square_size * min_size_ratio)

    # Determine target size
    current_max_dim = max(h, w)

    if current_max_dim > max_size:
        scale = max_size / current_max_dim
    elif current_max_dim < min_size:
        scale = min_size / current_max_dim
    else:
        scale = 1.0

    if scale != 1.0:
        new_h = int(h * scale)
        new_w = int(w * scale)
        pil_img = Image.fromarray(cropped)
        pil_resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized = np.asarray(pil_resized)
    else:
        resized = cropped
        new_h, new_w = h, w

    # Create square canvas and center the object
    proc_image = np.zeros((square_size, square_size, 4), dtype=np.uint8)

    start_y = (square_size - new_h) // 2
    start_x = (square_size - new_w) // 2
    end_y = start_y + new_h
    end_x = start_x + new_w

    proc_image[start_y:end_y, start_x:end_x] = resized

    return Image.fromarray(proc_image)


def resize_image(image: Image.Image, res_x: int, res_y: int) -> Image.Image:
    """ Function for resizing the input image. """

    resized_image = image.resize((res_x, res_y), Image.Resampling.LANCZOS)
    return resized_image


def add_white_background(image: Image.Image) -> Image.Image:
    """ Function for adding white BG to the image with transparent BG. """
    # Create white background with same size
    white_bg = Image.new('RGBA', image.size, (255, 255, 255, 255))

    # Composite the image onto white background
    result = Image.alpha_composite(white_bg, image.convert('RGBA'))

    # Convert to RGB to remove alpha channel
    return result.convert('RGB')
