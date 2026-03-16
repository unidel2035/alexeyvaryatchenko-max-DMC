"""
Convert an image so that every pixel is replaced with the nearest DMC thread color.
No quantization, no smoothing — pure per-pixel DMC mapping.

Downloads the Mona Lisa from Wikipedia and produces a DMC-palette version.
"""

import numpy as np
from PIL import Image
import requests
import io
import os

from dmc_colors import DMC_COLORS


def build_dmc_palette():
    """Build numpy array of all DMC colors for vectorized distance calculation."""
    colors = []
    keys = []
    for dmc_num, (name, r, g, b) in DMC_COLORS.items():
        colors.append([r, g, b])
        keys.append((dmc_num, name))
    return np.array(colors, dtype=np.float64), keys


def convert_image_to_dmc(img: Image.Image) -> Image.Image:
    """Replace every pixel in the image with the nearest DMC color."""
    palette, keys = build_dmc_palette()

    arr = np.array(img.convert("RGB"), dtype=np.float64)
    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3)

    # Weighted Euclidean distance (perceptual weights for human vision)
    weights = np.array([0.299, 0.587, 0.114])

    # Calculate distance from each pixel to each DMC color
    # Using chunked processing to avoid memory issues
    chunk_size = 10000
    result = np.empty_like(pixels)

    for start in range(0, len(pixels), chunk_size):
        end = min(start + chunk_size, len(pixels))
        chunk = pixels[start:end]

        # Broadcast: (chunk_size, 1, 3) - (1, n_colors, 3)
        diff = chunk[:, np.newaxis, :] - palette[np.newaxis, :, :]
        dist = np.sum(diff ** 2 * weights, axis=2)

        nearest_idx = np.argmin(dist, axis=1)
        result[start:end] = palette[nearest_idx]

    result_img = Image.fromarray(result.reshape(h, w, 3).astype(np.uint8), "RGB")
    return result_img


def load_mona_lisa() -> Image.Image:
    """Load the Mona Lisa image from local file."""
    source_path = os.path.join(os.path.dirname(__file__), "mona_lisa_source.png")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source image not found: {source_path}")
    img = Image.open(source_path)
    print(f"Loaded: {img.size[0]}x{img.size[1]} pixels")
    return img


def main():
    img = load_mona_lisa()

    print("Converting every pixel to nearest DMC color...")
    dmc_img = convert_image_to_dmc(img)

    output_path = os.path.join(os.path.dirname(__file__), "mona_lisa_dmc.png")
    dmc_img.save(output_path, "PNG")
    print(f"Saved: {output_path} ({dmc_img.size[0]}x{dmc_img.size[1]} pixels)")

    # Count unique DMC colors used
    arr = np.array(dmc_img)
    unique_colors = len(np.unique(arr.reshape(-1, 3), axis=0))
    print(f"Unique DMC colors used: {unique_colors}")

    return output_path


if __name__ == "__main__":
    main()
