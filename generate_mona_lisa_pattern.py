"""
Generate a DMC embroidery pattern from the Mona Lisa image.

Issue requirements:
- 18 squares per 1 inch of image
- Maximum physical width: 30cm
- Colors mapped to DMC palette with smooth transitions
- Output as an image with a URL
"""

import sys
import os
import io
import math

# Add the issue-1 tooling to path
sys.path.insert(0, "/tmp/gh-issue-solver-1773674563517")

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from dmc_colors import find_closest_dmc, DMC_COLORS
from image_processor import process_image, EmbroideryPattern


def generate_pattern_image(
    pattern: EmbroideryPattern,
    cell_size: int = 4,
    output_path: str = "mona_lisa_dmc_pattern.png",
) -> str:
    """
    Render the DMC embroidery pattern as a PNG image.

    Args:
        pattern: The EmbroideryPattern to render
        cell_size: Pixels per stitch cell (default 4 - small but visible)
        output_path: Path to save the output PNG

    Returns:
        Path to the saved image
    """
    width_px = pattern.width * cell_size
    height_px = pattern.height * cell_size

    img = Image.new("RGB", (width_px, height_px), (255, 255, 255))
    pixels = img.load()

    # Fill each cell with its DMC color
    for row_idx, row in enumerate(pattern.grid):
        for col_idx, cell in enumerate(row):
            dmc_num, name, r, g, b = cell
            x_start = col_idx * cell_size
            y_start = row_idx * cell_size
            for dy in range(cell_size):
                for dx in range(cell_size):
                    pixels[x_start + dx, y_start + dy] = (r, g, b)

    img.save(output_path, "PNG", optimize=True)
    print(f"Pattern image saved: {output_path} ({width_px}x{height_px} px)")
    return output_path


def smooth_dmc_colors(img_rgb: Image.Image, passes: int = 1) -> Image.Image:
    """
    Apply a gentle median-like smoothing to reduce sharp color transitions.
    Uses a 3x3 window to blend neighboring pixel colors within DMC palette.
    """
    arr = np.array(img_rgb, dtype=np.float32)

    for _ in range(passes):
        smoothed = arr.copy()
        h, w, c = arr.shape
        # Simple box blur (3x3) to smooth color transitions
        for row in range(1, h - 1):
            for col in range(1, w - 1):
                neighborhood = arr[row-1:row+2, col-1:col+2, :].reshape(-1, 3)
                smoothed[row, col, :] = neighborhood.mean(axis=0)
        arr = smoothed

    return Image.fromarray(arr.astype(np.uint8), "RGB")


def process_mona_lisa(
    image_path: str,
    stitches_per_inch: int = 18,
    max_width_cm: float = 30.0,
    max_colors: int = 50,
) -> tuple[EmbroideryPattern, str]:
    """
    Process the Mona Lisa image into a DMC embroidery pattern.

    Args:
        image_path: Path to the source image
        stitches_per_inch: Number of stitches per inch (18 for Aida 18-count)
        max_width_cm: Maximum physical width in cm
        max_colors: Maximum number of DMC colors

    Returns:
        Tuple of (EmbroideryPattern, output_image_path)
    """
    max_width_inch = max_width_cm / 2.54
    pattern_width = int(stitches_per_inch * max_width_inch)

    print(f"Settings: {stitches_per_inch} stitches/inch, max {max_width_cm}cm wide")
    print(f"Pattern width: {pattern_width} stitches")

    # Load the image
    img = Image.open(image_path).convert("RGB")
    print(f"Source image: {img.size[0]}x{img.size[1]} pixels")

    # Calculate pattern height to preserve aspect ratio
    aspect = img.height / img.width
    pattern_height = int(pattern_width * aspect)
    print(f"Pattern height: {pattern_height} stitches")
    print(
        f"Physical size: {pattern_width/stitches_per_inch*2.54:.1f}cm x "
        f"{pattern_height/stitches_per_inch*2.54:.1f}cm"
    )

    # Resize image to pattern dimensions using LANCZOS for best quality
    img_small = img.resize((pattern_width, pattern_height), Image.LANCZOS)

    # Apply gentle smoothing to reduce sharp color transitions
    print("Applying color smoothing...")
    img_smoothed = smooth_dmc_colors(img_small, passes=2)

    # Quantize colors to reduce the palette before mapping to DMC
    print(f"Quantizing to {max_colors} colors...")
    img_quantized = img_smoothed.quantize(
        colors=max_colors, method=Image.Quantize.MEDIANCUT
    )
    img_rgb = img_quantized.convert("RGB")

    # Map each pixel to the nearest DMC color
    print("Mapping pixels to DMC colors...")
    pixels_arr = np.array(img_rgb).reshape(-1, 3)
    dmc_pixels = [find_closest_dmc(int(r), int(g), int(b)) for r, g, b in pixels_arr]

    # Build grid
    grid = []
    for row_idx in range(pattern_height):
        row = []
        for col_idx in range(pattern_width):
            row.append(dmc_pixels[row_idx * pattern_width + col_idx])
        grid.append(row)

    # Build color legend
    color_counts: dict = {}
    for cell in dmc_pixels:
        dmc_num, name, r, g, b = cell
        if dmc_num not in color_counts:
            color_counts[dmc_num] = [name, r, g, b, 0]
        color_counts[dmc_num][4] += 1

    _SYMBOLS = (
        "■ □ ▲ ▼ ◆ ○ ● ★ ✦ ✚ ✿ ♦ ♠ ♣ ♥ ▪ △ ▽ ◇ ◈ ◉ ◎ ⊕ ⊗ ⊙ ⊛ ⊞ ⊟ ⊠ ⊡ "
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z "
        "a b c d e f g h i j k l m n o p q r s t u v w x y z "
        "0 1 2 3 4 5 6 7 8 9"
    ).split()

    color_legend = {}
    for i, (dmc_num, info) in enumerate(
        sorted(color_counts.items(), key=lambda x: -x[1][4])
    ):
        symbol = _SYMBOLS[i % len(_SYMBOLS)]
        name, r, g, b, count = info
        color_legend[dmc_num] = (name, r, g, b, symbol, count)

    pattern = EmbroideryPattern(
        width=pattern_width,
        height=pattern_height,
        grid=grid,
        color_legend=color_legend,
    )

    print(f"Pattern created: {pattern.width}x{pattern.height}, {len(color_legend)} colors")

    # Generate the output image (cell_size=4 gives readable result)
    output_path = "/tmp/gh-issue-solver-1773675512333/mona_lisa_dmc_pattern.png"
    generate_pattern_image(pattern, cell_size=4, output_path=output_path)

    return pattern, output_path


if __name__ == "__main__":
    pattern, output_path = process_mona_lisa(
        image_path="/tmp/mona_lisa.jpg",
        stitches_per_inch=18,
        max_width_cm=30.0,
        max_colors=50,
    )

    print(f"\nResult:")
    print(f"  Pattern: {pattern.width} x {pattern.height} stitches")
    print(f"  Colors: {len(pattern.color_legend)} DMC colors")
    print(f"  Image: {output_path}")

    # Print top 10 colors by usage
    print("\nTop 10 DMC colors used:")
    for dmc, (name, r, g, b, symbol, count) in sorted(
        pattern.color_legend.items(), key=lambda x: -x[1][5]
    )[:10]:
        pct = count / (pattern.width * pattern.height) * 100
        print(f"  DMC {dmc:>6} ({name:30s}) {symbol}: {count:5d} stitches ({pct:.1f}%)")
