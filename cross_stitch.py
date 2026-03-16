"""
Generate a cross-stitch pattern from the Mona Lisa image.

Steps:
1. Downscale source to ~150 stitches wide (preserving aspect ratio)
2. Map each pixel to nearest DMC thread color
3. Output: pixel image, grid chart with symbols, and color legend
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import string

from dmc_colors import DMC_COLORS


def is_blue_purple(r, g, b):
    """Check if a color is blue or purple — not suitable for Mona Lisa."""
    # Blue: blue channel dominates
    if b > r * 1.2 and b > g * 1.1 and b > 50:
        return True
    # Purple/violet
    if b > g * 1.3 and r < b and b > 50:
        return True
    return False


def build_dmc_palette(exclude_blue=False):
    """Build numpy array of all DMC colors."""
    colors = []
    keys = []
    for dmc_num, (name, r, g, b) in DMC_COLORS.items():
        if exclude_blue and is_blue_purple(r, g, b):
            continue
        colors.append([r, g, b])
        keys.append((dmc_num, name))
    return np.array(colors, dtype=np.float64), keys


def map_pixels_to_dmc(arr, max_colors=60, exclude_blue=True):
    """Replace every pixel with nearest DMC color, limiting to max_colors most used."""
    palette, keys = build_dmc_palette(exclude_blue=exclude_blue)
    weights = np.array([0.299, 0.587, 0.114])

    h, w, _ = arr.shape
    pixels = arr.reshape(-1, 3).astype(np.float64)

    # First pass: find nearest DMC for each pixel
    diff = pixels[:, np.newaxis, :] - palette[np.newaxis, :, :]
    dist = np.sum(diff ** 2 * weights, axis=2)
    nearest_idx = np.argmin(dist, axis=1)

    if max_colors and len(set(nearest_idx)) > max_colors:
        # Keep only top N most frequent colors
        unique, counts = np.unique(nearest_idx, return_counts=True)
        top_indices = unique[np.argsort(-counts)[:max_colors]]
        top_set = set(top_indices)
        reduced_palette = palette[top_indices]

        # Re-map pixels not in top set to nearest within reduced palette
        for i in range(len(nearest_idx)):
            if nearest_idx[i] not in top_set:
                pixel = pixels[i]
                d = np.sum((reduced_palette - pixel) ** 2 * weights, axis=1)
                nearest_idx[i] = top_indices[np.argmin(d)]

        print(f"Reduced palette: {len(set(nearest_idx))} colors (from full match)")

    mapped = palette[nearest_idx].reshape(h, w, 3).astype(np.uint8)
    idx_map = nearest_idx.reshape(h, w)

    return mapped, idx_map, keys, palette


def is_blue_artifact(rgb, neighbors_rgb):
    """Check if a pixel is an unwanted blue tint in a non-blue area."""
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    # Blue dominates significantly
    if not (b > r * 1.3 and b > g * 1.2 and b > 35):
        return False
    # Check if neighbors are also blue — if yes, it's intentional
    if len(neighbors_rgb) == 0:
        return True
    blue_neighbors = 0
    for nr, ng, nb in neighbors_rgb:
        if float(nb) > float(nr) * 1.3 and float(nb) > float(ng) * 1.2:
            blue_neighbors += 1
    # If most neighbors are NOT blue, this pixel is an artifact
    return blue_neighbors < len(neighbors_rgb) * 0.4


def remove_blue_artifacts(idx_map, palette, keys):
    """Replace blue artifact pixels with median of non-blue neighbors, remapped to DMC."""
    h, w = idx_map.shape
    weights = np.array([0.299, 0.587, 0.114])
    result = idx_map.copy()
    changed = 0

    # Build set of allowed (non-problematic-blue) palette indices
    for y in range(h):
        for x in range(w):
            rgb = palette[result[y, x]]
            # Gather neighbor colors
            neighbors = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbors.append(palette[result[ny, nx]])

            if is_blue_artifact(rgb, neighbors):
                # Replace with median of non-blue neighbors
                non_blue = [n for n in neighbors if not is_blue_artifact(n, [])]
                if len(non_blue) >= 2:
                    median = np.median(non_blue, axis=0)
                else:
                    median = np.median(neighbors, axis=0)
                # Find nearest DMC to median
                d = np.sum((palette - median) ** 2 * weights, axis=1)
                result[y, x] = np.argmin(d)
                changed += 1

    print(f"  Removed {changed} blue artifact pixels")
    return result


def remove_isolated_pixels(idx_map, passes=2):
    """Replace isolated (single) pixels with the most common neighbor color."""
    h, w = idx_map.shape
    result = idx_map.copy()
    total_changed = 0

    for p in range(passes):
        changed = 0
        new_result = result.copy()
        for y in range(h):
            for x in range(w):
                current = result[y, x]
                # Gather neighbor indices
                neighbors = []
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            neighbors.append(result[ny, nx])

                if len(neighbors) == 0:
                    continue

                # Count how many neighbors share the same color
                same = sum(1 for n in neighbors if n == current)

                # If pixel is isolated (0 or 1 same-color neighbor), replace
                if same <= 1:
                    # Use most common neighbor color
                    from collections import Counter
                    counts = Counter(neighbors)
                    most_common = counts.most_common(1)[0][0]
                    if most_common != current:
                        new_result[y, x] = most_common
                        changed += 1

        result = new_result
        total_changed += changed
        print(f"  Pass {p+1}: replaced {changed} isolated pixels")

    return result


# Symbols for chart — easily distinguishable on print
SYMBOLS = list(string.ascii_uppercase) + list(string.digits) + [
    "@", "#", "$", "%", "&", "*", "+", "=", "~", "^",
    "!", "?", "<", ">", "/", "\\", "|", ":", ";", ".",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z",
]


def generate_chart(mapped_arr, idx_map, keys, cell_size=14):
    """Generate a grid chart with symbols and 10-cell ruler marks."""
    h, w = idx_map.shape

    # Determine which DMC colors are actually used and assign symbols
    unique_indices = sorted(set(idx_map.flatten()))
    if len(unique_indices) > len(SYMBOLS):
        print(f"Warning: {len(unique_indices)} colors but only {len(SYMBOLS)} symbols. Some will repeat.")
    idx_to_symbol = {}
    for i, idx in enumerate(unique_indices):
        idx_to_symbol[idx] = SYMBOLS[i % len(SYMBOLS)]

    margin_top = cell_size * 2
    margin_left = cell_size * 3
    chart_w = margin_left + w * cell_size + 1
    chart_h = margin_top + h * cell_size + 1

    chart = Image.new("RGB", (chart_w, chart_h), (255, 255, 255))
    draw = ImageDraw.Draw(chart)

    # Try to load a monospace font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", cell_size - 3)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", cell_size - 5)
    except (IOError, OSError):
        font = ImageFont.load_default()
        small_font = font

    # Draw column numbers (every 10)
    for x in range(0, w, 10):
        px = margin_left + x * cell_size + cell_size // 2
        draw.text((px, 2), str(x + 1), fill=(0, 0, 0), font=small_font, anchor="mt")

    # Draw row numbers (every 10)
    for y in range(0, h, 10):
        py = margin_top + y * cell_size + cell_size // 2
        draw.text((margin_left - 4, py), str(y + 1), fill=(0, 0, 0), font=small_font, anchor="rm")

    # Fill cells with background color and symbol
    for y in range(h):
        for x in range(w):
            x0 = margin_left + x * cell_size
            y0 = margin_top + y * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            idx = idx_map[y, x]
            r, g, b = mapped_arr[y, x]
            sym = idx_to_symbol[idx]

            # Cell background
            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b))

            # Symbol color: white on dark, black on light
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
            cx = x0 + cell_size // 2
            cy = y0 + cell_size // 2
            draw.text((cx, cy), sym, fill=text_color, font=font, anchor="mm")

    # Grid lines — thick every 10, thin otherwise
    for x in range(w + 1):
        px = margin_left + x * cell_size
        lw = 2 if x % 10 == 0 else 1
        color = (0, 0, 0) if x % 10 == 0 else (180, 180, 180)
        draw.line([(px, margin_top), (px, margin_top + h * cell_size)], fill=color, width=lw)

    for y in range(h + 1):
        py = margin_top + y * cell_size
        lw = 2 if y % 10 == 0 else 1
        color = (0, 0, 0) if y % 10 == 0 else (180, 180, 180)
        draw.line([(margin_left, py), (margin_left + w * cell_size, py)], fill=color, width=lw)

    return chart, unique_indices, idx_to_symbol


def generate_legend(unique_indices, idx_to_symbol, keys, cell_size=20):
    """Generate a legend image: symbol + color swatch + DMC number + name."""
    palette_arr, _ = build_dmc_palette()
    n = len(unique_indices)
    cols = 3
    rows = (n + cols - 1) // cols
    col_width = 280
    row_height = cell_size + 6

    leg_w = cols * col_width + 20
    leg_h = rows * row_height + 50

    legend = Image.new("RGB", (leg_w, leg_h), (255, 255, 255))
    draw = ImageDraw.Draw(legend)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", cell_size - 5)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", cell_size - 2)
    except (IOError, OSError):
        font = ImageFont.load_default()
        title_font = font

    draw.text((10, 5), f"DMC Legend — {n} colors", fill=(0, 0, 0), font=title_font)

    for i, idx in enumerate(unique_indices):
        col = i // rows
        row = i % rows
        x0 = 10 + col * col_width
        y0 = 35 + row * row_height

        dmc_num, dmc_name = keys[idx]
        r, g, b = int(palette_arr[idx][0]), int(palette_arr[idx][1]), int(palette_arr[idx][2])
        sym = idx_to_symbol[idx]

        # Symbol
        draw.text((x0, y0 + 2), sym, fill=(0, 0, 0), font=font)

        # Color swatch
        sw_x = x0 + 18
        draw.rectangle([sw_x, y0 + 1, sw_x + cell_size, y0 + cell_size + 1], fill=(r, g, b), outline=(0, 0, 0))

        # DMC number and name
        draw.text((sw_x + cell_size + 5, y0 + 2), f"{dmc_num} {dmc_name}", fill=(0, 0, 0), font=font)

    return legend


def main():
    base_dir = os.path.dirname(__file__)
    source_path = os.path.join(base_dir, "mona_lisa_source.png")

    img = Image.open(source_path).convert("RGB")
    orig_w, orig_h = img.size
    print(f"Original: {orig_w}x{orig_h}")

    # Downscale to ~150 stitches wide
    target_w = 150
    ratio = target_w / orig_w
    target_h = int(orig_h * ratio)
    small = img.resize((target_w, target_h), Image.LANCZOS)
    print(f"Downscaled to: {target_w}x{target_h} stitches")

    # Map to DMC
    arr = np.array(small)
    mapped, idx_map, keys, palette = map_pixels_to_dmc(arr)

    print(f"DMC colors before cleanup: {len(set(idx_map.flatten()))}")

    # Post-processing
    print("Removing isolated pixels...")
    idx_map = remove_isolated_pixels(idx_map, passes=3)

    # Rebuild mapped image from cleaned idx_map
    h, w = idx_map.shape
    mapped = palette[idx_map.flatten()].reshape(h, w, 3).astype(np.uint8)

    unique_indices = sorted(set(idx_map.flatten()))
    print(f"DMC colors after cleanup: {len(unique_indices)}")

    # Save pixel image (upscaled for visibility)
    pixel_img = Image.fromarray(mapped)
    scale = 4
    pixel_up = pixel_img.resize((target_w * scale, target_h * scale), Image.NEAREST)
    pixel_path = os.path.join(base_dir, "mona_lisa_pattern.png")
    pixel_up.save(pixel_path, "PNG")
    print(f"Saved pixel pattern: {pixel_path}")

    # Generate chart with symbols
    print("Generating chart...")
    chart, unique_idx, idx_to_sym = generate_chart(mapped, idx_map, keys, cell_size=14)
    chart_path = os.path.join(base_dir, "mona_lisa_chart.png")
    chart.save(chart_path, "PNG")
    print(f"Saved chart: {chart_path}")

    # Generate legend
    legend = generate_legend(unique_idx, idx_to_sym, keys)
    legend_path = os.path.join(base_dir, "mona_lisa_legend.png")
    legend.save(legend_path, "PNG")
    print(f"Saved legend: {legend_path}")

    # Print summary
    print(f"\n=== Cross-stitch pattern ready ===")
    print(f"Size: {target_w} x {target_h} stitches")
    print(f"On Aida 14: {target_w/14*2.54:.1f} x {target_h/14*2.54:.1f} cm")
    print(f"On Aida 16: {target_w/16*2.54:.1f} x {target_h/16*2.54:.1f} cm")
    print(f"On Aida 18: {target_w/18*2.54:.1f} x {target_h/18*2.54:.1f} cm")
    print(f"DMC colors: {len(unique_indices)}")
    print(f"\nFiles:")
    print(f"  Pattern:  {pixel_path}")
    print(f"  Chart:    {chart_path}")
    print(f"  Legend:   {legend_path}")


if __name__ == "__main__":
    main()
