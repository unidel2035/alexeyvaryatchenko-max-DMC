"""
Image processor for converting images to DMC embroidery patterns.
Quantizes colors to the DMC palette and produces a cross-stitch pattern grid.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from dmc_colors import find_closest_dmc


@dataclass
class EmbroideryPattern:
    """Represents a cross-stitch embroidery pattern."""

    width: int
    height: int
    # grid[row][col] = (dmc_number, name, r, g, b)
    grid: list[list[tuple]]
    # color_legend: {dmc_number: (name, r, g, b, symbol, count)}
    color_legend: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize pattern to a JSON-friendly dict."""
        return {
            "width": self.width,
            "height": self.height,
            "grid": [
                [
                    {
                        "dmc": cell[0],
                        "name": cell[1],
                        "rgb": [cell[2], cell[3], cell[4]],
                    }
                    for cell in row
                ]
                for row in self.grid
            ],
            "color_legend": {
                dmc: {
                    "name": v[0],
                    "rgb": list(v[1:4]),
                    "symbol": v[4],
                    "count": v[5],
                }
                for dmc, v in self.color_legend.items()
            },
        }

    def to_json(self) -> str:
        """Serialize pattern to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_html_table(self) -> str:
        """Render pattern as an HTML table for web display."""
        # Build legend symbols
        legend_html = self._legend_html()
        table_html = self._table_html()
        return f"""
<div class="pattern-container">
  <div class="pattern-info">
    <p><strong>Размер:</strong> {self.width} × {self.height} крестиков</p>
    <p><strong>Цветов:</strong> {len(self.color_legend)}</p>
  </div>
  {table_html}
  {legend_html}
</div>
""".strip()

    def _table_html(self) -> str:
        cell_size = max(8, min(20, 300 // max(self.width, self.height)))
        rows = []
        for row in self.grid:
            cells = []
            for cell in row:
                dmc, name, r, g, b = cell
                symbol = self.color_legend.get(dmc, ("", 0, 0, 0, "?", 0))[4]
                # Contrast text color
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "#000" if luminance > 128 else "#fff"
                cells.append(
                    f'<td style="background:rgb({r},{g},{b});width:{cell_size}px;'
                    f'height:{cell_size}px;font-size:{max(6, cell_size - 4)}px;'
                    f'color:{text_color};text-align:center;border:1px solid rgba(0,0,0,0.15);'
                    f'padding:0;" title="DMC {dmc} - {name}">{symbol}</td>'
                )
            rows.append(f"<tr>{''.join(cells)}</tr>")
        return (
            '<div style="overflow:auto;max-height:600px;">'
            '<table style="border-collapse:collapse;font-family:monospace;">'
            + "".join(rows)
            + "</table></div>"
        )

    def _legend_html(self) -> str:
        items = []
        for dmc, (name, r, g, b, symbol, count) in sorted(
            self.color_legend.items(), key=lambda x: -x[1][5]
        ):
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "#000" if luminance > 128 else "#fff"
            items.append(
                f'<tr>'
                f'<td style="background:rgb({r},{g},{b});color:{text_color};'
                f'width:24px;height:24px;text-align:center;font-weight:bold;">{symbol}</td>'
                f'<td style="padding:2px 8px;">DMC {dmc}</td>'
                f'<td style="padding:2px 8px;">{name}</td>'
                f'<td style="padding:2px 8px;">{count} кр.</td>'
                f'</tr>'
            )
        return (
            "<h3>Легенда цветов</h3>"
            '<table style="border-collapse:collapse;">'
            "<tr><th>Симв.</th><th>DMC №</th><th>Название</th><th>Кол-во</th></tr>"
            + "".join(items)
            + "</table>"
        )


# Printable symbols used in the pattern grid (avoid ambiguous ones)
_SYMBOLS = (
    "■ □ ▲ ▼ ◆ ○ ● ★ ✦ ✚ ✿ ♦ ♠ ♣ ♥ ▪ △ ▽ ◇ ◈ ◉ ◎ ⊕ ⊗ ⊙ ⊛ ⊞ ⊟ ⊠ ⊡ "
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z "
    "a b c d e f g h i j k l m n o p q r s t u v w x y z "
    "0 1 2 3 4 5 6 7 8 9"
).split()


def process_image(
    image_data: bytes,
    pattern_width: int = 60,
    pattern_height: Optional[int] = None,
    max_colors: int = 30,
) -> EmbroideryPattern:
    """
    Convert an image to a DMC embroidery pattern.

    Args:
        image_data: Raw image bytes (JPEG, PNG, etc.)
        pattern_width: Width of the pattern in stitches (default 60)
        pattern_height: Height of the pattern in stitches. If None, auto-calculated
                        to maintain aspect ratio.
        max_colors: Maximum number of distinct DMC colors to use (default 30)

    Returns:
        EmbroideryPattern with grid and color legend.
    """
    img = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Calculate pattern height to preserve aspect ratio
    if pattern_height is None:
        aspect = img.height / img.width
        pattern_height = max(1, round(pattern_width * aspect * 0.5))  # stitches are taller

    # Resize image to pattern dimensions
    img_small = img.resize((pattern_width, pattern_height), Image.LANCZOS)

    # Reduce colors first using PIL's quantize for efficiency
    img_quantized = img_small.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
    img_rgb = img_quantized.convert("RGB")

    # Map each pixel to nearest DMC color
    import numpy as np
    pixels = list(map(tuple, np.array(img_rgb).reshape(-1, 3)))
    dmc_pixels = [find_closest_dmc(r, g, b) for r, g, b in pixels]

    # Build grid
    grid = []
    for row_idx in range(pattern_height):
        row = []
        for col_idx in range(pattern_width):
            row.append(dmc_pixels[row_idx * pattern_width + col_idx])
        grid.append(row)

    # Build color legend
    color_counts: dict[str, list] = {}
    for cell in dmc_pixels:
        dmc_num, name, r, g, b = cell
        if dmc_num not in color_counts:
            color_counts[dmc_num] = [name, r, g, b, 0]
        color_counts[dmc_num][4] += 1

    # Assign symbols to colors (sorted by frequency descending)
    color_legend = {}
    for i, (dmc_num, info) in enumerate(
        sorted(color_counts.items(), key=lambda x: -x[1][4])
    ):
        symbol = _SYMBOLS[i % len(_SYMBOLS)]
        name, r, g, b, count = info
        color_legend[dmc_num] = (name, r, g, b, symbol, count)

    return EmbroideryPattern(
        width=pattern_width,
        height=pattern_height,
        grid=grid,
        color_legend=color_legend,
    )


def load_image_from_path(path: str) -> bytes:
    """Load image bytes from a file path."""
    with open(path, "rb") as f:
        return f.read()
