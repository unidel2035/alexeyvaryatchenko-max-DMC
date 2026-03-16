"""Tests for the image processor module."""

import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PIL import Image

from image_processor import process_image, EmbroideryPattern


def _make_test_image(width: int = 100, height: int = 100, color: tuple = (255, 0, 0)) -> bytes:
    """Create a simple solid-color test image as bytes."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_gradient_image(width: int = 100, height: int = 100) -> bytes:
    """Create a gradient test image."""
    img = Image.new("RGB", (width, height))
    pixels = []
    for y in range(height):
        for x in range(width):
            r = int(255 * x / width)
            g = int(255 * y / height)
            b = 128
            pixels.append((r, g, b))
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestProcessImage:
    def test_basic_processing(self):
        image_data = _make_test_image()
        pattern = process_image(image_data, pattern_width=20)
        assert isinstance(pattern, EmbroideryPattern)
        assert pattern.width == 20
        assert pattern.height > 0

    def test_pattern_dimensions(self):
        image_data = _make_test_image(200, 100)
        pattern = process_image(image_data, pattern_width=40, pattern_height=20)
        assert pattern.width == 40
        assert pattern.height == 20

    def test_auto_height(self):
        image_data = _make_test_image(200, 200)
        pattern = process_image(image_data, pattern_width=40)
        # height should be auto-calculated (approximately half due to stitch aspect ratio)
        assert pattern.height > 0

    def test_grid_dimensions_match(self):
        image_data = _make_test_image()
        pattern = process_image(image_data, pattern_width=15, pattern_height=10)
        assert len(pattern.grid) == 10
        for row in pattern.grid:
            assert len(row) == 15

    def test_grid_cells_are_valid_dmc(self):
        from dmc_colors import DMC_COLORS
        image_data = _make_test_image()
        pattern = process_image(image_data, pattern_width=10, pattern_height=5)
        for row in pattern.grid:
            for cell in row:
                dmc_num, name, r, g, b = cell
                assert dmc_num in DMC_COLORS, f"Unknown DMC number: {dmc_num}"

    def test_color_legend_populated(self):
        image_data = _make_gradient_image()
        pattern = process_image(image_data, pattern_width=20, max_colors=10)
        assert len(pattern.color_legend) > 0
        assert len(pattern.color_legend) <= 10

    def test_color_legend_structure(self):
        image_data = _make_test_image()
        pattern = process_image(image_data, pattern_width=10, pattern_height=5)
        for dmc_num, info in pattern.color_legend.items():
            name, r, g, b, symbol, count = info
            assert isinstance(name, str)
            assert 0 <= r <= 255
            assert isinstance(symbol, str)
            assert count > 0

    def test_color_counts_sum(self):
        image_data = _make_gradient_image()
        pattern = process_image(image_data, pattern_width=20, pattern_height=10)
        total_cells = pattern.width * pattern.height
        total_counted = sum(v[5] for v in pattern.color_legend.values())
        assert total_counted == total_cells

    def test_max_colors_respected(self):
        image_data = _make_gradient_image()
        pattern = process_image(image_data, pattern_width=30, max_colors=5)
        assert len(pattern.color_legend) <= 5

    def test_single_color_image(self):
        image_data = _make_test_image(50, 50, color=(0, 0, 0))
        pattern = process_image(image_data, pattern_width=10, pattern_height=5)
        # A solid black image should map to DMC 310 (Black)
        assert len(pattern.color_legend) == 1
        dmc_num = list(pattern.color_legend.keys())[0]
        assert dmc_num == "310"

    def test_jpeg_input(self):
        img = Image.new("RGB", (80, 80), (100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        pattern = process_image(buf.getvalue(), pattern_width=15)
        assert pattern.width == 15


class TestEmbroideryPatternSerialization:
    def test_to_dict(self):
        image_data = _make_test_image()
        pattern = process_image(image_data, pattern_width=5, pattern_height=3)
        d = pattern.to_dict()
        assert d["width"] == 5
        assert d["height"] == 3
        assert len(d["grid"]) == 3
        assert len(d["grid"][0]) == 5

    def test_to_json(self):
        import json
        image_data = _make_test_image()
        pattern = process_image(image_data, pattern_width=5, pattern_height=3)
        j = pattern.to_json()
        parsed = json.loads(j)
        assert parsed["width"] == 5

    def test_to_html_table(self):
        image_data = _make_gradient_image()
        pattern = process_image(image_data, pattern_width=10, pattern_height=8)
        html = pattern.to_html_table()
        assert "<table" in html
        assert "Легенда цветов" in html
        assert "DMC" in html

    def test_html_contains_dimensions(self):
        image_data = _make_test_image()
        pattern = process_image(image_data, pattern_width=12, pattern_height=7)
        html = pattern.to_html_table()
        assert "12" in html
        assert "7" in html
