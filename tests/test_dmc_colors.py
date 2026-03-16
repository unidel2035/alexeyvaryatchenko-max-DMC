"""Tests for the DMC color palette module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dmc_colors import DMC_COLORS, find_closest_dmc, get_all_colors


def test_dmc_colors_not_empty():
    assert len(DMC_COLORS) > 100, "Should have more than 100 DMC colors"


def test_dmc_colors_structure():
    for num, val in list(DMC_COLORS.items())[:5]:
        name, r, g, b = val
        assert isinstance(name, str)
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255


def test_find_closest_black():
    dmc, name, r, g, b = find_closest_dmc(0, 0, 0)
    assert dmc == "310", f"Closest to black should be DMC 310 (Black), got {dmc}"


def test_find_closest_white():
    dmc, name, r, g, b = find_closest_dmc(255, 255, 255)
    assert dmc == "blanc", f"Closest to white should be blanc, got {dmc}"


def test_find_closest_returns_tuple():
    result = find_closest_dmc(128, 64, 32)
    assert len(result) == 5
    dmc_num, name, r, g, b = result
    assert isinstance(dmc_num, str)
    assert isinstance(name, str)
    assert 0 <= r <= 255


def test_get_all_colors():
    colors = get_all_colors()
    assert len(colors) == len(DMC_COLORS)
    for item in colors[:3]:
        assert len(item) == 5  # (num, name, r, g, b)


def test_find_closest_pure_red():
    dmc, name, r, g, b = find_closest_dmc(255, 0, 0)
    # Should return some red-ish DMC color
    assert r > 150, f"Expected red-dominant color, got rgb({r},{g},{b})"


def test_find_closest_pure_blue():
    dmc, name, r, g, b = find_closest_dmc(0, 0, 255)
    # Should return some blue-ish DMC color
    assert b > g and b > r, f"Expected blue-dominant color, got rgb({r},{g},{b})"
