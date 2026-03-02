from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from image import Image, rgba


# Fixtures for colours
@pytest.fixture
def black() -> rgba:
    return rgba(0, 0, 0)


@pytest.fixture
def white() -> rgba:
    return rgba(255, 255, 255)


@pytest.fixture
def red() -> rgba:
    return rgba(255, 0, 0)


# --- Tests for rgba class ---


def test_rgba_defaults():
    """Test that rgba defaults to black with full alpha."""
    c = rgba()
    assert (c.r, c.g, c.b, c.a) == (0, 0, 0, 255)


def test_rgba_custom_init():
    """Test custom initialization of rgba."""
    c = rgba(10, 20, 30, 40)
    assert (c.r, c.g, c.b, c.a) == (10, 20, 30, 40)


def test_rgba_as_tuple(red):
    """Test the as_tuple method."""
    assert red.as_tuple() == (255, 0, 0, 255)


@pytest.mark.parametrize(
    "component, value", [("r", -1), ("g", 256), ("b", 1000), ("a", -100), ("r", 25.5)]
)
def test_rgba_out_of_range_raises_value_error(component, value):
    """Test that values outside 0-255 or non-integers raise ValueError."""
    with pytest.raises(ValueError):
        kwargs = {component: value}
        rgba(**kwargs)


def test_rgba_is_frozen(red):
    """Test that the rgba dataclass is immutable."""
    with pytest.raises(FrozenInstanceError):
        red.r = 100


# --- Tests for Image class ---


@pytest.fixture
def small_image() -> Image:
    """A small 10x10 image for testing."""
    return Image(10, 10)


def test_image_init_defaults(white):
    """Test that a new image defaults to white."""
    img = Image(2, 2)
    assert img.width == 2
    assert img.height == 2
    assert img[0, 0] == white
    assert img[1, 1] == white


def test_image_init_with_rgba(red):
    """Test image creation with an rgba object."""
    img = Image(2, 2, fill_colour=red)
    assert img[0, 0] == red


def test_image_init_with_3_tuple():
    """Test image creation with a 3-element tuple."""
    img = Image(2, 2, fill_colour=(0, 255, 0))
    assert img[0, 0] == rgba(0, 255, 0, 255)


def test_image_init_with_4_tuple():
    """Test image creation with a 4-element tuple."""
    img = Image(2, 2, fill_colour=(0, 0, 255, 128))
    assert img[0, 0] == rgba(0, 0, 255, 128)


@pytest.mark.parametrize("bad_colour", [(1, 2), (1, 2, 3, 4, 5), "red"])
def test_image_init_bad_colour_raises_error(bad_colour):
    """Test that invalid fill_colour types raise errors."""
    with pytest.raises((ValueError, TypeError)):
        Image(2, 2, fill_colour=bad_colour)


def test_image_properties(small_image):
    """Test width, height, and shape properties."""
    assert small_image.width == 10
    assert small_image.height == 10
    assert small_image.shape == (10, 10, 4)


def test_getitem_setitem(small_image, red, black, white):
    """Test getting and setting pixels using subscript notation."""
    assert small_image[5, 5] == white  # Default white
    small_image[5, 5] = red
    assert small_image[5, 5] == red
    small_image[0, 0] = black
    assert small_image[0, 0] == black


def test_getitem_out_of_bounds_raises_index_error(small_image):
    """Test that accessing pixels out of bounds raises IndexError."""
    with pytest.raises(IndexError):
        _ = small_image[10, 10]
    with pytest.raises(IndexError):
        small_image[10, 10] = rgba()


def test_clear_method(small_image, red, white):
    """Test the clear() method."""
    assert small_image[0, 0] == white
    small_image.clear(red)
    assert small_image[0, 0] == red
    assert small_image[9, 9] == red


def test_pixels_property(small_image, black):
    """Test the pixels property for getting and setting raw data."""
    assert isinstance(small_image.pixels, np.ndarray)
    new_pixels = np.full((10, 10, 4), black.as_tuple(), dtype=np.uint8)
    small_image.pixels = new_pixels
    assert small_image[0, 0] == black
    assert np.array_equal(small_image.pixels, new_pixels)


def test_save_method(small_image, tmp_path):
    """Test that the save method creates a file."""
    file_path = tmp_path / "test_image.png"
    small_image.save(str(file_path))
    assert Path(file_path).exists()
    # Optional: check more properties of the saved file
    with PILImage.open(file_path) as saved_img:
        assert saved_img.size == (10, 10)
        assert saved_img.mode == "RGBA"
