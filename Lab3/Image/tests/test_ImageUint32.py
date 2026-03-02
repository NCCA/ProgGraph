from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from image import ImageUint32, rgba_uint32
from PIL import Image as PILImage


# Fixtures for colours
@pytest.fixture
def black() -> rgba_uint32:
    return rgba_uint32(0, 0, 0, 255)


@pytest.fixture
def white() -> rgba_uint32:
    return rgba_uint32(255, 255, 255)


@pytest.fixture
def red() -> rgba_uint32:
    return rgba_uint32(255, 0, 0)


# --- Tests for rgba_uint32 class ---


def test_rgba_uint32_defaults():
    """Test that rgba_uint32 defaults to black with full alpha."""
    c = rgba_uint32()
    assert (c.r, c.g, c.b, c.a) == (0, 0, 0, 255)


def test_rgba_uint32_custom_init():
    """Test custom initialization of rgba_uint32."""
    c = rgba_uint32(10, 20, 30, 40)
    assert (c.r, c.g, c.b, c.a) == (10, 20, 30, 40)


def test_rgba_uint32_as_tuple(red):
    """Test the as_tuple method."""
    assert red.as_tuple() == (255, 0, 0, 255)


@pytest.mark.parametrize(
    "component, value", [("r", -1), ("g", 256), ("b", 1000), ("a", -100), ("r", 25.5)]
)
def test_rgba_uint32_out_of_range_raises_value_error(component, value):
    """Test that values outside 0-255 or non-integers raise ValueError."""
    with pytest.raises(ValueError):
        kwargs = {component: value}
        rgba_uint32(**kwargs)


def test_rgba_uint32_is_frozen(red):
    """Test that the rgba_uint32 dataclass is immutable."""
    with pytest.raises(FrozenInstanceError):
        red.r = 100


# --- Tests for Image class ---


@pytest.fixture
def small_image() -> Image:
    """A small 10x10 image for testing."""
    return ImageUint32(10, 10)


def test_image_init_defaults(white):
    """Test that a new image defaults to white."""
    img = ImageUint32(2, 2)
    assert img.width == 2
    assert img.height == 2
    assert img[0, 0] == white
    assert img[1, 1] == white


def test_image_init_with_rgba_uint32(red):
    """Test image creation with an rgba_uint32 object."""
    img = ImageUint32(2, 2, fill_colour=red)
    assert img[0, 0] == red


def test_image_init_with_3_tuple():
    """Test image creation with a 3-element tuple."""
    img = ImageUint32(2, 2, fill_colour=(0, 255, 0))
    assert img[0, 0] == rgba_uint32(0, 255, 0, 255)


def test_image_init_with_4_tuple():
    """Test image creation with a 4-element tuple."""
    img = ImageUint32(2, 2, fill_colour=(0, 0, 255, 128))
    assert img[0, 0] == rgba_uint32(0, 0, 255, 128)


@pytest.mark.parametrize("bad_colour", [(1, 2), (1, 2, 3, 4, 5), "red"])
def test_image_init_bad_colour_raises_error(bad_colour):
    """Test that invalid fill_colour types raise errors."""
    with pytest.raises((ValueError, TypeError)):
        ImageUint32(2, 2, fill_colour=bad_colour)


def test_image_properties(small_image):
    """Test width, height, and shape properties."""
    assert small_image.width == 10
    assert small_image.height == 10
    assert small_image.shape == (10, 10)


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
        small_image[10, 10] = rgba_uint32()


def test_clear_method(small_image, red, white):
    """Test the clear() method."""
    assert small_image[0, 0] == white
    small_image.clear(red)
    assert small_image[0, 0] == red
    assert small_image[9, 9] == red


def test_save_method(small_image, tmp_path):
    """Test that the save method creates a file."""
    file_path = tmp_path / "test_image.png"
    small_image.save(str(file_path))
    assert Path(file_path).exists()
    # Optional: check more properties of the saved file
    with PILImage.open(file_path) as saved_img:
        assert saved_img.size == (10, 10)
        assert saved_img.mode == "RGBA"
