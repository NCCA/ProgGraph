from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from PIL import Image as PILImage


# make the rgba_uint32immutable using frozen=True
@dataclass(frozen=True)
class rgba_uint32:
    """A dataclass to represent an rgba_uint32colour, with validation."""

    r: int = 0
    g: int = 0
    b: int = 0
    a: int = 255

    def __post_init__(self):
        """Validate that rgba_uint32values are within the 0-255 range."""
        for component in ("r", "g", "b", "a"):
            value = getattr(self, component)
            if not isinstance(value, int) or not (0 <= value <= 255):
                raise ValueError(
                    f"rgba_uint32component '{component}' must be an integer between 0 and 255, but got {value}"
                )

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return the colour as a tuple."""
        return (self.r, self.g, self.b, self.a)

    def __iter__(self):
        """Return an iterator over the rgba_uint32components."""
        return iter((self.r, self.g, self.b, self.a))

    def to_uint32(self) -> np.uint32:
        """Pack the rgba_uint32colour into a single 32-bit unsigned integer (big-endian)."""
        return np.uint32((self.r << 24) | (self.g << 16) | (self.b << 8) | self.a)

    @classmethod
    def from_uint32(cls, value: np.uint32) -> "rgba":
        """Unpack a 32-bit unsigned integer into an rgba_uint32color."""
        # Assumes big-endian packing (R is in the most significant byte)
        r = (value >> 24) & 0xFF
        g = (value >> 16) & 0xFF
        b = (value >> 8) & 0xFF
        a = value & 0xFF
        return cls(r=int(r), g=int(g), b=int(b), a=int(a))


class ImageUint32:
    """A class to represent an image, built on numpy and Pillow."""

    def __init__(
        self, width: int, height: int, fill_colour: Union[rgba, tuple, None] = None
    ):
        """
        Initialize the Image object.

        Args:
            width: The width of the image in pixels.
            height: The height of the image in pixels.
            fill_colour: The initial colour of the image. Can be an rgba_uint32object,
                         a 3 or 4 element tuple, or None for a default white image.
        """
        self._width = width
        self._height = height

        if fill_colour is None:
            # Default to a white image
            colour_obj = rgba_uint32(r=255, g=255, b=255)
        elif isinstance(fill_colour, tuple):
            if len(fill_colour) == 3:
                colour_obj = rgba_uint32(
                    r=fill_colour[0], g=fill_colour[1], b=fill_colour[2]
                )
            elif len(fill_colour) == 4:
                colour_obj = rgba_uint32(
                    r=fill_colour[0],
                    g=fill_colour[1],
                    b=fill_colour[2],
                    a=fill_colour[3],
                )
            else:
                raise ValueError("fill_colour tuple must have 3 or 4 elements.")
        elif isinstance(fill_colour, rgba_uint32):
            colour_obj = fill_colour
        else:
            raise TypeError("fill_colour must be a tuple or an rgba_uint32object.")

        # Use a 2D array of uint32 for pixel data, with big-endian byte order
        # to ensure (R,G,B,A) packing is consistent for PIL.
        self._rgba_data = np.full(
            (self._height, self._width),
            colour_obj.to_uint32(),
            dtype=np.dtype(">u4"),
        )

    @property
    def width(self) -> int:
        """Get the width of the image in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Get the height of the image in pixels."""
        return self._height

    def set_pixel(self, x: int, y: int, colour: rgba_uint32):
        """
        Set the colour of a single pixel.

        Args:
            x: The x-coordinate of the pixel.
            y: The y-coordinate of the pixel.
            colour: The rgba_uint32object representing the colour.
        """
        if 0 <= x < self._width and 0 <= y < self._height:
            self._rgba_data[y, x] = colour.to_uint32()

    def get_pixel(
        self,
        x: int,
        y: int,
    ) -> Tuple[int, int, int, int]:
        """
        Get the colour of a single pixel.

        Args:
            x: The x-coordinate of the pixel.
            y: The y-coordinate of the pixel.

        Returns:
            A tuple (r, g, b, a) representing the colour.
        """
        if 0 <= x < self._width and 0 <= y < self._height:
            uint32_val = self._rgba_data[y, x]
            # Unpack from big-endian uint32
            r = (uint32_val >> 24) & 0xFF
            g = (uint32_val >> 16) & 0xFF
            b = (uint32_val >> 8) & 0xFF
            a = uint32_val & 0xFF
            return (int(r), int(g), int(b), int(a))
        raise IndexError("Pixel coordinates out of bounds.")

    def clear(self, colour: rgba_uint32):
        """
        Clear the image with a given colour.

        Args:
            colour: The rgba_uint32 object representing the colour to fill the image with.
        """
        self._rgba_data[:] = colour.to_uint32()

    @property
    def pixels(self) -> np.ndarray:
        """Get the raw pixel data as a numpy array of uint32."""
        return self._rgba_data

    @pixels.setter
    def pixels(self, value: np.ndarray):
        """Set the raw pixel data from a numpy array of uint32."""
        if value.shape != (self._height, self._width) or value.dtype != np.dtype(">u4"):
            raise ValueError(
                f"Input array must have shape=({self._height}, {self._width}) "
                f"and dtype='>u4', but got shape={value.shape} and dtype={value.dtype}"
            )
        self._rgba_data = value

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the pixel data array."""
        return self._rgba_data.shape

    def save(self, name: str) -> None:
        """
        Save the image to a file.

        Args:
            name: The path to save the file to. The format is determined from the extension.
        """
        # Create a view of the data with shape (height, width, 4) and dtype uint8.
        # The big-endian dtype of the source array ensures the byte order is (R,G,B,A).
        img_array = self._rgba_data.view(np.uint8).reshape(
            (self._height, self._width, 4)
        )
        img = PILImage.fromarray(img_array)
        img.save(name)

    def __getitem__(self, key: tuple[int, int]) -> rgba:
        """
        Get the colour of a pixel using subscript notation (e.g., img[x, y]).
        """
        x, y = key
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError("Pixel coordinates out of bounds.")
        return rgba_uint32.from_uint32(self._rgba_data[y, x])

    def __setitem__(self, key: tuple[int, int], colour: rgba_uint32):
        """
        Set the colour of a pixel using subscript notation (e.g., img[x, y] = colour).
        """
        x, y = key
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError("Pixel coordinates out of bounds.")
        self._rgba_data[y, x] = colour.to_uint32()

    def line(self, sx: int, sy: int, ex: int, ey: int, colour: rgba_uint32) -> None:
        dx, dy = abs(ex - sx), abs(ey - sy)
        x, y = sx, sy
        sx_sign = 1 if ex > sx else -1
        sy_sign = 1 if ey > sy else -1
        if dx > dy:
            err = dx / 2
            while x != ex:
                self.set_pixel(x, y, colour=colour)
                err -= dy
                if err < 0:
                    y += sy_sign
                    err += dx
                x += sx_sign
        else:
            err = dy / 2
            while y != ey:
                self.set_pixel(x, y, colour=colour)
                err -= dx
                if err < 0:
                    x += sx_sign
                    err += dy
                y += sy_sign
        self.set_pixel(ex, ey, colour=colour)

    def rectangle(
        self, tx: int, ty: int, bx: int, by: int, colour: rgba_uint32
    ) -> None:
        x0, x1 = sorted((tx, bx))
        y0, y1 = sorted((ty, by))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.set_pixel(x, y, colour=colour)


if __name__ == "__main__":
    # Create a red image using an rgba_uint32object
    red_colour = rgba_uint32(r=255, g=0, b=0)
    img_red = ImageUint32(512, 512, red_colour)
    img_red.save("red.png")

    # Create a green image using a tuple
    img_green = ImageUint32(512, 512, (0, 255, 0))
    img_green.save("green.png")

    # Create a blue image with transparency
    blue_colour = rgba_uint32(r=0, g=0, b=255, a=128)
    img_blue = ImageUint32(512, 512, blue_colour)
    img_blue.save("blue_transparent.png")

    # Create a default (white) image
    img_white = ImageUint32(256, 256)
    img_white.save("white.png")

    # Set a single pixel to black
    img_white.set_pixel(128, 128, rgba())  # rgba() defaults to black
    img_white.save("white_with_black_pixel.png")

    # Clear the image to a semi-transparent purple
    purple_colour = rgba_uint32(r=128, g=0, b=128, a=128)
    img_white.clear(purple_colour)
    img_white.save("purple.png")

    print("Generated test images.")
