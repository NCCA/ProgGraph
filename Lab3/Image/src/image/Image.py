from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from PIL import Image as PILImage


# make the rgba immutable using frozen=True
@dataclass(frozen=True)
class rgba:
    """A dataclass to represent an RGBA color, with validation."""

    r: int = 0
    g: int = 0
    b: int = 0
    a: int = 255

    def __post_init__(self):
        """Validate that RGBA values are within the 0-255 range."""
        for component in ("r", "g", "b", "a"):
            value = getattr(self, component)
            if not isinstance(value, int) or not (0 <= value <= 255):
                raise ValueError(
                    f"RGBA component '{component}' must be an integer between 0 and 255, but got {value}"
                )

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return the colour as a tuple."""
        return (self.r, self.g, self.b, self.a)

    def __iter__(self):
        """Return an iterator over the RGBA components."""
        return iter((self.r, self.g, self.b, self.a))


class Image:
    """A class to represent an image, built on numpy and Pillow."""

    def __init__(
        self, width: int, height: int, fill_colour: Union[rgba, tuple, None] = None
    ):
        """
        Initialize the Image object.

        Args:
            width: The width of the image in pixels.
            height: The height of the image in pixels.
            fill_colour: The initial colour of the image. Can be an rgba object,
                         a 3 or 4 element tuple, or None for a default white image.
        """
        self._width = width
        self._height = height

        fill_colour = self._validate_rgba(fill_colour)
        self._rgba_data = np.full(
            (self._height, self._width, 4), fill_colour, dtype=np.uint8
        )

    def _check_bounds(self, x: int, y: int) -> None:
        """
        Check if the given x,y coordinates are within the bounds of the image.

        Args:
            x (int): The x coordinate to check.
            y (int): The y coordinate to check.

        Raises:
            IndexError: If the coordinates are out of range.
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(
                f"x,y values out of range {x=} {self.width=} {y=} {self.height=}"
            )

    def _check_bounds(self, x: int, y: int) -> None:
        """
        Check if the given x,y coordinates are within the bounds of the image.

        Args:
            x (int): The x coordinate to check.
            y (int): The y coordinate to check.

        Raises:
            IndexError: If the coordinates are out of range.
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(
                f"x,y values out of range {x=} {self.width=} {y=} {self.height=}"
            )

    def _validate_rgba(
        self, value: Union[rgba, tuple, None]
    ) -> Tuple[int, int, int, int]:
        """
        Check to see if a value is correct and return a tuple of RGBA values.

        Args:
            value: The value to check.

        Returns:
            A tuple of RGBA values.
        """
        match value:
            case None:
                return (255, 255, 255, 255)
            case rgba(r, g, b, a):
                return (r, g, b, a)
            case (r, g, b):
                return (r, g, b, 255)
            case (r, g, b, a):
                return (r, g, b, a)
            case _:  # catch all
                raise TypeError(f"Invalid type for RGBA colour: {type(value).__name__}")

    @property
    def width(self) -> int:
        """Get the width of the image in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Get the height of the image in pixels."""
        return self._height

    def set_pixel(self, x: int, y: int, colour: Union[rgba, tuple, None]):
        """
        Set the colour of a single pixel.

        Args:
            x: The x-coordinate of the pixel.
            y: The y-coordinate of the pixel.
            colour: The rgba object representing the colour.
        """
        self._check_bounds(x, y)
        colour = self._validate_rgba(colour)
        self._rgba_data[y, x] = colour

    def get_pixel(self, x: int, y: int) -> Tuple[int, int, int, int]:  # noqa
        """
        Set the colour of a single pixel.

        Args:
            x: The x-coordinate of the pixel.
            y: The y-coordinate of the pixel.
            colour: The rgba object representing the colour.
        """
        self._check_bounds(x, y)
        return self._rgba_data[y, x]

    def clear(self, colour: Union[rgba, tuple, None]):
        """
        Clear the image with a given colour.

        Args:
            colour: The rgba object representing the colour to fill the image with.
        """
        colour = self._validate_rgba(colour)
        self._rgba_data[:] = colour

    @property
    def pixels(self) -> np.ndarray:
        """Get the raw pixel data as a numpy array."""
        return self._rgba_data

    @pixels.setter
    def pixels(self, value: np.ndarray):
        """Set the raw pixel data from a numpy array."""
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
        img = PILImage.fromarray(self._rgba_data)
        img.save(name)

    def __getitem__(self, key: tuple[int, int]) -> rgba:
        """
        Get the colour of a pixel using subscript notation (e.g., img[x, y]).
        """
        x, y = key
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise IndexError("Pixel coordinates out of bounds.")
        # Return an rgba object for consistency, converting numpy types to int
        pixel_data = self._rgba_data[y, x]
        return rgba(
            int(pixel_data[0]),
            int(pixel_data[1]),
            int(pixel_data[2]),
            int(pixel_data[3]),
        )

    def __setitem__(self, key: tuple[int, int], colour: Union[rgba, tuple, None]):
        """
        Set the colour of a pixel using subscript notation (e.g., img[x, y] = colour).
        """
        x, y = key
        self._check_bounds(x, y)
        colour = self._validate_rgba(colour)
        self._rgba_data[y, x] = colour

    def line(
        self, sx: int, sy: int, ex: int, ey: int, colour: Union[rgba, tuple, None]
    ) -> None:
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
        self, tx: int, ty: int, bx: int, by: int, colour: Union[rgba, tuple, None]
    ) -> None:
        x0, x1 = sorted((tx, bx))
        y0, y1 = sorted((ty, by))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.set_pixel(x, y, colour=colour)


if __name__ == "__main__":
    # Create a red image using an rgba object
    red_colour = rgba(r=255, g=0, b=0)
    img_red = Image(512, 512, red_colour)
    img_red.save("red.png")

    # Create a green image using a tuple
    img_green = Image(512, 512, (0, 255, 0))
    img_green.save("green.png")

    # Create a blue image with transparency
    blue_colour = rgba(r=0, g=0, b=255, a=128)
    img_blue = Image(512, 512, blue_colour)
    img_blue.save("blue_transparent.png")

    # Create a default (white) image
    img_white = Image(256, 256)
    img_white.save("white.png")

    # Set a single pixel to black
    img_white.set_pixel(128, 128, rgba())  # rgba() defaults to black
    img_white.save("white_with_black_pixel.png")

    # Clear the image to a semi-transparent purple
    purple_colour = rgba(r=128, g=0, b=128, a=128)
    img_white.clear(purple_colour)
    img_white.save("purple.png")

    print("Generated test images.")
