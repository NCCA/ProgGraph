#!/usr/bin/env -S uv run --script
import random

from image import Image, rgba

WIDTH = 400
HEIGHT = 400
MAX_SHAPES = 1500
MIN_SHAPES = 500


def get_random_color() -> rgba:
    """Returns a random rgba color."""
    return rgba(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def draw_random_line(img: Image):
    """Draws a random line on the image."""
    width = img.width
    height = img.height
    sx = random.randint(0, width - 1)
    sy = random.randint(0, height - 1)
    ex = random.randint(0, width - 1)
    ey = random.randint(0, height - 1)
    img.line(sx, sy, ex, ey, get_random_color())


def main():
    """Generates a series of random images."""

    img = Image(WIDTH, HEIGHT, rgba(255, 255, 255))
    num_shapes = random.randint(MIN_SHAPES, MAX_SHAPES)

    for _ in range(num_shapes):
        draw_random_line(img)

    filename = "random_lines.png"
    if img.save(filename):
        print(f"Saved {filename}")
    else:
        print(f"Failed to save {filename}")


if __name__ == "__main__":
    main()
