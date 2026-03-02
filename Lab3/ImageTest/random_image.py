#!/usr/bin/env -S uv run --script
import random

from image import RGBA, Image

WIDTH = 400
HEIGHT = 400
NUM_IMAGES = 5
MAX_SHAPES = 500
MIN_SHAPES = 5


def get_random_color() -> RGBA:
    """Returns a random RGBA color."""
    return RGBA(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def draw_random_line(img: Image):
    """Draws a random line on the image."""
    width = img.width()
    height = img.height()
    sx = random.randint(0, width - 1)
    sy = random.randint(0, height - 1)
    ex = random.randint(0, width - 1)
    ey = random.randint(0, height - 1)
    img.line(sx, sy, ex, ey, get_random_color())


def draw_random_rectangle(img: Image):
    """Draws a random rectangle on the image."""
    width = img.width()
    height = img.height()
    tx = random.randint(0, width - 1)
    ty = random.randint(0, height - 1)
    bx = random.randint(0, width - 1)
    by = random.randint(0, height - 1)
    img.rectangle(tx, ty, bx, by, get_random_color())


def main():
    """Generates a series of random images."""

    drawing_functions = [draw_random_line, draw_random_rectangle]

    for i in range(NUM_IMAGES):
        img = Image(WIDTH, HEIGHT, fill=RGBA(0, 0, 0))
        num_shapes = random.randint(MIN_SHAPES, MAX_SHAPES)

        for _ in range(num_shapes):
            draw_func = random.choice(drawing_functions)
            draw_func(img)

        filename = f"random_image_{i}.png"
        if img.save(filename):
            print(f"Saved {filename}")
        else:
            print(f"Failed to save {filename}")


if __name__ == "__main__":
    main()
