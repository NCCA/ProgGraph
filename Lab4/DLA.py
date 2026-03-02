#!/usr/bin/env python

import cProfile
import pstats
import random

from image import Image, rgba


class DLA:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.image = Image(width, height)
        # Clear to white (a=255, opaque)
        self.image.clear(rgba(255, 255, 255, 255))

    def random_seed(self) -> None:
        """Place a random seed. Seeds are transparent (a=0)."""
        x = random.randint(1, self.width - 1)
        y = random.randint(1, self.height - 1)
        self.image.set_pixel(x, y, rgba(255, 255, 255, 0))

    def save_image(self, filename: str) -> None:
        """Save the image to a file."""
        self.image.save(filename)

    def _random_start(self) -> tuple[int, int]:
        x = random.randint(1, self.width - 2)
        y = random.randint(1, self.height - 2)
        return x, y

    def walk(self) -> bool:
        """Perform a random walk for one particle."""
        x, y = self._random_start()
        walking = True
        found = False

        while walking:
            # now move
            x += random.choice([-1, 0, 1])
            y += random.choice([-1, 0, 1])
            # check we are in bounds (FIXED: y vs height)
            if x < 1 or x >= self.width - 1 or y < 1 or y >= self.height - 1:
                walking = False
                found = False
                break
            # now check if we are near the seed
            else:
                # Use a flag to break out of nested loops
                found_neighbor = False
                for x_offset in [-1, 0, 1]:
                    for y_offset in [-1, 0, 1]:
                        try:
                            # Use __getitem__ which returns an rgba object
                            color = self.image[x + x_offset, y + y_offset]
                            # Check for a stuck particle (a==0)
                            if color.a == 0:
                                # This particle is now stuck. Set it to transparent red.
                                # (FIXED: a=0 to allow growth)
                                self.image[x, y] = rgba(255, 0, 0, 0)
                                walking = False
                                found = True
                                found_neighbor = True
                                break
                        except IndexError:
                            # This should not happen with the corrected bounds check, but good to have
                            walking = False
                            found = False
                            break
                    if found_neighbor:
                        break
        return found

    def finalize_image(self):
        """Make the DLA structure visible for saving."""
        # The aggregate is stored with a=0. We'll make it red and opaque.
        # The background (a=255) will be white.
        stuck_pixels_mask = self.image.pixels[:, :, 3] == 0
        # Set stuck pixels to red
        self.image.pixels[stuck_pixels_mask] = [255, 0, 0, 255]
        # Set background pixels to white
        self.image.pixels[~stuck_pixels_mask] = [255, 255, 255, 255]


def run_sim(width: int, height: int, num_steps: int, num_seeds: int = 100):
    dla = DLA(width, height)
    for _ in range(num_seeds):
        dla.random_seed()

    for i in range(num_steps):
        if dla.walk():
            print(f"Particle {i} found a spot.")

    # Finalize the image before saving
    dla.finalize_image()
    dla.save_image("dla.png")


if __name__ == "__main__":
    # Profile the function
    profiler = cProfile.Profile()
    profiler.enable()
    run_sim(400, 400, 55000, 10)  # Increased seeds for a better image
    profiler.disable()

    # Create stats object
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative").print_stats(10)
