#!/usr/bin/env -S uv run --script
import matplotlib.pyplot as plt
from Emitter import Emitter
from Vec3 import Vec3


def plot_emitter(emitter):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for _ in range(20):  # run for 200 frames
        emitter.update(0.5)
        # extract positions and colours
        positions = [p.position for p in emitter.particles]
        colours = [p.colour.to_tuple() for p in emitter.particles]

        # ax.clear()
        if positions:
            x_coords = [p.x for p in positions]
            y_coords_sim = [p.y for p in positions]
            z_coords_sim = [p.z for p in positions]
            ax.scatter(x_coords, z_coords_sim, y_coords_sim, c=colours)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-20, 20])
    plt.title("Particle System")

    plt.show()


def main():
    emitter = Emitter(Vec3(0, 0, 0), 500)
    # plot_emitter(emitter)

    for i in range(200):
        emitter.write_geo(f"geo/frame_{i:04}.geo")
        emitter.update(0.5)


if __name__ == "__main__":
    main()
