import random

from Particle import Particle
from Random import Random
from Vec3 import Vec3

GRAVITY = Vec3(0, -9.81, 0)


class Emitter:
    def __init__(self, position: Vec3, num_particles: int):
        self._position = position
        self._num_particles = num_particles
        self._particles = []
        self._init_particles()

    def _init_particles(self):
        for _ in range(self.num_particles):
            particle = self._create_particle()
            self._particles.append(particle)

    def _create_particle(self):
        EMIT_DIR = Vec3(0, 1, 0)
        SPREAD = 15
        direction = (
            EMIT_DIR * Random.random_positive_float()
            + Random.random_vector_on_sphere() * SPREAD
        )
        direction.y = abs(direction.y)
        print(direction)
        max_life = random.randint(10, 50)
        colour = Random.random_positive_vec3()
        # note I need a unique vec3 here other wise position is basically shared
        pos = Vec3(self._position.x, self._position.y, self._position.z)
        particle = Particle(pos, direction, colour, max_life, 0, 1.0)
        return particle

    def render(self):
        for particle in self._particles:
            print(particle)

    def update(self, dt):
        for i, particle in enumerate(self._particles):
            particle.direction += GRAVITY * dt * 0.5
            particle.position += particle.direction * dt
            particle.life += 1
            if particle.life > particle.max_life:
                self._particles[i] = self._create_particle()

    def write_geo(self, filename):
        with open(filename, "w") as file:
            #  write header see here https://www.sidefx.com/docs/houdini/io/formats/geo.html
            file.write("PGEOMETRY V5\n")
            file.write(f"NPoints {len(self._particles)} NPrims 1\n")
            file.write("NPointGroups 0 NPrimGroups 0\n")
            file.write("NPointAttrib 2  NVertexAttrib 0 NPrimAttrib 1 NAttrib 0\n")
            file.write("PointAttrib \n")
            file.write("Cd 3 float 1 1 1\n")
            file.write("psca    1  float 1 \n")
            for particle in self.particles:
                file.write(
                    f"{particle.position.x} {particle.position.y} {particle.position.z} 1 "
                )
                file.write(
                    f"( {particle.colour.x} {particle.colour.y} {particle.colour.z} {particle.scale}) \n"
                )

            file.write("PrimAttrib \n")
            file.write("generator   1  index 1 papi \n")
            file.write(f"Part {len(self._particles)} ")
            index_values = [i for i in range(len(self._particles))]
            file.write(" ".join(map(str, index_values)))
            file.write(" [0] \n")
            file.write("beginExtra \n")
            file.write("endExtra\n")

    @property
    def particles(self):
        return self._particles

    @property
    def num_particles(self):
        return self._num_particles

    @num_particles.setter
    def num_particles(self, value):
        if not isinstance(value, int) or value < 0:
            raise TypeError("num_particles must be a positive int")
        self._num_particles = value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        if not isinstance(value, Vec3):
            raise TypeError("position must be a Vec3")
        self._position = value
