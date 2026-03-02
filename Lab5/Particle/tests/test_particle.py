import pytest
from Particle import Particle
from Vec3 import Vec3


# Test 2
def test_particle_creation():
    particle = Particle()
    assert particle.position == Vec3()
    assert particle.direction == Vec3()
    assert particle.colour == Vec3()
    assert particle.max_life == 1000
    assert particle.life == 0
    assert pytest.approx(particle.scale) == 1.0


# Test 2
def test_particle_creation_with_args():
    position = Vec3(1, 2, 3)
    direction = Vec3(4, 5, 6)
    colour = Vec3(7, 8, 9)
    max_life = 100
    life = 50
    scale = 2.0
    particle = Particle(position, direction, colour, max_life, life, scale)
    assert particle.position == position
    assert particle.direction == direction
    assert particle.colour == colour
    assert particle.max_life == max_life
    assert particle.life == life
    assert pytest.approx(particle.scale) == scale
