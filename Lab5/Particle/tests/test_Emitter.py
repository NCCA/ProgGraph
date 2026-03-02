import pytest
from Emitter import Emitter
from Vec3 import Vec3


def test_emitter_creation():
    emitter = Emitter(Vec3(0, 0, 0), 10)
    assert emitter.position == Vec3(0, 0, 0)
    assert emitter.num_particles == 10


def test_setters():
    emitter = Emitter(Vec3(0, 0, 0), 10)
    emitter.position = Vec3(1, 2, 3)
    assert emitter.position == Vec3(1, 2, 3)
    emitter.num_particles = 20
    assert emitter.num_particles == 20
    with pytest.raises(TypeError):
        emitter.num_particles = -1
    with pytest.raises(TypeError):
        emitter.position = 1
