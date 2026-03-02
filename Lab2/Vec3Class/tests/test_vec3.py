from vec3 import Vec3


def test_create_vec3():
    v = Vec3(1, 2, 3)
    assert v.x == 1
    assert v.y == 2
    assert v.z == 3


def test_equality():
    assert Vec3(1, 2, 3) == Vec3(1, 2, 3)
    assert Vec3(1, 2, 3) != Vec3(3, 2, 1)


def test_addition():
    v1 = Vec3(1, 2, 3)
    v2 = Vec3(4, 5, 6)
    assert v1 + v2 == Vec3(5, 7, 9)


def test_subtract():
    v1 = Vec3(4, 5, 6)
    v2 = Vec3(1, 2, 3)
    assert v1 - v2 == Vec3(3, 3, 3)


def test_magnitude():
    v = Vec3(3, 4, 0)
    assert v.magnitude() == 5


def test_scalar_multiplication():
    v = Vec3(1, 2, 3)
    assert v * 2 == Vec3(2, 4, 6)
    assert 3 * v == Vec3(3, 6, 9)


def test_dot_product():
    v1 = Vec3(1, 2, 3)
    v2 = Vec3(4, 5, 6)
    assert v1.dot(v2) == 1 * 4 + 2 * 5 + 3 * 6  # 32
    assert v2.dot(v1) == 32


def test_cross_product():
    v1 = Vec3(1, 2, 3)
    v2 = Vec3(4, 5, 6)
    assert v1.cross(v2) == Vec3(-3, 6, -3)
    assert v2.cross(v1) == Vec3(3, -6, 3)
