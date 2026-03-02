from Random import Random


def test_positive_float():
    for _ in range(1000):
        v = Random.random_positive_float()
        assert v >= 0 and v <= 1


def test_positive_float_range():
    for _ in range(1000):
        v = Random.random_positive_float(20)
        assert v >= 0 and v <= 20.0


def test_random_vec3():
    for _ in range(1000):
        v = Random.random_vec3()
        assert v.x >= -1.0 and v.x <= 1
        assert v.y >= -1.0 and v.y <= 1
        assert v.z >= -1.0 and v.z <= 1


def test_random_pos_vec3():
    for _ in range(1000):
        v = Random.random_positive_vec3()
        assert v.x >= 0 and v.x <= 1
        assert v.y >= 0 and v.y <= 1
        assert v.z >= 0 and v.z <= 1
