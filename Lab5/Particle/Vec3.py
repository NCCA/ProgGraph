import math


class Vec3:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self._x = x
        self._y = y
        self._z = z

    def __str__(self):
        return f"Vec3({self._x}, {self._y}, {self._z})"

    def __repr__(self):
        return f"Vec3({self._x}, {self._y}, {self._z})"

    def __eq__(self, other):
        if not isinstance(other, Vec3):
            return NotImplemented
        return (
            math.isclose(self._x, other._x)
            and math.isclose(self._y, other._y)
            and math.isclose(self._z, other._z)
        )

    def to_tuple(self):
        return (self._x, self._y, self._z)

    def to_list(self):
        return [self._x, self._y, self._z]

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("x must be a number")
        self._x = float(value)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("x must be a number")
        self._y = float(value)

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError("x must be a number")
        self._z = float(value)

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self._x * other._x, self._y * other._y, self._z * other._z)
        elif isinstance(other, (int, float)):
            return Vec3(self._x * other, self._y * other, self._z * other)
        else:
            raise TypeError(
                "unsupported operand type(s) for *: 'Vec3' and '{}'".format(
                    type(other).__name__
                )
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __iadd__(self, other):
        if isinstance(other, Vec3):
            self._x += other._x
            self._y += other._y
            self._z += other._z
        elif isinstance(other, (int, float)):
            self._x += other
            self._y += other
            self._z += other
        else:
            raise TypeError(
                "unsupported operand type(s) for +=: 'Vec3' and '{}'".format(
                    type(other).__name__
                )
            )
        return self

    def __add__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self._x + other._x, self._y + other._y, self._z + other._z)
        elif isinstance(other, (int, float)):
            return Vec3(self._x + other, self._y + other, self._z + other)
        else:
            raise TypeError(
                "unsupported operand type(s) for +: 'Vec3' and '{}'".format(
                    type(other).__name__
                )
            )
