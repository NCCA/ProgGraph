import math


class Vec3:
    """
    A simple 3D vector class supporting basic vector operations.
    """

    def __init__(self, x: float, y: float, z: float) -> None:
        """
        Initialize a Vec3 instance.

        Args:
            x (float): The x component.
            y (float): The y component.
            z (float): The z component.
        """
        self.x: float = x
        self.y: float = y
        self.z: float = z

    def __eq__(self, other: object) -> bool:
        """
        Check if two Vec3 instances are equal within floating point tolerance.

        Args:
            other (object): The object to compare.

        Returns:
            bool: True if equal, False otherwise.
        """
        if not isinstance(other, Vec3):
            return NotImplemented
        return (
            math.isclose(self.x, other.x)
            and math.isclose(self.y, other.y)
            and math.isclose(self.z, other.z)
        )

    def __add__(self, other: "Vec3") -> "Vec3":
        """
        Add two Vec3 vectors.

        Args:
            other (Vec3): The vector to add.

        Returns:
            Vec3: The result of vector addition.
        """
        if not isinstance(other, Vec3):
            return NotImplemented
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        """
        Subtract one Vec3 vector from another.

        Args:
            other (Vec3): The vector to subtract.

        Returns:
            Vec3: The result of vector subtraction.
        """
        if not isinstance(other, Vec3):
            return NotImplemented
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def magnitude(self) -> float:
        """
        Calculate the magnitude (Euclidean norm) of the vector.

        Returns:
            float: The magnitude of the vector.
        """
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __mul__(self, scalar: float) -> "Vec3":
        """
        Multiply the vector by a scalar.

        Args:
            scalar (float): The scalar value.

        Returns:
            Vec3: The result of scalar multiplication.
        """
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        """
        Multiply the vector by a scalar (right-hand side).

        Args:
            scalar (float): The scalar value.

        Returns:
            Vec3: The result of scalar multiplication.
        """
        return self.__mul__(scalar)

    def dot(self, other: "Vec3") -> float:
        """
        Compute the dot product of two vectors.

        Args:
            other (Vec3): The other vector.

        Returns:
            float: The dot product.
        """
        if not isinstance(other, Vec3):
            return NotImplemented
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        """
        Compute the cross product of two vectors.

        Args:
            other (Vec3): The other vector.

        Returns:
            Vec3: The cross product vector.
        """
        if not isinstance(other, Vec3):
            return NotImplemented
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
