from dataclasses import dataclass, field

from Vec3 import Vec3


@dataclass
class Particle:
    position: Vec3 = field(default_factory=Vec3)
    direction: Vec3 = field(default_factory=Vec3)
    colour: Vec3 = field(default_factory=Vec3)
    max_life: int = 1000
    life: int = 0
    scale: float = 1.0

    def __str__(self):
        return f"Particle(position={self.position}, direction={self.direction}, colour={self.colour}, max_life={self.max_life}, life={self.life}, scale={self.scale})"
