import hou
from Emitter import Emitter
from Vec3 import Vec3


def write_particles(emitter, filename):
    geo = hou.Geometry()
    geo.addAttrib(hou.attribType.Point, "Cd", hou.Vector3(0, 0, 0))
    geo.addAttrib(hou.attribType.Point, "pscale", 1.0)

    point_objects = []
    for particle in emitter.particles:
        p = geo.createPoint()
        pos = particle.position
        p.setPosition(hou.Vector3(pos.x, pos.y, pos.z))
        colour = particle.colour
        p.setAttribValue("Cd", hou.Vector3(colour.x, colour.y, colour.z))
        p.setAttribValue("pscale", particle.scale)
        point_objects.append(p)
    geo.saveToFile(filename)


def main():
    emitter = Emitter(Vec3(0, 0, 0), 500)
    for frame in range(100):
        write_particles(emitter, f"particles_{frame:03}.geo")
        emitter.update(0.1)


if __name__ == "__main__":
    main()
