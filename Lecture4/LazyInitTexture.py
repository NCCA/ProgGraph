#!/usr/bin/env -S uv run --script
class Texture:
    """
    This class demonstrates the Lazy Initialization pattern.
    Instead of a private constructor and a static factory method like in C++,
    we use a class method `get_texture` to control instance creation.
    """

    # This class-level dictionary serves the same purpose as the C++ static map.
    # It caches the texture instances that have been created.
    _textures = {}

    def __init__(self, name):
        """
        The constructor is "private" by convention. It should only be called
        from within the class method `get_texture`.
        In a real-world scenario, this is where you would load the texture
        data from a file.
        """
        print(f"    -> Loading texture data for: {name}")
        self.name = name

    @classmethod
    def get_texture(cls, name: str):
        """
        This class method is the factory for Texture objects.
        It checks if a texture has already been instantiated. If so, it
        returns the existing object; otherwise, it creates a new one.
        """
        print(f"Requesting texture: '{name}'")
        # Check if the instance already exists in our cache.
        if name not in cls._textures:
            # Lazy Initialization: The object is created only when it's needed.
            print(f"    -> '{name}' not in cache. Creating new instance.")
            cls._textures[name] = cls(name)
        else:
            print(f"    -> '{name}' found in cache. Re-using existing instance.")

        # Return the cached instance.
        return cls._textures[name]

    @classmethod
    def print_current_textures(cls):
        """
        A helper method to display the current state of the texture cache.
        """
        print("\n--- Texture Cache State ---")
        print(f"Number of instances created = {len(cls._textures)}")
        for name in cls._textures:
            print(f"- {name}")
        print("---------------------------\n")


def level_load():
    """A function to demonstrate texture requests from another scope."""
    print("--- Inside foo() ---")
    Texture.get_texture("new.tga")
    Texture.print_current_textures()
    print("--- Exiting foo() ---\n")


if __name__ == "__main__":
    print("Starting texture requests...")

    Texture.get_texture("diffuse.tga")
    Texture.print_current_textures()

    Texture.get_texture("specular.tga")
    Texture.print_current_textures()

    # Requesting "diffuse.tga" again should not create a new instance.
    Texture.get_texture("diffuse.tga")
    Texture.print_current_textures()

    # Call another function that requests textures.
    level_load()

    print("Final state of the texture cache:")
    Texture.print_current_textures()
