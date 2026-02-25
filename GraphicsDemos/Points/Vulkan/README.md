# Vulkan Triangle

This should work ok on linux, on the mac you will need to install the sdk https://vulkan.lunarg.com/doc/sdk/1.4.328.1/mac/getting_started.html and set the following envars

```
export VK_ICD_FILENAMES=/usr/local/share/vulkan/icd.d/MoltenVK_icd.json
export VK_LAYER_PATH=/usr/local/share/vulkan/explicit_layer.d
```

There is also an issue with the vulkan bindings on mac where you need to explicitly set the libvulkan.dylib path to /usr/local/lib/libvulkan.dylib

This is done in the .venv/lib/python3.14/site-packages/vulkan/_vulkan.py

```
# Load SDK
_lib_names = ('libvulkan.so.1', 'vulkan-1.dll', '/usr/local/lib/libvulkan.dylib')
```

To compile the shaders you can use the following command:

```
glslangValidator -V shader.vert -o shader.vert.spv
glslangValidator -V shader.frag -o shader.frag.spv
```

The binary shaders are included in the project for ease.
