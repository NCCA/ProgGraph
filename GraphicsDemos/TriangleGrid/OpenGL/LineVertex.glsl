#version 400 core
layout (location = 0) in vec3  inPosition;
layout (location = 1) in vec3  inNormal;
layout (location = 2) in vec2  inUV;
uniform mat4 MVP;
out vec2 uv;
void main()
{
    uv = inUV;
    gl_Position = MVP*vec4(inPosition, 1.0);
}
