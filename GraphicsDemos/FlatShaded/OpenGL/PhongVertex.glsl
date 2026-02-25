#version 400 core
layout (location = 0) in vec3  inPosition;
layout (location = 1) in vec3  inNormal;
layout (location = 2) in vec2  inUV;

uniform mat4 M;
uniform mat4 MVP;

out vec3 fragPos;
out vec3 normal;
out vec2 uv;

void main()
{
    uv = inUV;
    // Transform vertex position to world space
    fragPos = vec3(M * vec4(inPosition, 1.0));
    // Transform normal to world space (using the normal matrix)
    normal = mat3(transpose(inverse(M))) * inNormal;
    // Final position for clipping and rasterization
    gl_Position = MVP * vec4(inPosition, 1.0);
}
