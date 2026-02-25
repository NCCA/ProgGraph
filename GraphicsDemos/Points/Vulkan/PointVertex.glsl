#version 450 core
layout (location = 0) in vec3  inPosition;
layout (location = 1) in vec3 inColour;

layout(binding = 0) uniform MVP
{
    mat4 mvp;
};

layout (location = 1) out vec3 vertColour;

void main()
{
  gl_Position = mvp * vec4(inPosition, 1.0);
  vertColour = inColour;
}
