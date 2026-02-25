#version 400 core
layout (location = 0) in vec3  inPosition;
layout (location = 1) in vec3 inColour;
uniform mat4 MVP;
out vec3 vertColour;
void main()
{
  gl_Position = MVP*vec4(inPosition, 1.0);
  vertColour = inColour;
}
