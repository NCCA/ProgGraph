#version 400 core
layout (location = 0) out vec4 fragColour;
in vec2 uv;
void main()
{
  fragColour.rg=uv;
}
