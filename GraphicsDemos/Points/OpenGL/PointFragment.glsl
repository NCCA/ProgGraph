#version 400 core
in vec3 vertColour;
layout (location = 0) out vec4 fragColour;

void main()
{
  fragColour.rgb=vertColour;
}
