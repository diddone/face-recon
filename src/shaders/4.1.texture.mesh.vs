#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 outColor;

//uniform mat4 transformation;

void main()
{
	gl_Position = vec4(aPos, 1.0);
	gl_Position.z = 0.5;
	outColor = aColor;
}