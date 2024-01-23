#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 outColor;

uniform mat4 transformation;

void main()
{
	gl_Position = transformation * vec4(aPos, 1.0);
	gl_Position.y *= -1;

	gl_Position.x = gl_Position.x / gl_Position.z;
	gl_Position.y = gl_Position.y / gl_Position.z;

	// gl_Position.z = 0.50001;
	outColor = aColor;
}