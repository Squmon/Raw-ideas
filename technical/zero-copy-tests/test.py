import sys
import numpy as np
from OpenGL.GL import *
import glfw
import jax


x = jax.numpy.ones(10)
# Vertex Shader Source
VERTEX_SHADER_SOURCE = """
#version 330 core
layout (location = 0) in vec2 aPos;

void main()
{
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
}
"""

# Fragment Shader Source
FRAGMENT_SHADER_SOURCE = """
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0, 0.5, 0.2, 1.0);
}
"""

def compile_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        info_log = glGetShaderInfoLog(shader).decode('utf-8')
        raise RuntimeError(f"Shader compilation failed:\n{info_log}")
    return shader

def create_shader_program(vertex_source, fragment_source):
    vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_source)
    fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_source)
    
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    
    if not glGetProgramiv(program, GL_LINK_STATUS):
        info_log = glGetProgramInfoLog(program).decode('utf-8')
        raise RuntimeError(f"Program linking failed:\n{info_log}")
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

def main():
    # Initialize GLFW
    if not glfw.init():
        sys.exit("Failed to initialize GLFW")
        
    # Configure OpenGL 3.3 Core Profile
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    # Uncomment the line below if you are on macOS:
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    # Create window
    window = glfw.create_window(800, 600, "Hello Triangle - PyOpenGL", None, None)
    if not window:
        glfw.terminate()
        sys.exit("Failed to create GLFW window")
        
    glfw.make_context_current(window)
    
    # Triangle vertex data
    vertices = np.array([
         0.0,  0.5,  # Top vertex
        -0.5, -0.5,  # Bottom-left vertex
         0.5, -0.5   # Bottom-right vertex
    ], dtype=np.float32)
    
    # Set up VAO and VBO
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    
    glBindVertexArray(VAO)
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    
    # Unbind buffers
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    
    # Build and compile shader program
    shader_program = create_shader_program(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)
    
    # Render loop
    while not glfw.window_should_close(window):
        glfw.poll_events()
        
        # Clear screen with a background color
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Draw the triangle
        glUseProgram(shader_program)
        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        
        glfw.swap_buffers(window)
        
    # Cleanup resources
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteProgram(shader_program)
    
    glfw.terminate()

if __name__ == "__main__":
    main()