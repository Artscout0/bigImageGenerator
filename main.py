import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from PIL import Image

# Function to generate a gradient image with customizable size
def create_gradient_image(width=30000, height=30000, output_file="complex_gradient_image.tiff"):
    # Initialize GLFW
    if not glfw.init():
        raise Exception("GLFW could not be initialized!")

    # Set GLFW window hints for OpenGL context
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # Make the window invisible
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # Create an off-screen OpenGL context with the given framebuffer size
    window = glfw.create_window(width, height, "Off-screen", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window could not be created!")

    # Make the window's context current
    glfw.make_context_current(window)

    # GLSL fragment shader to create a complex gradient
    fragment_shader_source = """
    #version 410 core
    out vec4 FragColor;
    in vec2 TexCoords;

    void main()
    {
        float x = TexCoords.x;
        float y = TexCoords.y;

        float r = 0.5 * (1.0 + sin(20.0 * 3.141592 * x + 5.0 * y));
        float g = 0.5 * (1.0 + sin(20.0 * 3.141592 * y + 5.0 * x));
        float b = 0.5 * (1.0 + sin(40.0 * 3.141592 * (x * y) + 10.0 * x));

        r += 0.1 * (1.0 + sin(50.0 * 3.141592 * y));
        g += 0.1 * (1.0 + sin(50.0 * 3.141592 * x));
        b += 0.1 * (1.0 + sin(80.0 * 3.141592 * (x + y)));

        FragColor = vec4(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0), 1.0);
    }
    """

    # Simple vertex shader
    vertex_shader_source = """
    #version 410 core
    layout(location = 0) in vec2 aPos;
    out vec2 TexCoords;

    void main()
    {
        TexCoords = (aPos + 1.0) / 2.0;
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
    """

    # Compile shaders and link them into a program
    shader = compileProgram(
        compileShader(vertex_shader_source, GL_VERTEX_SHADER),
        compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    )

    # Vertex data for a fullscreen quad
    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
         1.0,  1.0,
        -1.0,  1.0
    ], dtype=np.float32)

    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    # Create a VAO and VBO for vertex data
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # Unbind VAO
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # Create a framebuffer for off-screen rendering
    FBO = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, FBO)

    # Create a texture to store the framebuffer color output
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

    # Check if the framebuffer is complete
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise Exception("Framebuffer is not complete!")

    # Bind the shader program and set the viewport
    glUseProgram(shader)
    glViewport(0, 0, width, height)

    # Render to the framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, FBO)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glBindVertexArray(VAO)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    # Read pixels from the framebuffer
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_SHORT)
    image = np.frombuffer(pixels, dtype=np.uint16).reshape((height, width, 3))

    # Save as a high bit-depth TIFF
    img = Image.fromarray(image, mode='RGB')
    img.save(output_file)

    # Cleanup
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glDeleteFramebuffers(1, [FBO])
    glDeleteTextures([texture])
    glDeleteBuffers(1, [VBO, EBO])
    glDeleteVertexArrays(1, [VAO])
    glfw.terminate()

    print(f"Complex gradient image created successfully! Saved as '{output_file}'.")

# Call the function with customizable width and height
create_gradient_image(width=40_000, height=40_000, output_file="custom_gradient_image.tiff")