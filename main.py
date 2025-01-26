import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from PIL import Image
import ctypes
import tifffile as tiff  # New import for tifffile

# Function to generate a gradient image with customizable size using tiling
def create_gradient_image(tile_size=4096, tiles_x=4, tiles_y=4, output_file="complex_gradient_image.tiff"):

    # Initialize GLFW
    if not glfw.init():
        raise Exception("GLFW could not be initialized!")

    try:
        # Set GLFW window hints for OpenGL context
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # Make the window invisible
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        # Create an off-screen OpenGL context (temporary window)
        window = glfw.create_window(1, 1, "Off-screen", None, None)
        if not window:
            glfw.terminate()
            raise Exception("GLFW window could not be created!")

        # Make the window's context current
        glfw.make_context_current(window)

        # Query maximum texture size after making context current
        max_texture_size = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
        print(f"Maximum texture size supported: {max_texture_size}x{max_texture_size}")
        if tile_size > max_texture_size:
            raise Exception(f"Desired tile size {tile_size} is larger than the maximum texture size {max_texture_size}!")
            
        # Set tile size to the maximum texture size
        # tile_size = 4096 # max_texture_size
        print(f"Using tile size: {tile_size}x{tile_size}")

        # Validate that the desired image size is compatible with tiling
        expected_width = tiles_x * tile_size
        expected_height = tiles_y * tile_size
        if tile_size != expected_width or tile_size != expected_height:
            print(f"Adjusting image size to {expected_width}x{expected_height} to fit {tiles_x}x{tiles_y} tiles.")
            width = expected_width
            height = expected_height

        print(f"Generating a {tiles_x}x{tiles_y} tiled image with each tile sized {tile_size}x{tile_size}.")

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

        # Prepare the final image as a NumPy array to use with tifffile
        final_image_array = np.zeros((height, width, 3), dtype=np.uint8)

        for tile_x in range(tiles_x):
            for tile_y in range(tiles_y):
                # Current tile position
                pos_x = tile_x * tile_size
                pos_y = tile_y * tile_size
                print(f"Rendering tile ({tile_x + 1}, {tile_y + 1}) at position ({pos_x}, {pos_y}) with size {tile_size}x{tile_size}")

                # Bind the framebuffer
                glBindFramebuffer(GL_FRAMEBUFFER, FBO)

                # Create a texture for the current tile
                texture = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tile_size, tile_size, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

                # Attach texture to framebuffer
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

                # Check if the framebuffer is complete
                if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                    glDeleteTextures(1, [texture])
                    raise Exception("Framebuffer is not complete!")

                # Set the viewport to the size of the current tile
                glViewport(0, 0, tile_size, tile_size)

                # Use the shader program
                glUseProgram(shader)

                # Render to the framebuffer
                glClearColor(0.0, 0.0, 0.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT)
                glBindVertexArray(VAO)
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

                # Read pixels from the framebuffer
                glPixelStorei(GL_PACK_ALIGNMENT, 1)
                pixels = glReadPixels(0, 0, tile_size, tile_size, GL_RGB, GL_UNSIGNED_BYTE)

                # Convert to NumPy array and flip vertically
                tile_image = np.frombuffer(pixels, dtype=np.uint8).reshape((tile_size, tile_size, 3))
                tile_image = np.flipud(tile_image)  # OpenGL's origin is bottom-left

                # Paste the tile into the final image array
                final_image_array[pos_y:pos_y+tile_size, pos_x:pos_x+tile_size, :] = tile_image

                # Cleanup textures
                glDeleteTextures(1, [texture])

        # Save the final image using tifffile
        try:
            tiff.imwrite(output_file, final_image_array, compression='none', photometric='rgb', bigtiff=True)
            print(f"Complex gradient image created successfully! Saved as '{output_file}'.")
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except TypeError as te:
            print(f"TypeError: {te}")
        except struct.error as se:
            print(f"StructError: {se}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        # Cleanup
        glDeleteFramebuffers(1, [FBO])
        glDeleteBuffers(1, [VBO, EBO])
        glDeleteVertexArrays(1, [VAO])
        glDeleteProgram(shader)

    finally:
        glfw.terminate()

# Call the function with 4x4 tiling
create_gradient_image(tile_size=4096, tiles_x=4, tiles_y=4, output_file="custom_gradient_image.tiff")