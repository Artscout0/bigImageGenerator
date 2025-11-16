import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import ctypes
import tifffile as tiff
import os
import time

# --- GPU Performance Hints ---
os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['AMD_POWER_EXPRESS_REQUEST_HIGH_PERFORMANCE'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYOPENGL_PLATFORM"] = "glfw"

def create_gradient_image(tile_size=4096, tiles_x=4, tiles_y=4, output_file="complex_gradient_image.tiff"):
    """
    Generates a large tiled image using OpenGL and streams the output directly
    to a BigTIFF file for maximum performance and unlimited image size.
    """
    total_start_time = time.perf_counter()

    if not glfw.init():
        raise Exception("GLFW could not be initialized!")

    try:
        # Configure and create an invisible OpenGL window
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        window = glfw.create_window(1, 1, "Off-screen", None, None)
        if not window:
            glfw.terminate()
            raise Exception("GLFW window could not be created!")
        glfw.make_context_current(window)

        # Determine GPU's maximum texture size
        max_texture_size = int(glGetIntegerv(GL_MAX_TEXTURE_SIZE))
        print(f"Maximum texture size supported: {max_texture_size}x{max_texture_size}")
        if tile_size > max_texture_size:
            raise ValueError(f"Desired tile size {tile_size} exceeds GPU maximum of {max_texture_size}!")

        # Calculate final image dimensions
        width = tiles_x * tile_size
        height = tiles_y * tile_size
        print(f"Generating image: {width}x{height} ({tiles_x}x{tiles_y} tiles of {tile_size}x{tile_size})")
        bytes_needed = width * height * 3
        print(f"Estimated final file size (uncompressed): {bytes_needed / (1024**3):.2f} GiB")

        # GLSL Shaders
        fragment_shader_source = """
        #version 410 core
        out vec4 FragColor;
        in vec2 TexCoords;
        uniform vec2 u_tile_offset;
        uniform vec2 u_total_tiles;
        void main() {
            vec2 global_uv = (TexCoords + u_tile_offset) / u_total_tiles;
            float x = global_uv.x;
            float y = global_uv.y;
            float r = 0.5 * (1.0 + sin(20.0 * 3.141592 * x + 5.0 * y));
            float g = 0.5 * (1.0 + sin(20.0 * 3.141592 * y + 5.0 * x));
            float b = 0.5 * (1.0 + sin(40.0 * 3.141592 * (x * y) + 10.0 * x));
            r += 0.1 * (1.0 + sin(50.0 * 3.141592 * y));
            g += 0.1 * (1.0 + sin(50.0 * 3.141592 * x));
            b += 0.1 * (1.0 + sin(80.0 * 3.141592 * (x + y)));
            FragColor = vec4(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0), 1.0);
        }
        """
        vertex_shader_source = """
        #version 410 core
        layout(location = 0) in vec2 aPos;
        out vec2 TexCoords;
        void main() {
            TexCoords = (aPos + 1.0) / 2.0;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
        """
        shader = compileProgram(compileShader(vertex_shader_source, GL_VERTEX_SHADER), compileShader(fragment_shader_source, GL_FRAGMENT_SHADER))
        glUseProgram(shader)
        u_tile_offset_loc = glGetUniformLocation(shader, "u_tile_offset")
        u_total_tiles_loc = glGetUniformLocation(shader, "u_total_tiles")
        glUniform2f(u_total_tiles_loc, float(tiles_x), float(tiles_y))

        # Setup fullscreen quad geometry
        vertices = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        VAO = glGenVertexArrays(1)
        VBO, EBO = glGenBuffers(2)
        glBindVertexArray(VAO)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

        # Setup Framebuffer Object (FBO) for offscreen rendering
        FBO = glGenFramebuffers(1)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tile_size, tile_size, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindFramebuffer(GL_FRAMEBUFFER, FBO)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer is not complete!")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        # @TODO: Implement correct display like in the other files (line by line, ms per tile, etc)
        # --- Main Render and Write Logic ---
        # Define a generator function that will render and yield one tile at a time
        def tile_generator():
            nonlocal total_render_time_s
            for tile_y in range(tiles_y):
                for tile_x in range(tiles_x):
                    print(f"\rRendering Tile ({tile_y * tiles_x + tile_x + 1}/{tiles_x * tiles_y})...", end="", flush=True)
                    tile_start = time.perf_counter()

                    glBindFramebuffer(GL_FRAMEBUFFER, FBO)
                    glViewport(0, 0, tile_size, tile_size)
                    glUseProgram(shader)
                    glUniform2f(u_tile_offset_loc, float(tile_x), float(tile_y))
                    glClear(GL_COLOR_BUFFER_BIT)
                    glBindVertexArray(VAO)
                    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
                    
                    glPixelStorei(GL_PACK_ALIGNMENT, 1)
                    pixels = glReadPixels(0, 0, tile_size, tile_size, GL_RGB, GL_UNSIGNED_BYTE)
                    glBindFramebuffer(GL_FRAMEBUFFER, 0)

                    tile_image = np.frombuffer(pixels, dtype=np.uint8).reshape((tile_size, tile_size, 3))
                    
                    
                    total_render_time_s += (time.perf_counter() - tile_start)
                    
                    # Yield the flipped tile to the writer
                    yield np.flipud(tile_image)
            
            # After the loop, print a newline to move past the status line
            print("\nAll tiles rendered.")

        # Use tifffile.imwrite with the generator. This is the most optimal method.
        print("Starting image generation and streaming to TIFF...")
        total_render_time_s = 0
        
        tiff.imwrite(
            output_file,
            tile_generator(),
            bigtiff=True,
            shape=(height, width, 3),
            dtype=np.uint8,
            tile=(tile_size, tile_size),
            photometric='rgb',
            compression='none' # Fastest. Use 'lzw' or 'deflate' for smaller files.
        )

        print(f"\nTotal Render time: {total_render_time_s:.2f} seconds")
        print(f"\n--- Image saved successfully to '{output_file}' ---")
        total_end_time = time.perf_counter()
        print(f"Total process time: {total_end_time - total_start_time:.2f} seconds")

    finally:
        # Cleanup all resources
        if 'shader' in locals(): glDeleteProgram(shader)
        if 'texture' in locals(): glDeleteTextures(1, [texture])
        if 'FBO' in locals(): glDeleteFramebuffers(1, [FBO])
        if 'VBO' in locals(): glDeleteBuffers(2, [VBO, EBO])
        if 'VAO' in locals(): glDeleteVertexArrays(1, [VAO])
        glfw.terminate()

if __name__ == "__main__":
    create_gradient_image(
        tile_size=4096,
        tiles_x=4,
        tiles_y=4,
        output_file="custom_gradient_image.tiff"
    )