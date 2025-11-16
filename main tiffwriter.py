import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import ctypes
import tifffile as tiff
import os
import time


# --- GPU Performance Hints ---
# Hint to the NVIDIA driver to use the high-performance GPU (for NVIDIA systems)
os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'

# Hint to the AMD driver to use the high-performance GPU (for AMD systems)
os.environ['AMD_POWER_EXPRESS_REQUEST_HIGH_PERFORMANCE'] = '1'

# General compatibility settings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYOPENGL_PLATFORM"] = "glfw"


def create_gradient_image(tile_size=4096, tiles_x=4, tiles_y=4, output_file="complex_gradient_image.tiff"):
    """
    Generates a large tiled image using OpenGL for rendering and streams the output
    directly to a BigTIFF file to handle arbitrary image sizes without high RAM usage.
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
        max_texture_size_raw = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
        try:
            max_texture_size = int(max_texture_size_raw)
        except (TypeError, IndexError):
            try:
                max_texture_size = int(max_texture_size_raw[0])
            except (TypeError, IndexError):
                max_texture_size = None

        if max_texture_size is None:
            print(f"Warning: Could not determine GL_MAX_TEXTURE_SIZE. Raw value: {max_texture_size_raw!r}")
        else:
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

        # --- Main Render and Write Loop ---
        total_render_time_s = 0.0
        # Create an on-disk memory-map to hold the full image without using RAM
        memmap_path = output_file + ".memmap"
        final_image = np.memmap(memmap_path, dtype=np.uint8, mode='w+', shape=(height, width, 3))

        # Iterate row-major: top-to-bottom (tile_y), then left-to-right (tile_x)
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                print(f"Rendering tile ({tile_y + 1}, {tile_x + 1}) ({tile_y * tiles_x + tile_x + 1}/{tiles_x * tiles_y})...", end="", flush=True)
                tile_start = time.perf_counter()

                # Bind FBO and set viewport for the current tile
                glBindFramebuffer(GL_FRAMEBUFFER, FBO)
                glViewport(0, 0, tile_size, tile_size)

                # Render the tile
                glUseProgram(shader)
                glUniform2f(u_tile_offset_loc, float(tile_x), float(tile_y))
                glClear(GL_COLOR_BUFFER_BIT)
                glBindVertexArray(VAO)
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

                # Read the rendered pixels from the FBO
                glPixelStorei(GL_PACK_ALIGNMENT, 1)
                pixels = glReadPixels(0, 0, tile_size, tile_size, GL_RGB, GL_UNSIGNED_BYTE)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)

                # Reshape buffer into a NumPy array and flip it vertically
                tile_image = np.frombuffer(pixels, dtype=np.uint8).reshape((tile_size, tile_size, 3))
                tile_image = np.flipud(tile_image)

                # Write the rendered tile into the correct position in the on-disk memmap
                pos_x = tile_x * tile_size
                pos_y = tile_y * tile_size
                final_image[pos_y:pos_y + tile_size, pos_x:pos_x + tile_size, :] = tile_image

                tile_end = time.perf_counter()
                tile_duration_ms = (tile_end - tile_start) * 1000
                total_render_time_s += (tile_end - tile_start)
                print(f" Done. ({tile_duration_ms:.1f} ms)")
        
        # Flush the memory map to ensure all tiles are written to disk
        final_image.flush()

        print(f"\n--- All tiles rendered. Total render time: {total_render_time_s:.2f} seconds ---\n")

        # Use tifffile.imwrite to save the memory-mapped file as a BigTIFF.
        # This reads from the memmap file and writes to the TIFF file efficiently.
        print("Saving final image from memmap to BigTIFF... This may take a moment.\n")
        save_start_time = time.perf_counter()
        tiff.imwrite(
            output_file,
            final_image,
            bigtiff=True,
            photometric='rgb',
            compression='none' # Fastest. Use 'lzw' or 'deflate' for smaller files.
        )
        save_end_time = time.perf_counter()

        print(f"--- Image saved successfully to '{output_file}' ---")
        print(f"Save time: {save_end_time - save_start_time:.2f} seconds\n")
        total_end_time = time.perf_counter()
        print(f"Total process time: {total_end_time - total_start_time:.2f} seconds\n")

    finally:
        # Cleanup all resources
        if 'final_image' in locals() and isinstance(final_image, np.memmap):
            # Close the memmap and delete the temporary file
            try:
                final_image._mmap.close()
                os.remove(memmap_path)
                print(f"Temporary memmap file '{memmap_path}' removed.\n")
            except Exception as e:
                print(f"Warning: Could not remove memmap file '{memmap_path}': {e}\n")

        if 'shader' in locals(): glDeleteProgram(shader)
        if 'texture' in locals(): glDeleteTextures(1, [texture])
        if 'FBO' in locals(): glDeleteFramebuffers(1, [FBO])
        if 'VBO' in locals(): glDeleteBuffers(2, [VBO, EBO])
        if 'VAO' in locals(): glDeleteVertexArrays(1, [VAO])
        glfw.terminate()


# Call the function (example)
if __name__ == "__main__":
    # adjust tile_size / tiles_x / tiles_y to match your GPU max texture size and desired grid
    create_gradient_image(tile_size=4096, tiles_x=4, tiles_y=4, output_file="custom_gradient_image.tiff")