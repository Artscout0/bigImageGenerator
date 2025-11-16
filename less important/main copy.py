import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from PIL import Image
import ctypes
import tifffile as tiff
import os
import time

# Hint to the NVIDIA driver to use the high-performance GPU
# This must be done before any OpenGL/GLFW initialization.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.environ["PYOPENGL_PLATFORM"] = "glfw"
os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'


# Function to generate a gradient image with customizable size using tiling
def create_gradient_image(tile_size=4096, tiles_x=4, tiles_y=4, output_file="complex_gradient_image.tiff"):

    # --- NEW: Start total process timer ---
    total_start_time = time.perf_counter()

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
        max_texture_size_raw = glGetIntegerv(GL_MAX_TEXTURE_SIZE)
        # Normalize to a plain int (PyOpenGL may return an int or an array-like)
        try:
            max_texture_size = int(max_texture_size_raw)
        except Exception:
            try:
                max_texture_size = int(max_texture_size_raw[0])
            except Exception:
                max_texture_size = None
        if max_texture_size is None:
            print(f"Maximum texture size query returned (raw): {max_texture_size_raw!r}")
        else:
            print(f"Maximum texture size supported: {max_texture_size}x{max_texture_size}")
        if max_texture_size is not None and tile_size > max_texture_size:
            raise Exception(f"Desired tile size {tile_size} is larger than the maximum texture size {max_texture_size}!")
        elif max_texture_size is None and tile_size is not None:
           # if we couldn't determine the limit, warn instead of crashing
           print(f"Warning: unable to determine GL_MAX_TEXTURE_SIZE; proceeding but tile_size={tile_size} may be invalid.")

        print(f"Using tile size: {tile_size}x{tile_size}")

        # Validate that the desired image size is compatible with tiling
        expected_width = tiles_x * tile_size
        expected_height = tiles_y * tile_size
        width = expected_width
        height = expected_height
        print(f"Adjusting image size to {width}x{height} to fit {tiles_x}x{tiles_y} tiles.")

        print(f"Generating a {tiles_x}x{tiles_y} tiled image with each tile sized {tile_size}x{tile_size}.")

        # GLSL fragment shader to create a complex gradient
        fragment_shader_source = """
        #version 410 core
        out vec4 FragColor;
        in vec2 TexCoords; // This is the local UV (0.0 to 1.0) for the current tile

        uniform vec2 u_tile_offset; // (current_tile_x, current_tile_y)
        uniform vec2 u_total_tiles; // (total_tiles_x, total_tiles_y)

        void main()
        {
            // Calculate global UVs (from 0.0 to 1.0) across the *entire* image
            vec2 global_uv = (TexCoords + u_tile_offset) / u_total_tiles;
            
            // Use global_uv.x and global_uv.y instead of TexCoords
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

        # Get uniform locations
        glUseProgram(shader)
        u_tile_offset_loc = glGetUniformLocation(shader, "u_tile_offset")
        u_total_tiles_loc = glGetUniformLocation(shader, "u_total_tiles")
        glUniform2f(u_total_tiles_loc, float(tiles_x), float(tiles_y)) # Pass as floats
        
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

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tile_size, tile_size, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Framebuffer is not complete!")
        
        # Set viewport once as it's constant for all tiles
        glViewport(0, 0, tile_size, tile_size)
        glBindFramebuffer(GL_FRAMEBUFFER, 0) # Unbind FBO for now

        # Prepare the final image as a NumPy array to use with tifffile
        #final_image_array = np.zeros((height, width, 3), dtype=np.uint8)

        # Calculate required bytes and warn
        bytes_needed = int(height) * int(width) * 3
        gib = bytes_needed / (1024**3)
        print(f"Final image will require approximately {gib:.1f} GiB on disk. Ensure you have that free.")
        memmap_path = os.path.join(os.path.dirname(output_file), os.path.basename(output_file) + ".memmap")
        # Create on-disk memmap (this allocates the file on disk, not RAM)
        final_image_array = np.memmap(memmap_path, dtype=np.uint8, mode='w+', shape=(height, width, 3))
        # --- NEW: Initialize total render timer ---
        total_render_time_s = 0.0

        for tile_x in range(tiles_x):
            for tile_y in range(tiles_y):
                # Current tile position
                pos_x = tile_x * tile_size
                pos_y = tile_y * tile_size
                
                # --- NEW: Modify print to show progress on one line ---
                print(f"Rendering tile ({tile_x + 1}, {tile_y + 1})...", end="", flush=True)

                # --- NEW: Start single tile timer ---
                tile_start_time = time.perf_counter()

                # Bind the framebuffer for rendering
                glBindFramebuffer(GL_FRAMEBUFFER, FBO)

                # Use the shader program and set uniforms for the current tile
                glUseProgram(shader)
                glUniform2f(u_tile_offset_loc, float(tile_x), float(tile_y)) # Pass as floats

                # Render to the framebuffer
                glClearColor(0.0, 0.0, 0.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT)
                glBindVertexArray(VAO)
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

                # Read pixels from the framebuffer
                glPixelStorei(GL_PACK_ALIGNMENT, 1)
                pixels = glReadPixels(0, 0, tile_size, tile_size, GL_RGB, GL_UNSIGNED_BYTE)

                # Unbind FBO after reading
                glBindFramebuffer(GL_FRAMEBUFFER, 0)

                # Convert to NumPy array and flip vertically
                tile_image = np.frombuffer(pixels, dtype=np.uint8).reshape((tile_size, tile_size, 3))
                tile_image = np.flipud(tile_image)  # OpenGL's origin is bottom-left

                # Paste the tile into the final image array
                # Note: NumPy slicing is (row, column) which is (y, x)
                final_image_array[pos_y:pos_y+tile_size, pos_x:pos_x+tile_size, :] = tile_image

                # --- NEW: End single tile timer and report ---
                tile_end_time = time.perf_counter()
                tile_duration_ms = (tile_end_time - tile_start_time) * 1000
                total_render_time_s += (tile_end_time - tile_start_time)
                
                print(f" Done. Took {tile_duration_ms:.2f} ms")


        # --- NEW: Report total render time ---
        print(f"\n--- All tiles rendered. Total render time: {total_render_time_s:.2f} seconds ---\n")


         # Save the final image using tifffile
        # try:
        #     print("Saving final image... This may take a moment.")
          
        #     # --- NEW: Start save timer ---
        #     save_start_time = time.perf_counter()
          
        #     tiff.imwrite(output_file, final_image_array, compression='none', photometric='rgb', bigtiff=True)
          
        #     # --- NEW: End save timer and report ---
        #     save_end_time = time.perf_counter()
        #     save_duration_s = save_end_time - save_start_time
          
        #     print(f"Image saved successfully as '{output_file}'.")
        #     print(f"--- Save time: {save_duration_s:.2f} seconds ---")       
        # except ValueError as ve:
        #     print(f"ValueError: {ve}")
        # except TypeError as te:
        #     print(f"TypeError: {te}")
        # except Exception as e:
        #     print(f"An unexpected error occurred during saving: {e}")
        # Flush memmap to disk then write TIFF from the memmap (BigTIFF)
        try:
            final_image_array.flush()
            print("Saving final image to TIFF (BigTIFF)... This may take a while.")
            save_start_time = time.perf_counter()
            tiff.imwrite(output_file, final_image_array, photometric='rgb', compression='none', bigtiff=True)
            save_end_time = time.perf_counter()
            print(f"Image saved successfully as '{output_file}' in {save_end_time - save_start_time:.2f} s")
        except Exception as e:
            print(f"Failed to save the image: {e}")
        finally:
            # remove memmap backing file if you don't need it
            try:
                final_image_array._mmap.close()
            except Exception:
                pass
            # os.remove(memmap_path)  # uncomment to delete the memmap file when done
        # Cleanup
        glDeleteProgram(shader)
        glDeleteTextures(1, [texture])
        glDeleteFramebuffers(1, [FBO])
        glDeleteBuffers(1, [VBO, EBO])
        glDeleteVertexArrays(1, [VAO])

        # --- NEW: Report total process time ---
        total_end_time = time.perf_counter()
        total_duration_s = total_end_time - total_start_time
        print(f"\n--- Total process time: {total_duration_s:.2f} seconds ---\n")

    finally:
        glfw.terminate()

# Call the function with 4x4 tiling
create_gradient_image(tile_size=4096, tiles_x=4, tiles_y=4, output_file="custom_gradient_image.tiff")