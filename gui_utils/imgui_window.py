# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import imgui
import imgui.integrations.glfw
import imgui.integrations.opengl

from . import glfw_window
from . import imgui_utils
from . import text_utils

#----------------------------------------------------------------------------

class ImguiWindow(glfw_window.GlfwWindow):
    def __init__(self, *, title='ImguiWindow', font=None, font_sizes=range(14,24), **glfw_kwargs):
        if font is None:
            font = text_utils.get_default_font()
        font_sizes = {int(size) for size in font_sizes}
        super().__init__(title=title, **glfw_kwargs)

        # Init fields.
        self._imgui_context  = None
        self._imgui_renderer = None
        self._imgui_fonts    = None
        self._cur_font_size  = max(font_sizes)

        # Delete leftover imgui.ini to avoid unexpected behavior.
        if os.path.isfile('imgui.ini'):
            os.remove('imgui.ini')

        # Init ImGui.
        self._imgui_context = imgui.create_context()
        self._imgui_renderer = _GlfwRenderer(self._glfw_window)
        self._attach_glfw_callbacks()
        imgui.get_io().ini_saving_rate = 0 # Disable creating imgui.ini at runtime.
        imgui.get_io().mouse_drag_threshold = 0 # Improve behavior with imgui_utils.drag_custom().
        self._imgui_fonts = {size: imgui.get_io().fonts.add_font_from_file_ttf(font, size) for size in font_sizes}
        self._imgui_renderer.refresh_font_texture()

    def close(self):
        self.make_context_current()
        self._imgui_fonts = None
        if self._imgui_renderer is not None:
            self._imgui_renderer.shutdown()
            self._imgui_renderer = None
        if self._imgui_context is not None:
            #imgui.destroy_context(self._imgui_context) # Commented out to avoid creating imgui.ini at the end.
            self._imgui_context = None
        super().close()

    def _glfw_key_callback(self, *args):
        super()._glfw_key_callback(*args)
        self._imgui_renderer.keyboard_callback(*args)

    @property
    def font_size(self):
        return self._cur_font_size

    @property
    def spacing(self):
        return round(self._cur_font_size * 0.4)

    def set_font_size(self, target): # Applied on next frame.
        self._cur_font_size = min((abs(key - target), key) for key in self._imgui_fonts.keys())[1]

    def begin_frame(self):
        # Begin glfw frame.
        super().begin_frame()

        # Process imgui events.
        self._imgui_renderer.mouse_wheel_multiplier = self._cur_font_size / 10
        if self.content_width > 0 and self.content_height > 0:
            self._imgui_renderer.process_inputs()

        # Begin imgui frame.
        imgui.new_frame()
        imgui.push_font(self._imgui_fonts[self._cur_font_size])
        imgui_utils.set_default_style(spacing=self.spacing, indent=self.font_size, scrollbar=self.font_size+4)

    def end_frame(self):
        imgui.pop_font()
        imgui.render()
        imgui.end_frame()
        self._imgui_renderer.render(imgui.get_draw_data())
        super().end_frame()

#----------------------------------------------------------------------------
# Wrapper class to use FixedPipelineRenderer on macOS due to OpenGL limitations

class _GlfwRenderer(imgui.integrations.opengl.FixedPipelineRenderer):
    def __init__(self, window, *args, **kwargs):
        self.window = window
        super().__init__()
        self.mouse_wheel_multiplier = 1
        self._map_keys()

    def keyboard_callback(self, window, key, scancode, action, mods):
        # Handle keyboard input for imgui
        pass

    def scroll_callback(self, window, x_offset, y_offset):
        self.io.mouse_wheel += y_offset * self.mouse_wheel_multiplier

    def _map_keys(self):
        # Map GLFW keys to imgui keys
        import glfw
        io = self.io
        io.key_map[imgui.KEY_TAB] = glfw.KEY_TAB
        io.key_map[imgui.KEY_LEFT_ARROW] = glfw.KEY_LEFT
        io.key_map[imgui.KEY_RIGHT_ARROW] = glfw.KEY_RIGHT
        io.key_map[imgui.KEY_UP_ARROW] = glfw.KEY_UP
        io.key_map[imgui.KEY_DOWN_ARROW] = glfw.KEY_DOWN
        io.key_map[imgui.KEY_PAGE_UP] = glfw.KEY_PAGE_UP
        io.key_map[imgui.KEY_PAGE_DOWN] = glfw.KEY_PAGE_DOWN
        io.key_map[imgui.KEY_HOME] = glfw.KEY_HOME
        io.key_map[imgui.KEY_END] = glfw.KEY_END
        io.key_map[imgui.KEY_DELETE] = glfw.KEY_DELETE
        io.key_map[imgui.KEY_BACKSPACE] = glfw.KEY_BACKSPACE
        io.key_map[imgui.KEY_ENTER] = glfw.KEY_ENTER
        io.key_map[imgui.KEY_ESCAPE] = glfw.KEY_ESCAPE
        io.key_map[imgui.KEY_A] = glfw.KEY_A
        io.key_map[imgui.KEY_C] = glfw.KEY_C
        io.key_map[imgui.KEY_V] = glfw.KEY_V
        io.key_map[imgui.KEY_X] = glfw.KEY_X
        io.key_map[imgui.KEY_Y] = glfw.KEY_Y
        io.key_map[imgui.KEY_Z] = glfw.KEY_Z

    def process_inputs(self):
        import glfw
        io = imgui.get_io()

        window_size = glfw.get_window_size(self.window)
        fb_size = glfw.get_framebuffer_size(self.window)

        io.display_size = window_size
        io.display_fb_scale = (
            fb_size[0] / window_size[0] if window_size[0] > 0 else 0,
            fb_size[1] / window_size[1] if window_size[1] > 0 else 0
        )

        io.delta_time = 1.0/60.0

        if glfw.get_window_attrib(self.window, glfw.FOCUSED):
            mouse_pos = glfw.get_cursor_pos(self.window)
            io.mouse_pos = mouse_pos
        else:
            io.mouse_pos = -1, -1

        io.mouse_down[0] = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT)
        io.mouse_down[1] = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT)
        io.mouse_down[2] = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_MIDDLE)

#----------------------------------------------------------------------------
