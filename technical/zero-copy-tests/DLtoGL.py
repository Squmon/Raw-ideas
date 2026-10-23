from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from cuda.bindings import runtime as cudart
import cupy as cp
import glfw
from OpenGL.GL import *


def _pair(value, name: str) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        return float(value), float(value)
    result = tuple(value)
    if len(result) != 2:
        raise ValueError(f"{name} must be a number or a pair of numbers")
    return float(result[0]), float(result[1])


def _draw_vertices(points: Iterable[Sequence[float]], primitive: int) -> None:
    vertices = [tuple(point) for point in points]
    if any(len(point) not in (2, 3) for point in vertices):
        raise ValueError("each point must contain two or three coordinates")
    glBegin(primitive)
    try:
        for point in vertices:
            if len(point) == 2:
                glVertex2f(float(point[0]), float(point[1]))
            else:
                glVertex3f(float(point[0]), float(point[1]), float(point[2]))
    finally:
        glEnd()


def draw_image(texture_id: int, position, scale) -> None:
    """Draw a texture centered at ``position`` with the given width and height.

    ``scale`` may be a scalar (same width and height) or a ``(width, height)``
    pair. Coordinates use the current OpenGL projection.
    """
    x, y = _pair(position, "position")
    width, height = _pair(scale, "scale")
    half_width, half_height = width / 2, height / 2

    glPushAttrib(GL_ENABLE_BIT | GL_TEXTURE_BIT | GL_CURRENT_BIT)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    try:
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(x - half_width, y - half_height)
        glTexCoord2f(1, 0)
        glVertex2f(x + half_width, y - half_height)
        glTexCoord2f(1, 1)
        glVertex2f(x + half_width, y + half_height)
        glTexCoord2f(0, 1)
        glVertex2f(x - half_width, y + half_height)
        glEnd()
    finally:
        glPopAttrib()


def draw_points(positions, color=None, point_size: float = 4.0) -> None:
    """Draw a point cloud; if ``color`` is None, use the current GL color."""
    if point_size <= 0:
        raise ValueError("point_size must be positive")
    if color is not None:
        rgba = tuple(color)
        if len(rgba) not in (3, 4):
            raise ValueError("color must contain three or four components")
        if len(rgba) == 3:
            rgba += (1.0,)

    glPushAttrib(GL_CURRENT_BIT | GL_POINT_BIT)
    try:
        if color is not None:
            glColor4f(*(float(component) for component in rgba))
        glPointSize(float(point_size))
        _draw_vertices(positions, GL_POINTS)
    finally:
        glPopAttrib()


def draw_line(points, color=None, line_width: float = 1.0) -> None:
    """Draw a connected line through the supplied 2D or 3D points."""
    if line_width <= 0:
        raise ValueError("line_width must be positive")
    if color is not None:
        rgba = tuple(color)
        if len(rgba) not in (3, 4):
            raise ValueError("color must contain three or four components")
        if len(rgba) == 3:
            rgba += (1.0,)

    glPushAttrib(GL_CURRENT_BIT | GL_LINE_BIT)
    try:
        if color is not None:
            glColor4f(*(float(component) for component in rgba))
        glLineWidth(float(line_width))
        _draw_vertices(points, GL_LINE_STRIP)
    finally:
        glPopAttrib()


class Camera2D:
    """Simple 2D camera controlled by WASD (pan) and Q/E (zoom)."""

    def __init__(self, move_speed: float = 1.0, zoom_speed: float = 1.5):
        self.x = 0.0
        self.y = 0.0
        self.zoom = 1.0
        self.move_speed = move_speed
        self.zoom_speed = zoom_speed

    def update(self, window, delta_time: float) -> None:
        distance = self.move_speed * delta_time / self.zoom
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            self.y += distance
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            self.y -= distance
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            self.x -= distance
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            self.x += distance

        zoom_factor = self.zoom_speed * delta_time
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
            self.zoom *= 1.0 + zoom_factor
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
            self.zoom /= 1.0 + zoom_factor
        self.zoom = min(max(self.zoom, 0.05), 50.0)

    def apply(self) -> None:
        glPushMatrix()
        glScalef(self.zoom, self.zoom, 1.0)
        glTranslatef(-self.x, -self.y, 0.0)

    @staticmethod
    def restore() -> None:
        glPopMatrix()


def _check_cuda(result):
    """Универсальный обработчик результатов вызовов cuda-python."""
    if isinstance(result, tuple):
        err = result[0]
    else:
        err = result

    if int(err) != 0:
        raise RuntimeError(f"CUDA Error code: {err}")

    if isinstance(result, tuple):
        if len(result) == 2:
            return result[1]
        elif len(result) > 2:
            return result[1:]
    return None


class CudaGLProjector(ABC):
    """
    Абстрактный базовый класс для проекции DLPack-тензоров в OpenGL буферы.
    Инкапсулирует логику CUDA-OpenGL Interop (Map/Unmap/Pointers).
    """

    def __init__(self):
        self.cuda_res = None

    def _register_gl_buffer(
        self,
        buffer_id: int,
        flags=cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
    ):
        """Регистрация любого OpenGL буфера в контексте CUDA."""
        self.cuda_res = _check_cuda(
            cudart.cudaGraphicsGLRegisterBuffer(buffer_id, flags)
        )

    def project(self, tensor):
        """
        Универсальный публичный метод проецирования.
        Принимает любой DLPack-совместимый тензор (JAX, PyTorch, CuPy).
        """
        if self.cuda_res is None:
            raise RuntimeError("CUDA-GL ресурс не зарегистрирован!")

        # 1. Захват контроля над ресурсом со стороны CUDA
        _check_cuda(cudart.cudaGraphicsMapResources(1, self.cuda_res, 0))

        try:
            # 2. Получение указателя на память VRAM и ее размера
            dev_ptr, size = _check_cuda(
                cudart.cudaGraphicsResourceGetMappedPointer(self.cuda_res)
            )

            # 3. Передача управления методу соответствующей проекции
            self._process_and_upload(dev_ptr, size, tensor)

        finally:
            # 4. Гарантированный возврат контроля OpenGL
            _check_cuda(cudart.cudaGraphicsUnmapResources(1, self.cuda_res, 0))

    @abstractmethod
    def _process_and_upload(self, dev_ptr: int, size: int, tensor):
        """Кастомная логика зажима указателя в CuPy и загрузки в буфер OpenGL."""
        pass

    def close(self):
        """Очистка ресурсов CUDA."""
        if self.cuda_res is not None:
            _check_cuda(cudart.cudaGraphicsUnregisterResource(self.cuda_res))
            self.cuda_res = None


# =====================================================================
# 1. ТЕКСТУРА 2D (Карты признаков, Изображения, Маски)
# =====================================================================
class Texture2DProjector(CudaGLProjector):
    def __init__(self, width: int, height: int):
        super().__init__()
        self.width = width
        self.height = height

        # Выделение памяти под OpenGL 2D Текстуру
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, None
        )
        glBindTexture(GL_TEXTURE_2D, 0)

        # Выделение памяти под PBO
        self.pbo_id = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        buffer_size = width * height * 3 * 4  # 3 канала float32
        glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer_size, None, GL_STREAM_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        self._register_gl_buffer(self.pbo_id)

    def _process_and_upload(self, dev_ptr: int, size: int, tensor):
        # Оборачивание VRAM-памяти в CuPy массив [H, W, 3]
        mem = cp.cuda.UnownedMemory(dev_ptr, size, owner=None)
        mptr = cp.cuda.MemoryPointer(mem, 0)
        pbo_array = cp.ndarray(
            (self.height, self.width, 3), dtype=cp.float32, memptr=mptr
        )

        # Device-to-Device копирование из DLPack тензора
        cp.copyto(pbo_array, cp.from_dlpack(tensor))

        # Загрузка из PBO в саму текстуру 2D
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0, self.width, self.height, GL_RGB, GL_FLOAT, None
        )
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        glBindTexture(GL_TEXTURE_2D, 0)

    def close(self):
        super().close()
        if hasattr(self, "pbo_id"):
            glDeleteBuffers(1, [self.pbo_id])
        if hasattr(self, "texture_id"):
            glDeleteTextures(1, [self.texture_id])


# =====================================================================
# 2. ТЕКСТУРА 3D (Воксели, Объемный рендеринг, NeRF, МРТ)
# =====================================================================
class Texture3DProjector(CudaGLProjector):
    def __init__(self, width: int, height: int, depth: int):
        super().__init__()
        self.width = width
        self.height = height
        self.depth = depth

        # Выделение памяти под OpenGL 3D Текстуру
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.texture_id)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexImage3D(
            GL_TEXTURE_3D, 0, GL_RGB32F, width, height, depth, 0, GL_RGB, GL_FLOAT, None
        )
        glBindTexture(GL_TEXTURE_3D, 0)

        # PBO для объемных данных
        self.pbo_id = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        buffer_size = width * height * depth * 3 * 4  # 3 канала float32
        glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer_size, None, GL_STREAM_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        self._register_gl_buffer(self.pbo_id)

    def _process_and_upload(self, dev_ptr: int, size: int, tensor):
        # Оборачивание VRAM-памяти в CuPy 3D массив [D, H, W, 3]
        mem = cp.cuda.UnownedMemory(dev_ptr, size, owner=None)
        mptr = cp.cuda.MemoryPointer(mem, 0)
        pbo_array = cp.ndarray(
            (self.depth, self.height, self.width, 3), dtype=cp.float32, memptr=mptr
        )

        cp.copyto(pbo_array, cp.from_dlpack(tensor))

        # Загрузка из PBO в саму 3D-текстуру
        glBindTexture(GL_TEXTURE_3D, self.texture_id)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        glTexSubImage3D(
            GL_TEXTURE_3D,
            0,
            0,
            0,
            0,
            self.width,
            self.height,
            self.depth,
            GL_RGB,
            GL_FLOAT,
            None,
        )
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        glBindTexture(GL_TEXTURE_3D, 0)

    def close(self):
        super().close()
        if hasattr(self, "pbo_id"):
            glDeleteBuffers(1, [self.pbo_id])
        if hasattr(self, "texture_id"):
            glDeleteTextures(1, [self.texture_id])


# =====================================================================
# 3. VBO (Vertex Buffer Object: Точки, Геометрия, Частицы, Графы)
# =====================================================================
class VBOProjector(CudaGLProjector):
    def __init__(self, num_vertices: int, components_per_vertex: int = 3):
        super().__init__()
        self.num_vertices = num_vertices
        self.components = components_per_vertex

        # Выделение буфера вершин OpenGL (GL_ARRAY_BUFFER)
        self.vbo_id = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        buffer_size = num_vertices * components_per_vertex * 4  # float32
        glBufferData(GL_ARRAY_BUFFER, buffer_size, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        self._register_gl_buffer(self.vbo_id)

    def _process_and_upload(self, dev_ptr: int, size: int, tensor):
        # Оборачивание VRAM в CuPy массив [N, Components] (например, [N, 3] для XYZ)
        mem = cp.cuda.UnownedMemory(dev_ptr, size, owner=None)
        mptr = cp.cuda.MemoryPointer(mem, 0)
        vbo_array = cp.ndarray(
            (self.num_vertices, self.components), dtype=cp.float32, memptr=mptr
        )

        # Прямое копирование тензора напрямую в буфер вершин VBO без промежуточного PBO!
        cp.copyto(vbo_array, cp.from_dlpack(tensor))

    def bind_and_draw(self, primitive_type=GL_POINTS):
        """Вспомогательный метод для отрисовки вершин VBO."""
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(self.components, GL_FLOAT, 0, None)

        glDrawArrays(primitive_type, 0, self.num_vertices)

        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def close(self):
        super().close()
        if hasattr(self, "vbo_id"):
            glDeleteBuffers(1, [self.vbo_id])
