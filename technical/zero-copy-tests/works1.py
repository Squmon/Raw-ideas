from cuda.bindings import runtime as cudart
import cupy as cp
from OpenGL.GL import *

from cuda.bindings import runtime as cudart
import cupy as cp
from OpenGL.GL import *


def _check_cuda(result):
  """Универсальный обработчик результатов вызовов cuda-python"""
  if isinstance(result, tuple):
    err = result[0]
  else:
    err = result

  # Проверяем код ошибки (даже если это enum)
  if int(err) != 0:
    raise RuntimeError(f"CUDA Error code: {err}")

  # Если функция возвращала дополнительные значения, возвращаем их
  if isinstance(result, tuple):
    if len(result) == 2:
      return result[1]
    elif len(result) > 2:
      return result[1:]
  return None


class Texture2DProjector:
  def __init__(self, width: int, height: int):
    self.width = width
    self.height = height

    # 1. Создаем OpenGL текстуру
    self.texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, self.texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB32F,
        width,
        height,
        0,
        GL_RGB,
        GL_FLOAT,
        None,
    )
    glBindTexture(GL_TEXTURE_2D, 0)

    # 2. Создаем Pixel Buffer Object (PBO)
    self.pbo_id = glGenBuffers(1)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
    buffer_size = width * height * 3 * 4  # 3 канала, float32 (4 байта)
    glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer_size, None, GL_STREAM_DRAW)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    # 3. Регистрируем PBO в CUDA
    flags = (
        cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
    )
    self.cuda_res = _check_cuda(
        cudart.cudaGraphicsGLRegisterBuffer(self.pbo_id, flags)
    )

  def project(self, tensor) -> int:
    """Принимает ЛЮБОЙ DLPack-совместимый тензор (JAX, PyTorch, CuPy)

    и проецирует его в текстуру. Возвращает OpenGL texture_id.
    """
    # А) Мапим OpenGL-ресурс в пространство CUDA
    _check_cuda(cudart.cudaGraphicsMapResources(1, self.cuda_res, 0))

    # Б) Получаем указатель на память буфера
    dev_ptr, size = _check_cuda(
        cudart.cudaGraphicsResourceGetMappedPointer(self.cuda_res)
    )

    # В) Создаем CuPy-обертку поверх указателя PBO (Zero-Copy)
    mem = cp.cuda.UnownedMemory(dev_ptr, size, owner=None)
    mptr = cp.cuda.MemoryPointer(mem, 0)
    pbo_array = cp.ndarray(
        (self.height, self.width, 3), dtype=cp.float32, memptr=mptr
    )

    # Г) Перекладываем данные из входного тензора в PBO (Device-to-Device)
    cp.copyto(pbo_array, cp.from_dlpack(tensor))

    # Д) Размапиваем ресурс обратно для OpenGL
    _check_cuda(cudart.cudaGraphicsUnmapResources(1, self.cuda_res, 0))

    # Е) Обновляем саму OpenGL текстуру из PBO
    glBindTexture(GL_TEXTURE_2D, self.texture_id)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
    glTexSubImage2D(
        GL_TEXTURE_2D,
        0,
        0,
        0,
        self.width,
        self.height,
        GL_RGB,
        GL_FLOAT,
        None,
    )
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
    glBindTexture(GL_TEXTURE_2D, 0)

    return self.texture_id

  def close(self):
    """Корректная очистка ресурсов"""
    if hasattr(self, "cuda_res"):
      _check_cuda(cudart.cudaGraphicsUnregisterResource(self.cuda_res))
    if hasattr(self, "pbo_id"):
      glDeleteBuffers(1, [self.pbo_id])
    if hasattr(self, "texture_id"):
      glDeleteTextures(1, [self.texture_id])


import glfw
import jax
import jax.numpy as jnp
from OpenGL.GL import *
# Импортируем наш спроектированный класс (предположим, он в проекторе)
# from projector import Texture2DProjector


def main():
    if not glfw.init():
        return

    width, height = 512, 512
    window = glfw.create_window(width, height, "Modular Tensor Projector", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)

    # Инициализируем наш проектор
    projector = Texture2DProjector(width, height)
    y_coords, x_coords = jnp.mgrid[0:1:512j, 0:1:512j]

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # 1. Генерируем данные в JAX (или получаем из PyTorch)
        t = glfw.get_time()
        frame_gpu = jnp.sin(x_coords * 12 + t) * jnp.cos(y_coords * 12 - t)
        image_jax = jnp.stack(
            [
                frame_gpu,
                jnp.abs(jnp.sin(t)) * jnp.ones_like(frame_gpu),
                -frame_gpu,
            ],
            axis=-1,
        )

        # 2. ПРОЕЦИРУЕМ ТЕНЗОР В ТЕКСТУРУ ОДНИМ МЕТОДОМ!
        # На выходе получаем готовый OpenGL texture_id
        tex_id = projector.project(image_jax)

        # 3. Рендеринг текстуры в окне OpenGL
        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex_id)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(-1, -1)
        glTexCoord2f(1, 0)
        glVertex2f(1, -1)
        glTexCoord2f(1, 1)
        glVertex2f(1, 1)
        glTexCoord2f(0, 1)
        glVertex2f(-1, 1)
        glEnd()

        glfw.swap_buffers(window)

    # Очистка
    projector.close()
    glfw.terminate()


if __name__ == "__main__":
    main()
