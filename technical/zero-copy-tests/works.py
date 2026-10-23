from cuda.bindings import runtime as cudart
import cupy as cp
import glfw
import jax
import jax.numpy as jnp
from OpenGL.GL import *


def check_cuda_err(result):
    """Универсальная функция для проверки ошибок CUDA-runtime"""
    if isinstance(result, tuple):
        err = result[0]
        # В cuda-python код успешного выполнения обычно 0 (cudaSuccess)
        if err != 0:
            raise RuntimeError(f"CUDA Error code: {err}")
        # Если возвращалось только (err, value), возвращаем просто value
        if len(result) == 2:
            return result[1]
        # Если элементов больше (например, err, ptr, size), возвращаем срез без ошибки
        return result[1:]
    return result


def main():
    if not glfw.init():
        return

    width, height = 512, 512
    window = glfw.create_window(
        width, height, "JAX + PBO True Zero-Copy Visualizer", None, None
    )
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # Создаем OpenGL текстуру
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
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

    # Создаем Pixel Buffer Object (PBO) в видеопамяти
    pbo_id = glGenBuffers(1)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id)
    buffer_size = width * height * 3 * 4
    glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer_size, None, GL_STREAM_DRAW)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    # Регистрируем OpenGL PBO в контексте CUDA
    flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
    # Теперь здесь ожидаем ровно один возвращаемый ресурс
    cuda_res = check_cuda_err(cudart.cudaGraphicsGLRegisterBuffer(pbo_id, flags))

    y_coords, x_coords = jnp.mgrid[0:1:512j, 0:1:512j]

    print("Пайплайн запущен! Окно открыто. Закройте его для выхода.")


    while not glfw.window_should_close(window):
        glfw.poll_events()

        # 1. Считаем кадр на GPU в JAX (с исправленной размерностью)
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

        # 2. Мапим OpenGL-ресурс в пространство CUDA (передаем cuda_res напрямую без списка!)
        check_cuda_err(cudart.cudaGraphicsMapResources(1, cuda_res, 0))

        # Получаем указатель на память буфера в VRAM
        dev_ptr, size = check_cuda_err(
            cudart.cudaGraphicsResourceGetMappedPointer(cuda_res)
        )

        # 3. Создаем CuPy-обертку поверх указателя PBO (Zero-Copy)
        mem = cp.cuda.UnownedMemory(dev_ptr, size, owner=None)
        mptr = cp.cuda.MemoryPointer(mem, 0)
        pbo_array = cp.ndarray((height, width, 3), dtype=cp.float32, memptr=mptr)

        # 4. Перекладываем данные из JAX в PBO внутри VRAM (Device-to-Device)
        cp.copyto(pbo_array, cp.from_dlpack(image_jax))

        # 5. Размапиваем ресурс обратно для OpenGL (также без списка)
        check_cuda_err(cudart.cudaGraphicsUnmapResources(1, cuda_res, 0))

        # 6. Рендеринг текстуры силами OpenGL из PBO
        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, texture_id)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id)

        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            width,
            height,
            GL_RGB,
            GL_FLOAT,
            None,
        )
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

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

    # Очистка ресурсов
    check_cuda_err(cudart.cudaGraphicsUnregisterResource(cuda_res))
    glfw.terminate()


if __name__ == "__main__":
    main()
