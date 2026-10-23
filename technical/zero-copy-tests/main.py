import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"

import jax
import jax.numpy as jnp
from DLtoGL import *
import glfw

@jax.jit(static_argnames=("num_points",))
def foo(x_coords, y_coords, angles, t, num_points):
    frame_2d = jnp.sin(x_coords * 10 + t) * jnp.cos(y_coords * 10 - t)
    image_jax = jnp.stack(
        [frame_2d, jnp.abs(jnp.sin(t)) * jnp.ones_like(frame_2d), -frame_2d], axis=-1
    )

    # --- 2. Генерация данных для VBO (Облако точек / Частицы) ---
    radius = 0.5 + 0.2 * jnp.sin(t * 3)
    pts_x = radius * jnp.cos(angles + t)
    pts_y = radius * jnp.sin(angles + t)
    pts_z = jnp.zeros(num_points)
    points_jax = jnp.stack([pts_x, pts_y, pts_z], axis=-1)

    return image_jax, points_jax


def main():
    if not glfw.init():
        return

    width, height = 512, 512
    window = glfw.create_window(width, height, "Unified Tensor Projectors", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)

    # Инициализация проекторов
    proj_2d = Texture2DProjector(width, height)

    num_points = 40
    proj_vbo = VBOProjector(num_vertices=num_points, components_per_vertex=3)

    # Заготовка данных в JAX
    y_coords, x_coords = jnp.mgrid[0:1:512j, 0:1:512j]
    angles = jnp.linspace(0, 2 * jnp.pi, num_points)
    camera = Camera2D()
    previous_time = glfw.get_time()

    print("Проекторы инициализированы успешно. Рендеринг запущен.")

    while not glfw.window_should_close(window):
        glfw.poll_events()
        current_time = glfw.get_time()
        camera.update(window, current_time - previous_time)
        previous_time = current_time
        t = current_time

        image_jax, points_jax = foo(x_coords, y_coords, angles, t, num_points)
        proj_2d.project(image_jax)
        proj_vbo.project(points_jax)

        # --- 3. Отрисовка результатов силами OpenGL ---
        glClear(GL_COLOR_BUFFER_BIT)

        camera.apply()
        draw_image(proj_2d.texture_id, (0, 0), (2, 2))

        # Рендеринг динамического VBO поверх текстуры
        glPointSize(4.0)
        proj_vbo.bind_and_draw(GL_POINTS)
        camera.restore()

        glfw.swap_buffers(window)

    # Очистка
    proj_2d.close()
    proj_vbo.close()
    glfw.terminate()


if __name__ == "__main__":
    main()
