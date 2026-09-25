"""JAX Krylov exponential-action routines for linear systems."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm


def arnoldi_step(
    matrix_vector_product: Callable[[jax.Array], jax.Array],
    vector: jax.Array,
    krylov_dim: int = 30,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build a row-oriented Arnoldi basis and its projected Hessenberg matrix."""
    dimension = vector.shape[0]
    basis_size = min(krylov_dim, dimension)
    if basis_size < 1:
        raise ValueError("krylov_dim must be positive")

    vector_norm = jnp.linalg.norm(vector)
    safe_norm = jnp.where(vector_norm > 0, vector_norm, 1)
    basis = jnp.zeros((basis_size + 1, dimension), dtype=vector.dtype)
    basis = basis.at[0].set(vector / safe_norm)
    hessenberg = jnp.zeros((basis_size + 1, basis_size), dtype=vector.dtype)
    breakdown_tolerance = 10 * jnp.finfo(vector.real.dtype).eps

    def arnoldi_column(column: int, state: tuple[jax.Array, jax.Array]):
        basis, hessenberg = state
        residual = matrix_vector_product(basis[column])

        def orthogonalize(
            _: int, values: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            hessenberg, residual = values

            def project(row: int, projected: tuple[jax.Array, jax.Array]):
                hessenberg, residual = projected
                coefficient = jnp.vdot(basis[row], residual)
                hessenberg = hessenberg.at[row, column].add(coefficient)
                residual = residual - coefficient * basis[row]
                return hessenberg, residual

            return jax.lax.fori_loop(0, column + 1, project, (hessenberg, residual))

        hessenberg, residual = jax.lax.fori_loop(
            0, 2, orthogonalize, (hessenberg, residual)
        )
        residual_norm = jnp.linalg.norm(residual)
        hessenberg = hessenberg.at[column + 1, column].set(residual_norm)
        threshold = breakdown_tolerance * jnp.maximum(
            1.0, jnp.linalg.norm(hessenberg)
        )
        next_vector = jnp.where(
            residual_norm > threshold,
            residual / jnp.where(residual_norm > 0, residual_norm, 1),
            jnp.zeros_like(residual),
        )
        basis = basis.at[column + 1].set(next_vector)
        return basis, hessenberg

    basis, hessenberg = jax.lax.fori_loop(
        0, basis_size, arnoldi_column, (basis, hessenberg)
    )
    return basis[:basis_size], hessenberg[:basis_size, :basis_size], vector_norm


def expm_multiply_krylov(
    matrix_vector_product: Callable[[jax.Array], jax.Array],
    vector: jax.Array,
    delta_time: jax.Array,
    krylov_dim: int = 30,
) -> jax.Array:
    """Approximate ``exp(A * delta_time) @ vector`` with an Arnoldi projection."""
    basis, hessenberg, vector_norm = arnoldi_step(
        lambda value: delta_time * matrix_vector_product(value),
        vector,
        krylov_dim,
    )
    first_basis_vector = jnp.zeros((basis.shape[0],), dtype=vector.dtype).at[0].set(1)
    projected = expm(hessenberg) @ first_basis_vector
    result = vector_norm * (basis.T @ projected)
    return jnp.where(vector_norm > 0, result, jnp.zeros_like(result))


@partial(jax.jit, static_argnames=("krylov_dim",))
def compute_trajectory(
    matrix: jax.Array,
    initial_state: jax.Array,
    times: jax.Array,
    krylov_dim: int = 30,
) -> jax.Array:
    """Propagate over possibly nonuniform times with a JIT-compiled scan."""
    delta_times = jnp.diff(times)

    def step(current_state: jax.Array, delta_time: jax.Array):
        next_state = expm_multiply_krylov(
            lambda value: matrix @ value,
            current_state,
            delta_time,
            krylov_dim,
        )
        return next_state, next_state

    _, later_states = jax.lax.scan(step, initial_state, delta_times)
    return jnp.concatenate((initial_state[None, :], later_states), axis=0)