"""Focused tests for graph generation and alternate ODE solvers."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

from custom_solver import compute_trajectory
from synchronization import (
    ExperimentConfig,
    generate_system,
    solve_synchronization,
)


class SynchronizationSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(
            dim=3,
            scale=0.1,
            coupling=0.4,
            t_end=0.2,
            n_times=7,
        )

    def test_current_graph_generator_returns_bcoo(self) -> None:
        matrix_a, matrix_b, _, _, _ = generate_system(19, self.config)
        self.assertIsInstance(matrix_a, BCOO)
        self.assertIsInstance(matrix_b, BCOO)
        self.assertEqual(matrix_a.shape, (3, 3))
        np.testing.assert_allclose(matrix_a.todense(), -matrix_a.todense().T)
        np.testing.assert_allclose(matrix_a.todense(), matrix_b.todense())

    def test_small_world_generator_is_seeded_scaled_laplacian(self) -> None:
        config = ExperimentConfig(
            dim=10,
            graph_generator="small_world",
            small_world_degree=4,
            small_world_rewire_probability=0.0,
        )
        matrix_a, matrix_b, _, _, _ = generate_system(41, config)
        repeated_matrix = generate_system(41, config)[0]
        dense = np.asarray(matrix_a.todense())
        self.assertIsInstance(matrix_a, BCOO)
        self.assertIsInstance(matrix_b, BCOO)
        np.testing.assert_array_equal(dense, np.asarray(repeated_matrix.todense()))
        np.testing.assert_allclose(dense, dense.T, atol=1e-7)
        np.testing.assert_allclose(dense.sum(axis=1), 0.0, atol=2e-7)
        np.testing.assert_allclose(np.diag(dense), -0.4, atol=1e-7)

    def test_small_world_rejects_degree_not_less_than_dimension(self) -> None:
        config = ExperimentConfig(dim=4, graph_generator="small_world")
        with self.assertRaisesRegex(ValueError, "smaller than dim"):
            generate_system(41, config)

    def test_jax_solvers_match_scipy_reference(self) -> None:
        for same_dynamics in (True, False):
            config = ExperimentConfig(
                dim=3,
                scale=0.1,
                coupling=0.4,
                t_end=0.2,
                n_times=7,
                same_dynamics=same_dynamics,
            )
            reference = solve_synchronization(19, config, method="scipy")
            for method in ("jax_expm_vmap", "jax_krylov"):
                with self.subTest(method=method, same_dynamics=same_dynamics):
                    result = solve_synchronization(
                        19,
                        config,
                        method=method,
                        krylov_dim=6,
                    )
                    self.assertEqual(result.state_a.shape, (7, 3))
                    np.testing.assert_allclose(
                        result.state_a, reference.state_a, rtol=5e-4, atol=5e-5
                    )
                    np.testing.assert_allclose(
                        result.state_b, reference.state_b, rtol=5e-4, atol=5e-5
                    )

    def test_krylov_handles_zero_input_and_dimension_cap(self) -> None:
        matrix = BCOO.fromdense(jnp.diag(jnp.asarray([1.0, -0.5])))
        times = jnp.asarray([0.0, 0.1, 0.35])
        zero_result = compute_trajectory(
            matrix, jnp.zeros(2), times, krylov_dim=8
        )
        np.testing.assert_allclose(zero_result, 0.0)

        initial = jnp.asarray([1.0, 2.0])
        result = compute_trajectory(matrix, initial, times, krylov_dim=8)
        expected = jnp.stack(
            (jnp.exp(times), 2.0 * jnp.exp(-0.5 * times)), axis=1
        )
        np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-6)

    def test_krylov_preserves_complex_rotation(self) -> None:
        matrix = BCOO.fromdense(
            jnp.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=jnp.complex64)
        )
        times = jnp.asarray([0.0, 0.1, 0.4])
        initial = jnp.asarray([1.0 + 0.0j, 0.0 + 0.0j])
        result = compute_trajectory(matrix, initial, times, krylov_dim=2)
        expected = jnp.stack(
            (jnp.cos(times).astype(jnp.complex64), -jnp.sin(times).astype(jnp.complex64)),
            axis=1,
        )
        np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-6)

    def test_unknown_graph_generator_has_actionable_error(self) -> None:
        config = ExperimentConfig(graph_generator="not_registered")
        with self.assertRaisesRegex(ValueError, "Unknown graph_generator"):
            generate_system(1, config)


if __name__ == "__main__":
    unittest.main()