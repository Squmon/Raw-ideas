"""Matrix-exponential experiments for one-way linear synchronization."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from jax.experimental.sparse import BCOO
from jax.scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from custom_solver import expm_multiply_krylov


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters shared by generated systems and their numerical solutions."""

    dim: int = 10
    scale: float = 0.1
    coupling: float = 3.0
    t_end: float = 500.0
    n_times: int = 500
    same_dynamics: bool = True
    graph_generator: str = "current_random_antisymmetric"
    small_world_degree: int = 4
    small_world_rewire_probability: float = 0.1

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("dim must be positive")
        if self.n_times < 2:
            raise ValueError("n_times must be at least 2")
        if self.t_end <= 0:
            raise ValueError("t_end must be positive")
        if self.scale < 0:
            raise ValueError("scale must be non-negative")
        if self.small_world_degree < 2 or self.small_world_degree % 2:
            raise ValueError("small_world_degree must be a positive even integer")
        if not 0.0 <= self.small_world_rewire_probability <= 1.0:
            raise ValueError("small_world_rewire_probability must be between 0 and 1")


DEFAULT_CONFIG = ExperimentConfig()


@dataclass(frozen=True)
class SynchronizationSolution:
    """A complete generated system and its states sampled over time."""

    key: np.ndarray
    ts: np.ndarray
    state_a: np.ndarray
    state_b: np.ndarray
    matrix_a: np.ndarray
    matrix_b: np.ndarray
    sync_indices: tuple[int, int]
    coupling: float
    config_json: str

    @property
    def mean_squared_error(self) -> np.ndarray:
        return np.mean(np.square(self.state_a - self.state_b), axis=1)


def _key_data(key: jax.Array | np.ndarray | Iterable[int] | int) -> jax.Array:
    """Normalize legacy, typed, and integer keys to JAX key data."""
    if isinstance(key, (int, np.integer)):
        return jax.random.PRNGKey(int(key))
    try:
        return jax.random.key_data(key)
    except (TypeError, ValueError):
        return jnp.asarray(key, dtype=jnp.uint32)


def _current_random_antisymmetric(
    key: jax.Array, config: ExperimentConfig
) -> tuple[BCOO, jax.Array]:
    mask = jax.random.normal(key, (config.dim, config.dim)) > 0.2
    dense_matrix = (
        mask.astype(jnp.float32) - mask.T.astype(jnp.float32)
    )# / config.dim
    return BCOO.fromdense(dense_matrix), dense_matrix


def _small_world_laplacian(
    key: jax.Array, config: ExperimentConfig
) -> tuple[BCOO, jax.Array]:
    """Build negative graph-Laplacian dynamics from a Watts-Strogatz graph."""
    if config.small_world_degree >= config.dim:
        raise ValueError(
            "small_world_degree must be smaller than dim for the small-world generator"
        )
    graph_seed = int(jax.random.bits(key, (), dtype=jnp.uint32))
    graph = nx.watts_strogatz_graph(
        n=config.dim,
        k=config.small_world_degree,
        p=config.small_world_rewire_probability,
        seed=graph_seed,
    )
    adjacency = nx.to_numpy_array(graph, dtype=np.float32)
    degree_matrix = np.diag(adjacency.sum(axis=1))
    dense_matrix = jnp.asarray(
        -(degree_matrix - adjacency) / config.dim,
        dtype=jnp.float32,
    )
    return BCOO.fromdense(dense_matrix), dense_matrix


def _small_world_antisymmetric(
    key: jax.Array, config: ExperimentConfig
) -> tuple[BCOO, jax.Array]:
    """Build negative graph-Laplacian dynamics from a Watts-Strogatz graph."""
    if config.small_world_degree >= config.dim:
        raise ValueError(
            "small_world_degree must be smaller than dim for the small-world generator"
        )
    graph_seed = int(jax.random.bits(key, (), dtype=jnp.uint32))
    graph = nx.watts_strogatz_graph(
        n=config.dim,
        k=config.small_world_degree,
        p=config.small_world_rewire_probability,
        seed=graph_seed,
    )
    adjacency = nx.to_numpy_array(graph, dtype=np.float32)

    dense_matrix = (adjacency - adjacency.T)

    return BCOO.fromdense(dense_matrix), dense_matrix


def _scale_free_antisymmetric(
    key: jax.Array, config: ExperimentConfig
) -> tuple[BCOO, jax.Array]:
    """Build negative graph-Laplacian dynamics from a Watts-Strogatz graph."""
    if config.small_world_degree >= config.dim:
        raise ValueError(
            "small_world_degree must be smaller than dim for the small-world generator"
        )
    graph_seed = int(jax.random.bits(key, (), dtype=jnp.uint32))
    graph = nx.scale_free_graph(n=config.dim, seed=graph_seed)
    adjacency = nx.to_numpy_array(graph, dtype=np.float32)

    dense_matrix = (adjacency - adjacency.T)

    return BCOO.fromdense(dense_matrix), dense_matrix

GRAPH_GENERATORS = {
    "current_random_antisymmetric": _current_random_antisymmetric,
    "small_world_laplacian": _small_world_laplacian,
    "small_world_antisymmetric":_small_world_antisymmetric,
    "scale_free_antisymmetric":_scale_free_antisymmetric
}


def generate_system(
    key: jax.Array | np.ndarray | Iterable[int] | int,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[BCOO, BCOO, jax.Array, tuple[int, int], np.ndarray]:
    """Generate sparse dynamics matrices, initial state, and hub indices."""
    try:
        graph_generator = GRAPH_GENERATORS[config.graph_generator]
    except KeyError as error:
        supported = ", ".join(sorted(GRAPH_GENERATORS))
        raise ValueError(
            f"Unknown graph_generator {config.graph_generator!r}; choose from {supported}"
        ) from error

    normalized_key = _key_data(key)
    matrix_key, second_matrix_key, initial_a_key, initial_b_key = jax.random.split(
        normalized_key, 4
    )
    matrix_a, dense_a = graph_generator(matrix_key, config)
    if config.same_dynamics:
        matrix_b = matrix_a
    else:
        matrix_b, _ = graph_generator(second_matrix_key, config)
    hub = int(jnp.argmax(jnp.abs(dense_a).sum(axis=1)))
    initial = jnp.concatenate(
        (
            config.scale * jax.random.normal(initial_a_key, (config.dim,)),
            config.scale * jax.random.normal(initial_b_key, (config.dim,)),
        )
    )
    return matrix_a, matrix_b, initial, (hub, hub), np.asarray(normalized_key)


def build_generator(
    matrix_a: BCOO | np.ndarray,
    matrix_b: BCOO | np.ndarray,
    sync_indices: tuple[int, int],
    coupling: float,
) -> np.ndarray:
    """Build the block generator for da/dt=Aa, db/dt=Bb+k(a_i-b_j)e_j."""
    matrix_a = _as_dense_numpy(matrix_a)
    matrix_b = _as_dense_numpy(matrix_b)
    if matrix_a.ndim != 2 or matrix_a.shape[0] != matrix_a.shape[1]:
        raise ValueError("matrix_a must be square")
    if matrix_b.shape != matrix_a.shape:
        raise ValueError("matrix_b must have the same shape as matrix_a")

    dim = matrix_a.shape[0]
    index_a, index_b = sync_indices
    if not (0 <= index_a < dim and 0 <= index_b < dim):
        raise ValueError("synchronization indices must be within matrix dimensions")

    lower_left = np.zeros_like(matrix_a)
    lower_right = matrix_b.copy()
    lower_left[index_b, index_a] = coupling
    lower_right[index_b, index_b] -= coupling
    return np.block(
        [[matrix_a, np.zeros_like(matrix_a)], [lower_left, lower_right]]
    )


def _as_dense_numpy(matrix: BCOO | np.ndarray) -> np.ndarray:
    if isinstance(matrix, BCOO):
        matrix = matrix.todense()
    return np.asarray(matrix, dtype=np.float64)


def build_sparse_generator(
    matrix_a: BCOO,
    matrix_b: BCOO,
    state: jax.Array,
    sync_indices: tuple[int, int],
    coupling: jax.Array,
) -> jax.Array:
    """Apply the coupled block generator without materializing its blocks."""
    dim = matrix_a.shape[0]
    state_a, state_b = state[:dim], state[dim:]
    index_a, index_b = sync_indices
    error = coupling * (state_a[index_a] - state_b[index_b])
    derivative_b = (matrix_b @ state_b).at[index_b].add(error)
    return jnp.concatenate((matrix_a @ state_a, derivative_b))


def build_jax_generator(
    matrix_a: BCOO,
    matrix_b: BCOO,
    sync_indices: tuple[int, int],
    coupling: float,
) -> jax.Array:
    """Densify sparse blocks and assemble the combined JAX generator."""
    dense_a = matrix_a.todense()
    dense_b = matrix_b.todense()
    index_a, index_b = sync_indices
    lower_left = jnp.zeros_like(dense_a).at[index_b, index_a].set(coupling)
    lower_right = dense_b.at[index_b, index_b].add(-coupling)
    zeros = jnp.zeros_like(dense_a)
    return jnp.concatenate(
        (jnp.concatenate((dense_a, zeros), axis=1),
         jnp.concatenate((lower_left, lower_right), axis=1)),
        axis=0,
    )


@jax.jit
def _jax_expm_vmap(generator: jax.Array, initial_state: jax.Array, times: jax.Array) -> jax.Array:
    return jax.vmap(lambda time: expm(time * generator) @ initial_state)(times)


@partial(jax.jit, static_argnames=("sync_indices", "krylov_dim"))
def _jax_krylov_trajectory(
    matrix_a: BCOO,
    matrix_b: BCOO,
    initial_state: jax.Array,
    times: jax.Array,
    sync_indices: tuple[int, int],
    coupling: float,
    krylov_dim: int,
) -> jax.Array:
    def step(state: jax.Array, delta_time: jax.Array):
        next_state = expm_multiply_krylov(
            lambda value: build_sparse_generator(
                matrix_a, matrix_b, value, sync_indices, coupling
            ),
            state,
            delta_time,
            krylov_dim,
        )
        return next_state, next_state

    _, later_states = jax.lax.scan(step, initial_state, jnp.diff(times))
    return jnp.concatenate((initial_state[None, :], later_states), axis=0)


def solve_ode(
    matrix_a: BCOO,
    matrix_b: BCOO,
    initial_state: jax.Array,
    times: jax.Array,
    sync_indices: tuple[int, int],
    coupling: float,
    method: str = "scipy",
    krylov_dim: int = 30,
) -> jax.Array:
    """Solve the coupled linear ODE using a selectable exponential method."""
    if method == "scipy":
        generator = build_generator(matrix_a, matrix_b, sync_indices, coupling)
        time_values = np.asarray(times, dtype=np.float64)
        initial_values = np.asarray(initial_state)
        intervals = np.diff(time_values)
        if len(time_values) == 1:
            states = expm_multiply(generator * time_values[0], initial_values)[None, :]
        elif np.allclose(intervals, intervals[0]):
            states = expm_multiply(
                generator,
                initial_values,
                start=float(time_values[0]),
                stop=float(time_values[-1]),
                num=len(time_values),
                endpoint=True,
            )
        else:
            state = initial_values
            previous_time = 0.0
            sampled_states = []
            for time in time_values:
                state = expm_multiply(generator * (time - previous_time), state)
                sampled_states.append(state)
                previous_time = time
            states = np.stack(sampled_states)
        return jnp.asarray(states)
    if method == "jax_expm_vmap":
        dense_generator = build_jax_generator(
            matrix_a, matrix_b, sync_indices, coupling
        ).astype(initial_state.dtype)
        return _jax_expm_vmap(dense_generator, initial_state, times)
    if method == "jax_krylov":
        return _jax_krylov_trajectory(
            matrix_a,
            matrix_b,
            initial_state,
            times,
            sync_indices,
            coupling,
            krylov_dim,
        )
    raise ValueError(
        f"Unknown solver {method!r}; choose from scipy, jax_expm_vmap, jax_krylov"
    )


def solve_synchronization(
    key: jax.Array | np.ndarray | Iterable[int] | int,
    config: ExperimentConfig = DEFAULT_CONFIG,
    method: str = "scipy",
    krylov_dim: int = 30,
) -> SynchronizationSolution:
    """Generate and solve one synchronization system using the selected method."""
    matrix_a, matrix_b, initial, sync_indices, key_data = generate_system(key, config)
    times = jnp.linspace(0.0, config.t_end, config.n_times)
    states = solve_ode(
        matrix_a,
        matrix_b,
        initial,
        times,
        sync_indices,
        config.coupling,
        method,
        krylov_dim,
    )
    state_a, state_b = np.split(np.asarray(states), 2, axis=1)
    return SynchronizationSolution(
        key=key_data,
        ts=np.asarray(times),
        state_a=state_a,
        state_b=state_b,
        matrix_a=_as_dense_numpy(matrix_a),
        matrix_b=_as_dense_numpy(matrix_b),
        sync_indices=sync_indices,
        coupling=config.coupling,
        config_json=json.dumps(
            {**asdict(config), "solver": method, "krylov_dim": krylov_dim},
            sort_keys=True,
        ),
    )


def save_solution(
    solution: SynchronizationSolution,
    path: str | Path,
    seed: int,
) -> Path:
    """Save one solution as a portable compressed NumPy archive."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        key=solution.key,
        seed=np.asarray(seed, dtype=np.int64),
        ts=solution.ts,
        state_a=solution.state_a,
        state_b=solution.state_b,
        matrix_a=solution.matrix_a,
        matrix_b=solution.matrix_b,
        sync_indices=np.asarray(solution.sync_indices, dtype=np.int64),
        coupling=np.asarray(solution.coupling),
        config_json=np.asarray(solution.config_json),
    )
    return output_path


def run_pipeline(
    seeds: Iterable[int],
    output_dir: str | Path,
    config: ExperimentConfig = DEFAULT_CONFIG,
    method: str = "scipy",
    krylov_dim: int = 30,
) -> list[Path]:
    """Solve and save each supplied seed using the selected numerical method."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in seeds:
        solution = solve_synchronization(seed, config, method, krylov_dim)
        paths.append(
            save_solution(
                solution,
                destination / f"solution_seed_{seed}.npz",
                seed=seed,
            )
        )
    return paths