# Linear Synchronization Experiments

The scripts solve the notebook's one-way coupled deterministic linear system.
For state vectors `a` and `b`, the equations are `da/dt = A a` and
`db/dt = B b + k(a[i] - b[j]) e_j`. The random keys select graph matrices and
initial states; the differential equation itself has no stochastic term.

## Solver and graph choices

`A` and `B` are represented internally as JAX BCOO sparse matrices. The
`graph_generator` option currently accepts `current_random_antisymmetric`,
which preserves the existing thresholded random antisymmetric matrix
construction, and `small_world`, which builds a Watts-Strogatz graph and uses
its scaled negative Laplacian as the linear dynamics matrix. The small-world
generator accepts `small_world_degree` (an even integer less than `dim`) and
`small_world_rewire_probability` (between 0 and 1). Both are seeded from the
system key. New generator functions can be registered in `synchronization.py`.

The `solver` option selects among three approaches:

- `scipy`: reference implementation using SciPy `expm_multiply`.
- `jax_expm_vmap`: JIT-compiled JAX dense matrix exponential, vmapped over
  output times. It densifies the combined block generator and computes one
  full matrix exponential per sample, so it can use more memory than the other
  options.
- `jax_krylov`: JIT-compiled Arnoldi exponential action, scanning across time
  intervals and applying BCOO matrix-vector products without building the
  combined dense generator. It is an approximation controlled by
  `krylov_dim`.

JAX dispatches to available devices, including CUDA GPUs. The environment used
for validation reports a CUDA device, but performance depends on graph size,
solver, and compilation cost; benchmark for the target workload before
choosing a default. The default remains `scipy` for a trusted comparison.

## Generate systems

Use the science virtual environment:

```bash
/home/squmon/.PythonEnvs/science/bin/python ODE_sync/generated/run_pipeline.py \
  seed=42 dim=10 t_end=500 n_times=500 \
  solver=jax_krylov krylov_dim=20 \
  graph_generator=small_world small_world_degree=4 \
  small_world_rewire_probability=0.1
```

Each run solves one system and is saved under `generated/results/` with its
seed in both the filename and the NPZ metadata. Files also contain the random
key, time samples, both state trajectories, system matrices, coupling indices,
coupling strength, and serialized experiment configuration. Repeated
seed/configuration pairs produce the same keys and solutions.

Use Hydra multirun to iterate over seeds or other parameters. For example:

```bash
/home/squmon/.PythonEnvs/science/bin/python ODE_sync/generated/run_pipeline.py \
  -m seed=11,12,13 dim=10,20 coupling=1.0,3.0 \
  solver=scipy,jax_expm_vmap,jax_krylov
```

Each multirun job writes one archive. Its filename includes the seed and Hydra
job number, preventing systems with different parameter combinations from
overwriting one another. Other parameters can be set or swept using the same
Hydra override syntax. Set `same_dynamics=false` to generate a separate matrix
for system `b`.

Run the focused solver tests with:

```bash
/home/squmon/.PythonEnvs/science/bin/python -m unittest discover \
  -s ODE_sync/generated -p 'test_*.py'
```

## Inspect results

```bash
/home/squmon/.PythonEnvs/science/bin/python ODE_sync/generated/show_results.py \
  ODE_sync/generated/results
```

This generates `batch_table.html` and a sibling `batch_table_details/`
directory. The index is a clickable Plotly heatmap with random seeds on the
x-axis, dimensions on the y-axis, and final synchronization MSE encoded by the
color scale. Clicking a square opens that system's state, hub, error, and
difference plots. The pages are static HTML and can be opened directly in a
browser. When multiple runs share a seed/dimension pair (for example, a sweep
over coupling strength), the cell shows their mean final MSE and its detail
page contains plots for each matching run.

Detail pages retain component line plots through dimension 15. Above 15
dimensions, the `a`, `b`, and `a - b` panels switch to time-by-component
heatmaps with a diverging color ramp; hub traces and synchronization-error
plots remain unchanged.