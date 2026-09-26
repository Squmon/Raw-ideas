python ODE_sync/generated/run_pipeline.py -m \
  'seed=range(0,10)' 'dim=range(10,100,10)' coupling=3.0 \
  solver=jax_krylov t_end=1000 n_times=1000 output_dir=solutions/random_antisymmetric graph_generator=current_random_antisymmetric

python ODE_sync/generated/run_pipeline.py -m \
  'seed=range(0,10)' 'dim=range(10,100,10)' coupling=3.0 \
  solver=jax_krylov t_end=1000 n_times=1000 output_dir=solutions/scale_free_antisymmetric graph_generator=scale_free_antisymmetric

python ODE_sync/generated/run_pipeline.py -m \
  'seed=range(0,10)' 'dim=range(10,100,10)' coupling=3.0 \
  solver=jax_krylov t_end=1000 n_times=1000 output_dir=solutions/small_world_laplacian graph_generator=small_world_laplacian

python ODE_sync/generated/run_pipeline.py -m \
  'seed=range(0,6)' 'dim=range(10, 20, 2)' coupling=0.0 \
  solver=jax_krylov t_end=1000 n_times=1000 output_dir=solutions/zero_coupling_random_antisymmetric graph_generator=current_random_antisymmetric

python ODE_sync/generated/show_results.py \
  ODE_sync/generated/solutions/random_antisymmetric

python ODE_sync/generated/show_results.py \
  ODE_sync/generated/solutions/scale_free_antisymmetric

python ODE_sync/generated/show_results.py \
  ODE_sync/generated/solutions/small_world_laplacian

python ODE_sync/generated/show_results.py \
  ODE_sync/generated/solutions/zero_coupling_random_antisymmetric