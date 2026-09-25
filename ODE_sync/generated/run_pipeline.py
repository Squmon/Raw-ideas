"""Run one synchronization system per Hydra job."""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import jax
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from synchronization import ExperimentConfig, save_solution, solve_synchronization


def _patch_python314_hydra_help() -> None:
    """Adapt Hydra's lazy completion help to argparse's string check."""
    expand_help = argparse.HelpFormatter._expand_help

    def expand_lazy_help(formatter: argparse.HelpFormatter, action: argparse.Action) -> str:
        if action.help.__class__.__name__ == "LazyCompletionHelp":
            action.help = repr(action.help)
        return expand_help(formatter, action)

    argparse.HelpFormatter._expand_help = expand_lazy_help


@hydra.main(version_base="1.3", config_path="conf", config_name="pipeline")
def main(config: DictConfig) -> None:
    """Solve and save one experiment; use Hydra multirun to sweep parameters."""
    experiment_config = ExperimentConfig(
        dim=config.dim,
        scale=config.scale,
        coupling=config.coupling,
        t_end=config.t_end,
        n_times=config.n_times,
        same_dynamics=config.same_dynamics,
        graph_generator=config.graph_generator,
        small_world_degree=config.small_world_degree,
        small_world_rewire_probability=config.small_world_rewire_probability,
    )
    seed = int(config.seed)
    solution = solve_synchronization(
        jax.random.PRNGKey(seed),
        experiment_config,
        method=config.solver,
        krylov_dim=config.krylov_dim,
    )

    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir

    job_number = OmegaConf.select(HydraConfig.get(), "job.num", default=None)
    suffix = f"_job_{job_number:05d}" if job_number is not None else ""
    output_path = output_dir / f"solution_seed_{seed}{suffix}.npz"
    save_solution(solution, output_path, seed=seed)
    print(f"Saved seed {seed} to {output_path}")


if __name__ == "__main__":
    _patch_python314_hydra_help()
    main()