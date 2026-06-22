import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dolfinx import fem
import dolfinx.io as dio
from mpi4py import MPI

from NS_wind_est.airflow_estimator import AirflowEstimator
from NS_wind_est.scenario import ScenarioConfig



@dataclass(frozen=True)
class ScenarioRunResult:
    output_root: Path
    result_dirs: tuple[Path, ...]

    @property
    def n_runs(self) -> int:
        return len(self.result_dirs)


def match_boundary_names(name_to_id: dict[str, int], pattern: str) -> list[str]:
    regex = re.compile(pattern, re.IGNORECASE)
    return [name for name in name_to_id if regex.search(name)]


def save_velocity_csv(path: Path, velocity: fem.Function) -> None:
    coords = velocity.function_space.tabulate_dof_coordinates()
    values = velocity.x.array.reshape(-1, velocity.function_space.dofmap.bs)

    z = coords[:, 2] if coords.shape[1] > 2 else np.zeros(coords.shape[0])
    w = values[:, 2] if values.shape[1] > 2 else np.zeros(values.shape[0])
    data = np.column_stack([coords[:, 0], coords[:, 1], z, values[:, 0], values[:, 1], w])
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="Points:0,Points:1,Points:2,U:0,U:1,U:2",
        comments="",
    )


def create_estimator(config: ScenarioConfig) -> AirflowEstimator:
    domain, _, facet_tags = dio.gmshio.read_from_msh(str(config.mesh), MPI.COMM_WORLD, gdim=2)
    estimator = AirflowEstimator.from_domain(domain, facet_tags, meshfile=config.mesh)

    wall_names = match_boundary_names(estimator._boundary_name_to_id, config.wall_pattern)
    if wall_names:
        estimator.set_no_slip_bc(wall_names)

    outflow_names = match_boundary_names(estimator._boundary_name_to_id, config.outflow_pattern)
    if not outflow_names:
        raise ValueError(
            f"No outflow boundaries matched pattern {config.outflow_pattern!r} in {config.mesh.name}."
        )
    estimator.set_zero_pressure_bc(outflow_names)

    estimator.set_regularization(config.regularization)
    estimator.set_weights(
        kin_v=config.viscosity,
        misfit=config.weight_misfit,
        pde_err=config.weight_pde_res,
        reg=config.weight_reg,
        boundary=config.weight_boundary,
    )
    return estimator

def write_outputs(result_dir: Path, velocity: fem.Function, metadata: dict) -> None:
    from NS_wind_est.visualizer import plot_wind_csv

    estimate_path = result_dir / "wind_estimate.csv"
    plot_path = result_dir / "wind_estimate.png"
    metadata_path = result_dir / "metadata_wind_est.json"

    save_velocity_csv(estimate_path, velocity)
    plot_wind_csv(
        estimate_path,
        output_path=plot_path,
        title="NS wind estimate",
        show=False,
    )
    metadata["wind_estimate_csv"] = str(estimate_path)
    metadata["wind_estimate_png"] = str(plot_path)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def solve_estimator(estimator: AirflowEstimator, config: ScenarioConfig):
    solver_name = config.solver.strip().upper()
    if solver_name == "SFNS":
        return estimator.solve_SFNS(
            maxit=config.maxit,
            solver_tol=config.solver_tol,
            damping=config.damping,
            regularization=config.regularization,
            verbose=False,
        )
    if solver_name == "WFNS":
        return estimator.solve_WFNS(
            maxit=config.maxit,
            solver_tol=config.solver_tol,
            damping=config.damping,
            regularization=config.regularization,
            verbose=False,
        )
    raise ValueError(
        f"Unsupported solver {config.solver}, use one of: SFNS, WFNS."
    )


def run_case(
    config: ScenarioConfig,
    estimator: AirflowEstimator,
    sample_csv: Path,
    result_dir: Path,
    sample_size: int | None,
    verbose: bool,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)

    mapping_info = estimator.set_measurements_from_csv(
        sample_csv,
        count=sample_size,
        noise_std=config.add_gaussian_noise_std,
        max_xy_dist=config.max_xy_dist,
    )
    estimation_start = time.perf_counter()
    result = solve_estimator(estimator, config)
    estimation_runtime_sec = time.perf_counter() - estimation_start
    solver_status = getattr(estimator, "last_solver_status", {})
    status_text = "converged" if solver_status.get("converged") else "reached max iterations"
    iterations = solver_status.get("iterations", "?")
    final_change = solver_status.get("final_relative_change")
    change_text = "nan" if final_change is None else f"{float(final_change):.3e}"
    if verbose:
        print(
            f"NS solver {status_text}"
            f"({sample_size if sample_size is not None else 'all'} samples): "
            f"iterations={iterations}/{config.maxit}, \nfinal_relative_change={change_text},\n solver_tol={config.solver_tol:.3e}"
        )
    u_est = result.sub(0).collapse()

    metadata = {
        "scenario": config.name,
        "estimator": config.solver,
        "sample_name": sample_csv.stem,
        "sample_size": sample_size if sample_size is not None else int(mapping_info["n_input_samples"]),
        "samples_csv": str(sample_csv),
        "mesh": str(config.mesh),
        "wind_csv": str(config.wind_csv),
        "add_gaussian_noise_std": str(config.add_gaussian_noise_std),
        "solver": config.solver,
        "regularization": config.regularization,
        "maxit": config.maxit,
        "solver_tol": config.solver_tol,
        "damping": config.damping,
        "viscosity": config.viscosity,
        "weight_misfit": config.weight_misfit,
        "weight_pde_res": config.weight_pde_res,
        "weight_reg": config.weight_reg,
        "weight_boundary": config.weight_boundary,
        "n_discretization_points": int(u_est.function_space.tabulate_dof_coordinates().shape[0]),
        "estimation_runtime_sec": float(estimation_runtime_sec),
        "solver_converged": bool(solver_status.get("converged", False)),
        "solver_iterations": int(solver_status.get("iterations", 0)),
        "solver_max_iterations": int(solver_status.get("max_iterations", config.maxit)),
        "solver_final_relative_change": float(solver_status.get("final_relative_change", float("nan"))),
        **mapping_info,
    }

    write_outputs(result_dir, u_est, metadata)

    if verbose:
        print("---")



def run_scenario(
    scenario: str | Path | ScenarioConfig,
    samples: str | Path | None = None,
    *,
    all_samples: bool = False,
    verbose: bool = False,
) -> ScenarioRunResult:
    """Run one configured scenario for one sample CSV or all scenario samples."""
    if samples is not None and all_samples:
        raise ValueError("Pass either samples or all_samples=True, not both.")
    if samples is None and not all_samples:
        raise ValueError("Pass a sample CSV or set all_samples=True.")

    config = scenario if isinstance(scenario, ScenarioConfig) else ScenarioConfig.load(scenario)
    if samples is not None:
        sample_files = [Path(samples).resolve(strict=True)]
    else:
        sample_files = sorted(config.sample_dir.glob("sample_points*.csv"))
        if not sample_files:
            raise FileNotFoundError(f"No sample_points*.csv files found in {config.sample_dir}")

    sample_sizes = config.wind_measurement_counts or (None,)
    timestamp = time.strftime("%Y%m%d_%H%M")
    output_root = config.result_dir / f"{config.solver}-{timestamp}"
    estimator = create_estimator(config)
    result_dirs: list[Path] = []

    for sample_size in sample_sizes:
        for sample_csv in sample_files:
            if verbose:
                label = f" with {sample_size} samples" if sample_size is not None else ""
                print(f"Running {config.solver} for {sample_csv.name}{label}")
            result_dir = output_root
            if len(sample_sizes) > 1 and sample_size is not None:
                result_dir = result_dir / f"{sample_size}samples"
            if len(sample_files) > 1:
                result_dir = result_dir / sample_csv.stem
            run_case(config, estimator, sample_csv, result_dir, sample_size, verbose)
            result_dirs.append(result_dir)

    return ScenarioRunResult(output_root=output_root, result_dirs=tuple(result_dirs))
