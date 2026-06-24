# NS wind estimation

This repository contains a tool for two-dimensional steady-state airflow field
estimation from sparse velocity measurements with a Navier-Stokes based
estimator. The current focus is the isolated strong-form Navier-Stokes workflow
(`SFNS`): build or load a mesh, place a small number of wind measurements, solve
for the full velocity-pressure field, and write the estimated wind field for
inspection.

There are two intended entry points:

1. a scenario workflow for reproducible example cases, and
2. a direct Python API through `AirflowEstimator`, which is the more flexible
   entry point for integration into other projects.

## Installation

The recommended setup is the Docker/dev-container environment included in this
repository. It is based on `dolfinx/dolfinx:v0.9.0` and installs the Python
packages from `requirements.txt` during the image build.

With VS Code, open the repository and run:

```text
Dev Containers: Reopen in Container
```

The dev-container configuration lives in `.devcontainer/dolfinx0.9/` and uses
the repository `compose.yml`/`Dockerfile`.

Alternatively, build and start the container from a terminal:

```bash
docker compose up --build
```

Running outside the container is possible, but you need a DOLFINx-capable Python
environment with MPI, PETSc, Gmsh, NumPy, SciPy, pandas, PyYAML, and matplotlib.
In that case, install the pure Python dependencies with:

```bash
pip install -r requirements.txt
```

## Scenario workflow

The scenarios in `example_cases/` are example containers for reproducible
experiments. They follow a predefined configuration workflow through
`scenario.yaml`. This makes the scenario workflow convenient, but also quite
specific: it is meant for cases that can be described by the scenario
configuration, not as the most general way to use the estimator.

A scenario contains:

- a `.msh` file with the domain mesh and its Gmsh physical groups, for example
  `Walls` or `Outlet`. These physical groups allow us to assign boundary
  conditions by name in the workflow;
- sample CSV files that represent sparse airflow measurements in space. These
  files need to follow the column structure expected by the measurement reader;
- optionally, a ground-truth wind file such as `wind_gt.csv`. This file is not
  needed for the estimation itself, but it can be used with
  `scripts/generate_csv_samples.py` to draw random measurement/sample sets;
- a `scenario.yaml` file that manages paths, measurement settings, boundary
  matching rules, solver parameters, and output locations.

Run all sample CSVs configured for the apartment example:

```bash
python scripts/run_ns_wind_scenario.py example_cases/10x6_appartment
```

Run one explicit sample set:

```bash
python scripts/run_ns_wind_scenario.py   example_cases/10x6_appartment   --samples example_cases/10x6_appartment/samples/sample_points_n400_seed0.csv
```

The same workflow is available from Python:

```python
from NS_wind_est.scenario_runner import run_scenario

result = run_scenario("example_cases/10x6_appartment", verbose=True)
print(result.output_root)
print(result.result_dirs)
```

Outputs are written below the scenario `result_dir`, usually as
`<solver>-<timestamp>/...`. Each run writes:

- `wind_estimate.csv` with velocity values on the estimator velocity nodes,
- `wind_estimate.png` for a quick visual check,
- `metadata_wind_est.json` with mapping and solver metadata.

## Direct `AirflowEstimator` API

Use the class API when you want to build experiments in Python instead of going
through a YAML scenario. A minimal workflow looks like this:

```python
from NS_wind_est import AirflowEstimator

estimator = AirflowEstimator.from_mesh(
    "example_cases/10x6_appartment/geometry/appartment_2d_coarse.msh"
)

estimator.set_no_slip_bc("Walls")
estimator.set_zero_pressure_bc("Outlet")

estimator.set_measurements_from_csv(
    "example_cases/10x6_appartment/samples/sample_points_n400_seed0.csv",
    count=25,
    noise_std=0.1,
    max_xy_dist=0.2,
)

estimator.set_regularization("gradient")
estimator.set_weights(weight_misfit=10.0, weight_pde_res=1.0, weight_reg=1.0e-2)

result = estimator.solve_SFNS(maxit=25, solver_tol=1.0e-2, damping=0.5)

velocity = result.velocity
pressure = result.pressure
status = result.status
```

A runnable version of this example is available in
`scripts/example_airflow_estimator.py`.

## Mesh generation and boundary names

Meshes can be created, for example, in Gmsh. For the convenient named-boundary
workflow, the `.msh` file should contain Gmsh physical groups with assigned
names for relevant boundaries.

`AirflowEstimator.from_mesh(...)` reads these physical names and exposes them as
boundary names:

```python
print(estimator.boundary_names)
estimator.set_no_slip_bc("Walls")
estimator.set_zero_pressure_bc("Outlet")
```

The scenario workflow can also find boundary names by regular expression. In a
`scenario.yaml`, this looks like:

```yaml
geometry:
  wall_pattern: wall|obstacle
  outflow_pattern: outlet|outflow
```

## Ground-truth wind fields

A ground-truth wind field is not required for estimation. It is useful when you
want to assess accuracy or generate synthetic sparse measurements. Such files
can be produced with an external CFD tool such as OpenFOAM.

## Sample CSV files

Sample CSV files can be generated from a ground-truth wind file with:

```bash
python scripts/generate_csv_samples.py ...
```

They can then be loaded into the estimator with `set_measurements_from_csv(...)`.
This CSV interface expects at least these columns:

```text
Points:0, Points:1, U:0, U:1
```

Further columns such as `Points:2` or `U:2` may exist, but are ignored by the
current 2D estimator. Measurements are mapped to the nearest velocity node in
the estimator mesh. If multiple input samples map to the same node, the first
one is kept and the duplicates are counted in the returned mapping metadata.

For robotic deployment or other applications where measurements are already
available in memory, the CSV interface is not mandatory. Use
`set_measurements(...)` directly when you already know the mixed-space
measurement indices and values.

## Solvers

`SFNS` is the main solver path for the current repository direction. `WFNS`
(weak-form Navier-Stokes) is its predecessor and is still available for
experiments/comparison. The WFNS implementation is adapted from:
https://doi.org/10.1177/02783649251329421.
