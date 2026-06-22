import numpy as np
import pandas as pd

from mpi4py import MPI
from pathlib import Path
from scipy.spatial import cKDTree
from basix.ufl import element, mixed_element
from dolfinx import fem, mesh
import dolfinx.io as dio
from NS_wind_est.airflow_solvers import (
    AirflowSolverConfig,
    WfnsSolver,
    SfnsSolver
)

class AirflowMeasurements:

    def __init__(self, estimator: "AirflowEstimator"):
        self.estimator = estimator

    def set_from_csv(self,
                     samples_csv: str | Path,
                     count: int | None = None,
                     noise_std: float | None = None,
                     max_xy_dist: float | None = None) -> dict[str, float]:
        samples_csv = Path(samples_csv).resolve(strict=True)
        df = pd.read_csv(samples_csv)
        if count is not None:
            count = int(count)
            if count < 1:
                raise ValueError(f"count must be at least 1, got {count}.")
            if count > len(df):
                raise ValueError(
                    f"Requested {count} measurements from {samples_csv.name}, but file only contains {len(df)} rows."
                )
            df = df.iloc[:count].copy()

        required = ["Points:0", "Points:1", "U:0", "U:1"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in {samples_csv.name}: {missing}. "
                f"Expected at least {required}."
            )
        if len(df) == 0:
            raise ValueError(f"No sample rows found in {samples_csv}")

        samples_xy = df[["Points:0", "Points:1"]].to_numpy(dtype=float)
        samples_uv = df[["U:0", "U:1"]].to_numpy(dtype=float, copy=True)
        
        if noise_std is not None: 
            samples_uv += np.random.normal(0, noise_std, (len(df), 2))

        node_xy = np.asarray(self.estimator.V.tabulate_dof_coordinates(), dtype=float)[:, :2]
        tree = cKDTree(node_xy)
        dist, node_ids = tree.query(samples_xy, k=1, p=2.0, workers=-1)

        max_dist = float(np.max(dist))
        if max_xy_dist is not None and max_dist > max_xy_dist:
            raise ValueError(
                f"Maximum XY mapping distance exceeded: {max_dist:.6g} > {max_xy_dist:.6g}"
            )
        n_input = int(len(node_ids))
        _, first_idx = np.unique(node_ids, return_index=True)
        keep = np.sort(first_idx)
        n_dropped = n_input - int(len(keep))
        node_ids = node_ids[keep]
        samples_uv = samples_uv[keep]
        dist = dist[keep]

        x_ids = node_ids * 2
        y_ids = node_ids * 2 + 1
        velocity_ids_V = np.stack((x_ids, y_ids)).T.flatten().astype(np.int32)
        measurement_ids_W = self.estimator.V_to_W[velocity_ids_V]
        measurement_values = np.stack((samples_uv[:, 0], samples_uv[:, 1]), axis=1).flatten()

        self.estimator.set_measurements(
            measurement_ids_W=measurement_ids_W,
            measurement_values=measurement_values,
            clear_existing=True,
        )

        return {
            "n_input_samples": float(n_input),
            "n_used_samples": float(len(node_ids)),
            "n_dropped_duplicate_nodes": float(n_dropped),
            "max_xy_dist": float(np.max(dist)) if len(dist) else 0.0,
        }

    def coordinates(self) -> np.ndarray:
        coords_P2 = self.estimator.V.tabulate_dof_coordinates()
        W_to_V = {w: v for v, w in enumerate(self.estimator.V_to_W)}
        measured_v_ids = [W_to_V[i] for i in self.estimator.measurement_ids_W if i in W_to_V]
        measured_v_ids_unique = np.unique(np.array(measured_v_ids) // self.estimator.domain.geometry.dim)
        return coords_P2[measured_v_ids_unique]


class AirflowEstimator:

    def __init__(
        self,
        domain: mesh.Mesh,
        facet_tags: mesh.MeshTags | None = None,
    ):
        """Create an estimator for an existing DOLFINx mesh."""
        self.domain = domain
        self.facet_tags = facet_tags

        (
            self.W,
            self.W0,
            self.W1,
            self.V,
            self.Q,
            self.V_to_W,
            self.Q_to_W,
        ) = self.build_mixed_space(domain)

        self.w_measured = fem.Function(self.W)
        self.w_measured.x.array[:] = 0.0
        self.measurement_ids_W = np.array([], dtype=np.int32)

        self.domain.topology.create_connectivity(
            domain.topology.dim - 1, domain.topology.dim
        )

        self.viscosity = 1e-5
        self.weight_misfit = 1e2
        self.weight_pde_res = 1e0
        self.weight_reg = 1e-2
        self.weight_boundary = 1e0
        self.regularization_mode = "smooth"

        self.bcs: list[fem.DirichletBC] = []
        self.w_final: fem.Function | None = None
        self.last_solver_status: dict = {}
        self._boundary_name_to_id: dict[str, int] = {}
        self.measurements = AirflowMeasurements(self)

    @classmethod
    def from_mesh(
        cls,
        meshfile: str | Path,
        *,
        comm=MPI.COMM_WORLD,
        gdim: int = 2,
    ) -> "AirflowEstimator":
        """Create an estimator directly from a Gmsh ``.msh`` file."""
        meshfile = Path(meshfile).resolve(strict=True)
        domain, _, facet_tags = dio.gmshio.read_from_msh(
            str(meshfile), comm, gdim=gdim
        )
        return cls.from_domain(
            domain,
            facet_tags,
            meshfile=meshfile,
        )

    @classmethod
    def from_domain(
        cls,
        domain: mesh.Mesh,
        facet_tags: mesh.MeshTags | None = None,
        meshfile: str | Path | None = None,
    ) -> "AirflowEstimator":
        """Create an estimator from an existing DOLFINx mesh."""
        estimator = cls(domain, facet_tags)

        if meshfile is not None:
            estimator._boundary_name_to_id = cls._read_physical_name_map(meshfile)

        return estimator

    def _ensure_boundary_name_map(self):
        if self.facet_tags is None:
            raise RuntimeError("facet_tags is not set.")
        if not self._boundary_name_to_id:
            raise RuntimeError(
                "No boundary name->id mapping available. "
                "Provide meshfile during construction or set _boundary_name_to_id manually."
            )

    def set_no_slip_bc(self, wall_names: str | list[str]):
        """
        Apply no-slip (u=0) boundary condition on the given physical boundaries.

        Parameters
        ----------
        wall_names : str or list[str]
            Physical group names, e.g. 'Walls' or ['Walls', 'Obstacles'].
        """
        self._ensure_boundary_name_map()

        if isinstance(wall_names, str):
            names = [wall_names]
        else:
            names = list(wall_names)

        facets = np.concatenate([
            self.facet_tags.find(self._boundary_name_to_id[name]) for name in names
        ])

        u_D = fem.Function(self.V)
        u_D.x.array[:] = 0.0

        dofs = fem.locate_dofs_topological((self.W0, self.V),
                                           self.domain.topology.dim - 1,
                                           facets)
        bc = fem.dirichletbc(u_D, dofs, self.W0)
        self.add_dirichlet_bc(bc)
        return bc

    def set_zero_pressure_bc(self, outlet_names: str | list[str]):
        """
        Apply p=0 boundary condition on the given outlet boundaries.

        Parameters
        ----------
        outlet_names : str or list[str]
            Physical group names, e.g. 'Outflow'.
        """
        self._ensure_boundary_name_map()

        if isinstance(outlet_names, str):
            names = [outlet_names]
        else:
            names = list(outlet_names)

        facets = np.concatenate([
            self.facet_tags.find(self._boundary_name_to_id[name]) for name in names
        ])

        p_zero = fem.Function(self.Q)
        p_zero.x.array[:] = 0.0

        dofs = fem.locate_dofs_topological((self.W1, self.Q),
                                           self.domain.topology.dim - 1,
                                           facets)
        bc = fem.dirichletbc(p_zero, dofs, self.W1)
        self.add_dirichlet_bc(bc)
        return bc
        

    @staticmethod
    def build_mixed_space(domain, deg_u=2, deg_p=1):
        """Erzeugt das gemischte (velocity-pressure) Funktionsraumtuple."""
        elem_u = element("Lagrange", domain.basix_cell(), deg_u, shape=(domain.geometry.dim,))
        elem_p = element("Lagrange", domain.basix_cell(), deg_p)
        mixed_elem = mixed_element([elem_u, elem_p])

        W = fem.functionspace(domain, mixed_elem)
        W0, W1 = W.sub(0), W.sub(1)
        V, V_to_W = W0.collapse()
        Q, Q_to_W = W1.collapse()
        return W, W0, W1, V, Q, np.array(V_to_W, dtype=np.int32), np.array(Q_to_W, dtype=np.int32)

    @staticmethod
    def _read_physical_name_map(meshfile: Path) -> dict[str, int]:
        import gmsh

        meshfile = Path(meshfile).resolve(strict=True)
        gmsh.initialize()
        try:
            gmsh.open(str(meshfile))
            groups = gmsh.model.getPhysicalGroups()
            return {gmsh.model.getPhysicalName(dim, tag): tag for (dim, tag) in groups}
        finally:
            gmsh.finalize()

    @staticmethod
    def _num_dofs(space) -> int:
        return space.dofmap.index_map.size_global * space.dofmap.index_map_bs


    @staticmethod
    def normalize_regularization_mode(mode: str) -> str:
        mode_norm = mode.strip().lower()
        if mode_norm not in {"smooth", "value"}:
            raise ValueError("regularization mode must be 'smooth' or 'value'.")
        return mode_norm

    def set_regularization(self, mode: str):
        self.regularization_mode = self.normalize_regularization_mode(mode)

    def _build_solver_context(self) -> AirflowSolverConfig:
        return AirflowSolverConfig(
            domain=self.domain,
            W=self.W,
            V=self.V,
            V_to_W=self.V_to_W,
            bcs=self.bcs,
            w_measured=self.w_measured,
            measurement_ids_W=self.measurement_ids_W,
            viscosity=self.viscosity,
            weight_misfit=self.weight_misfit,
            weight_pde_res=self.weight_pde_res,
            weight_reg=self.weight_reg,
            weight_boundary=self.weight_boundary,
            regularization_mode=self.regularization_mode,
        )

    def solve_SFNS(self,
                               maxit: int = 10,
                               solver_tol: float = 1e-2,
                               damping: float | None = None,
                               regularization: str | None = None,
                               verbose: bool = False):
        solver = SfnsSolver(self._build_solver_context())
        result = solver.solve(
            maxit=maxit,
            solver_tol=solver_tol,
            damping=damping,
            regularization=regularization,
            verbose=verbose,
        )
        self.last_solver_status = solver.last_status
        self.w_final = result
        return result

    def solve_WFNS(self,
                                   maxit: int = 10,
                                   solver_tol: float = 1e-3,
                                   damping: float | None = None,
                                   regularization: str | None = None,
                                   verbose: bool = False):
        solver = WfnsSolver(self._build_solver_context())
        result = solver.solve(
            maxit=maxit,
            solver_tol=solver_tol,
            damping=damping,
            regularization=regularization,
            verbose=verbose,
        )
        self.last_solver_status = solver.last_status
        self.w_final = result
        return result

    def add_dirichlet_bc(self, bc: fem.DirichletBC | list[fem.DirichletBC]):
        if isinstance(bc, list):
            self.bcs += bc
        else:
            self.bcs.append(bc)

    def set_measurements(
        self,
        measurement_ids_W: np.ndarray,
        measurement_values: np.ndarray,
        clear_existing: bool = True,
    ):
        """
        Set explicit wind measurements in mixed space W.

        Parameters
        ----------
        measurement_ids_W : np.ndarray
            Flattened W-indices for velocity components.
        measurement_values : np.ndarray
            Flattened measurement values aligned with measurement_ids_W.
        clear_existing : bool
            If True, clear all previous measurements first.
        """
        ids = np.asarray(measurement_ids_W, dtype=np.int32).reshape(-1)
        values = np.asarray(measurement_values, dtype=float).reshape(-1)

        if ids.size == 0:
            raise ValueError("measurement_ids_W is empty.")
        if ids.size != values.size:
            raise ValueError(
                f"Length mismatch: len(ids)={ids.size} != len(values)={values.size}"
            )
        if np.any(ids < 0) or np.any(ids >= self.w_measured.x.array.size):
            raise ValueError("measurement_ids_W contains out-of-bounds indices.")

        if clear_existing:
            self.w_measured.x.array[:] = 0.0

        self.w_measured.x.array[ids] = values
        self.measurement_ids_W = ids
        self.w_final = None

    def set_measurements_from_csv(
        self,
        samples_csv: str | Path,
        count: int | None = None,
        noise_std: float | None = None,
        max_xy_dist: float | None = None,
    ) -> dict[str, float]:
        """
        Load wind samples from CSV and map them to nearest velocity nodes.

        Expected CSV columns: Points:0, Points:1, U:0, U:1.

        Parameters
        ----------
        samples_csv : str | Path
            Path to sample CSV.
        count : int | None
            Optional number of rows to import from the top of the CSV. If omitted, all rows are used.
        noise_std : float | None
            Optional standard deviation for gaussian noise level to be added to the measurements.
        max_xy_dist : float | None
            Optional maximum allowed nearest-neighbor mapping distance in XY.

        Returns
        -------
        dict[str, float]
            Mapping stats (input/used samples, dropped duplicates, max distance).
        """
        return self.measurements.set_from_csv(
            samples_csv=samples_csv,
            count=count,
            noise_std=noise_std,
            max_xy_dist=max_xy_dist,
        )

    def set_weights(self, kin_v: float | None = None, 
                          misfit: float | None = None, 
                          pde_err: float | None = None, 
                          reg: float | None = None,
                          boundary: float | None = None):
        if kin_v is not None:
            self.viscosity = kin_v
        if misfit is not None:
            self.weight_misfit = misfit
        if pde_err is not None:
            self.weight_pde_res = pde_err
        if reg is not None:
            self.weight_reg = reg
        if boundary is not None:
            self.weight_boundary = boundary
        
    def get_measurement_coordinates(self) -> np.ndarray:
        """Rekonstruiere Messpunkt-Koordinaten aus measurement_ids_W."""
        return self.measurements.coordinates()
