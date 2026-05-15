"""Minimal visualization of the sparse airflow estimation task.

The figure is designed to stay readable in a single IEEE column. It reuses the
central-obstacle scenario: sparse samples and the dense field are both drawn
from the existing CFD ground truth.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif"]
})


SAMPLE_COLOR = "#e60000"
SCENARIO_DIR = Path(__file__).resolve().parent / "10x6_labyrinth"
WIND_CSV = SCENARIO_DIR / "wind_gt.csv"
OCCUPANCY_PGM = SCENARIO_DIR / "geometry" / "occupancy.pgm"


def load_reference_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(WIND_CSV, delimiter=",", names=True)
    points = np.column_stack([data["Points0"], data["Points1"]])
    vectors = np.column_stack([data["U0"], data["U1"]])
    speed = np.linalg.norm(vectors, axis=1)
    return points, vectors, speed


def load_occupancy_mask() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    occupancy = np.array(Image.open(OCCUPANCY_PGM))
    occupied = np.flipud(occupancy == 0)
    resolution = 0.1
    origin_x, origin_y = -0.1, -0.1
    height, width = occupied.shape
    extent = (
        origin_x,
        origin_x + width * resolution,
        origin_y,
        origin_y + height * resolution,
    )
    return occupied, extent


def draw_occupancy(ax, occupied: np.ndarray, extent: tuple[float, float, float, float]) -> None:
    rgba = np.zeros((*occupied.shape, 4), dtype=float)
    rgba[occupied] = (0.72, 0.72, 0.72, 1.0)
    rgba[~occupied] = (1.0, 1.0, 1.0, 0.0)
    ax.imshow(
        rgba,
        extent=extent,
        origin="lower",
        interpolation="none",
        zorder=1,
    )
    ax.contour(
        occupied.astype(float),
        levels=[0.5],
        extent=extent,
        origin="lower",
        colors="0.05",
        linewidths=1.2,
        zorder=2,
    )


def sample_dense_vectors(
    reference_points: np.ndarray,
    reference_vectors: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    occupied: np.ndarray,
    extent: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(x_values, y_values)
    query = np.column_stack([xx.ravel(), yy.ravel()])
    tree = cKDTree(reference_points)
    _, nearest = tree.query(query, k=1)
    vectors = reference_vectors[nearest]

    xmin, xmax, ymin, ymax = extent
    rows = ((query[:, 1] - ymin) / (ymax - ymin) * occupied.shape[0]).astype(int)
    cols = ((query[:, 0] - xmin) / (xmax - xmin) * occupied.shape[1]).astype(int)
    rows = np.clip(rows, 0, occupied.shape[0] - 1)
    cols = np.clip(cols, 0, occupied.shape[1] - 1)
    valid = ~occupied[rows, cols]
    return query, vectors, valid


def dense_grid_field(
    reference_points: np.ndarray,
    reference_vectors: np.ndarray,
    occupied: np.ndarray,
    extent: tuple[float, float, float, float],
    nx: int = 220,
    ny: int = 130,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = extent
    x_values = np.linspace(0.0, 10.0, nx)
    y_values = np.linspace(0.0, 6.0, ny)
    xx, yy = np.meshgrid(x_values, y_values)
    query = np.column_stack([xx.ravel(), yy.ravel()])
    vectors = interpolate_vectors(reference_points, reference_vectors, query, k=5)

    rows = ((query[:, 1] - ymin) / (ymax - ymin) * occupied.shape[0]).astype(int)
    cols = ((query[:, 0] - xmin) / (xmax - xmin) * occupied.shape[1]).astype(int)
    rows = np.clip(rows, 0, occupied.shape[0] - 1)
    cols = np.clip(cols, 0, occupied.shape[1] - 1)
    outside = occupied[rows, cols].reshape(ny, nx)

    u = vectors[:, 0].reshape(ny, nx)
    v = vectors[:, 1].reshape(ny, nx)
    speed = np.sqrt(u**2 + v**2)
    return x_values, y_values, np.ma.array(u, mask=outside), np.ma.array(v, mask=outside), np.ma.array(speed, mask=outside)


def interpolate_vectors(
    reference_points: np.ndarray,
    reference_vectors: np.ndarray,
    query_points: np.ndarray,
    k: int = 6,
) -> np.ndarray:
    tree = cKDTree(reference_points)
    distances, nearest = tree.query(query_points, k=k)
    distances = np.maximum(distances, 1e-9)
    weights = 1.0 / distances**2
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ij,ijk->ik", weights, reference_vectors[nearest])


def sample_sparse_vectors(
    reference_points: np.ndarray,
    reference_vectors: np.ndarray,
    occupied: np.ndarray,
    extent: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    target_locations = np.array([
        [0.70, 1.70],
        [1.20, 4.10],
        [4.05, 5.35],
        [8.00, 0.65],
        [5.10, 3.05],
        [7.00, 1.00],
        [9.15, 5.00],
        [9.00, 3.00],
    ])
    tree = cKDTree(reference_points)
    _, nearest = tree.query(target_locations, k=1)
    points = target_locations
    vectors = reference_vectors[nearest]

    xmin, xmax, ymin, ymax = extent
    rows = ((points[:, 1] - ymin) / (ymax - ymin) * occupied.shape[0]).astype(int)
    cols = ((points[:, 0] - xmin) / (xmax - xmin) * occupied.shape[1]).astype(int)
    rows = np.clip(rows, 0, occupied.shape[0] - 1)
    cols = np.clip(cols, 0, occupied.shape[1] - 1)
    valid = ~occupied[rows, cols]
    return points[valid], vectors[valid]


def main() -> None:
    reference_points, reference_vectors, _ = load_reference_data()
    occupied, extent = load_occupancy_mask()
    sample_points, sample_vectors = sample_sparse_vectors(
        reference_points,
        reference_vectors,
        occupied,
        extent,
    )

    x_stream, y_stream, u_stream, v_stream, speed_stream = dense_grid_field(
        reference_points,
        reference_vectors,
        occupied,
        extent,
    )

    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.28))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.88, wspace=0.06)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(0.0, 6.0)
        ax.axis("off")

    draw_occupancy(axes[0], occupied, extent)
    axes[0].quiver(
        sample_points[:, 0],
        sample_points[:, 1],
        sample_vectors[:, 0],
        sample_vectors[:, 1],
        color=SAMPLE_COLOR,
        angles="xy",
        scale_units="xy",
        scale=1.8,
        width=0.013,
        headwidth=3.6,
        headlength=4,
        headaxislength=3.5,
        clip_on=True,
        zorder=5,
    )
    axes[0].scatter(
        sample_points[:, 0],
        sample_points[:, 1],
        s=4.0,
        facecolor=SAMPLE_COLOR,
        edgecolor="white",
        linewidth=0.1,
        zorder=6,
    )
    axes[0].set_title("Sparse Samples", fontsize=9, pad=3)

    axes[1].streamplot(
        x_stream,
        y_stream,
        u_stream,
        v_stream,
        color=speed_stream,
        cmap="Blues",
        density=1.15,
        linewidth=0.7,
        arrowsize=0.6,
        minlength=0.08,
        zorder=0,
    )
    draw_occupancy(axes[1], occupied, extent)
    axes[1].set_title("Dense Estimate", fontsize=9, pad=3)

    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "sparse_estimation_problem.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
