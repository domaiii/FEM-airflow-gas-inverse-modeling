from pathlib import Path

import dolfinx.io as dio
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from mpi4py import MPI
from scipy.interpolate import griddata

SCENARIO = Path("/app/scenarios/10x6_appartment")
MESH = SCENARIO / "geometry/appartment_2d.msh"
SAMPLES = SCENARIO / "samples/sample_points_n400_seed0.csv"
RESULTS = SCENARIO / "results/NSRM"
RUN = "sample_points_n400_seed0"
OUT = Path("/app/paper_plots/estimation_increasing_samples_sequence.pdf")

SAMPLE_COLOR = "#FFFFFFFF"


def add_boundaries(ax, domain, tags):
    domain.topology.create_connectivity(1, 0)
    f2v = domain.topology.connectivity(1, 0)
    coords = domain.geometry.x[:, :2]

    for tag in np.unique(tags.values):
        segments = []
        for facet in tags.indices[tags.values == tag]:
            vertices = f2v.links(int(facet))
            if len(vertices) == 2:
                segments.append(coords[np.asarray(vertices, dtype=np.int32)])

        if not segments:
            continue

        wall = int(tag) == 51
        ax.add_collection(LineCollection(
            segments,
            colors="black" if wall else "white",
            linewidths=1.3 if wall else 1.3,
            zorder=2,
        ))


def inside_mesh(domain, X, Y):
    domain.topology.create_connectivity(domain.topology.dim, 0)
    conn = domain.topology.connectivity(domain.topology.dim, 0).array
    cells = conn.reshape(len(conn) // 3, 3)
    points = domain.geometry.x[:, :2]
    tri = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    inside = tri.get_trifinder()(X.ravel(), Y.ravel()) >= 0
    return inside.reshape(X.shape)


def field_on_grid(csv_path, X, Y, inside, gt=False):
    df = pd.read_csv(csv_path)

    if gt:
        points = df[["Points:0", "Points:1"]].to_numpy(float)
        vectors = df[["U:0", "U:1"]].to_numpy(float)
    else:
        points = df[["x", "y"]].to_numpy(float)
        vectors = df[["wind_x", "wind_y"]].to_numpy(float)

    U = griddata(points, vectors[:, 0], (X, Y), method="linear")
    V = griddata(points, vectors[:, 1], (X, Y), method="linear")
    mask = (~inside) | np.isnan(U) | np.isnan(V)

    U = np.ma.array(U, mask=mask)
    V = np.ma.array(V, mask=mask)
    speed = np.ma.array(np.sqrt(U**2 + V**2), mask=mask)
    return U, V, speed


def draw_panel(ax, title, U, V, speed, xg, yg, domain, tags, samples=None):
    stream = ax.streamplot(
        xg, yg, U, V,
        color=speed,
        cmap="coolwarm",
        density=1.2,
        linewidth=1.0,
        arrowsize=0.4,
        minlength=0.12,
        zorder=1,
    )

    add_boundaries(ax, domain, tags)

    if samples is not None:
        
        ax.scatter(
            samples["x"],
            samples["y"],
            marker="*",
            s=72,                    # Fläche statt markersize
            facecolors="#000000",
            edgecolors="#FFFFFF",
            linewidths=0.6,
            zorder=1,
        )

    ax.set_title(title, fontsize=10, pad=3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xg.min(), xg.max())
    ax.set_ylim(yg.min(), yg.max())
    ax.axis("off")
    return stream


def main():
    domain, _, tags = dio.gmshio.read_from_msh(str(MESH), MPI.COMM_WORLD, gdim=2)
    mesh_xy = domain.geometry.x[:, :2]

    xg = np.linspace(mesh_xy[:, 0].min(), mesh_xy[:, 0].max(), 260)
    yg = np.linspace(mesh_xy[:, 1].min(), mesh_xy[:, 1].max(), 160)
    X, Y = np.meshgrid(xg, yg)
    inside = inside_mesh(domain, X, Y)
    sample_df = pd.read_csv(SAMPLES)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, dpi=300, figsize=(4.2,3.7),layout="constrained")

    # 10
    k = 10
    
    csv = RESULTS / f"{k}samples" / RUN / "wind_estimate.csv"
    U, V, speed = field_on_grid(csv, X, Y, inside)
    stream = draw_panel(ax1, f"Estimate: K = {k}", 
                        U, V, speed, xg, yg, domain, tags, sample_df.iloc[:k])
    
    # 25
    k = 25
    csv = RESULTS / f"{k}samples" / RUN / "wind_estimate.csv"
    U, V, speed = field_on_grid(csv, X, Y, inside)
    stream = draw_panel(ax2, f"Estimate: K = {k}", 
                        U, V, speed, xg, yg, domain, tags, sample_df.iloc[:k])

    # 50
    k = 50
    csv = RESULTS / f"{k}samples" / RUN / "wind_estimate.csv"
    U, V, speed = field_on_grid(csv, X, Y, inside)
    stream = draw_panel(ax3, f"Estimate: K = {k}", 
                        U, V, speed, xg, yg, domain, tags, sample_df.iloc[:k])

    # GT
    U, V, speed = field_on_grid(SCENARIO / "wind_gt.csv", X, Y, inside, gt=True)
    stream = draw_panel(
        ax4,
        "Ground Truth",
        U, V, speed,
        xg, yg,
        domain, tags,
    )

    fig.subplots_adjust(wspace=0.3)
    axs = [ax1, ax2, ax3, ax4]
    cb = fig.colorbar(stream.lines, ax=axs,location="bottom", shrink=0.6, aspect=25)
    cb.set_label("Velocity magnitude (m/s)", fontsize=9)
    cb.ax.tick_params(labelsize=9)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    main()