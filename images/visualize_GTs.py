from pathlib import Path

import dolfinx.io as dio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from basix.ufl import element
from dolfinx import fem
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from mpi4py import MPI
from scipy.spatial import cKDTree

from tools.csv_utilities import csv_to_function

SLICE_HEIGHT = 0.025
Z_TOL = 1e-1
PLOT_MODE = 'streamplot'
OUTPUT_PATH = Path(f'/app/wind_gt_10x6_all_{PLOT_MODE}.png')
OUTPUT_PATH_PDF = OUTPUT_PATH.with_suffix('.pdf')
SHOW_BOUNDARY_LEGEND = False
COLORBAR_VMAX = 4.0
INFLOW_COLOR = 'limegreen'
OUTFLOW_COLOR = 'red'

CASE_STYLES = {
    '10x6 Appartment': {
        51: {'label': 'Walls', 'color': 'black', 'linewidth': 1.3},
        52: {'label': 'Inflow', 'color': INFLOW_COLOR, 'linewidth': 8.0},
        53: {'label': 'Inflow', 'color': INFLOW_COLOR, 'linewidth': 8.0},
        54: {'label': 'Outflow', 'color': OUTFLOW_COLOR, 'linewidth': 8.0},
        55: {'label': 'Outflow', 'color': OUTFLOW_COLOR, 'linewidth': 8.0},
    },
    '10x6 Labyrinth': {
        31: {'label': 'Walls', 'color': 'black', 'linewidth': 1.3},
        32: {'label': 'Inflow', 'color': INFLOW_COLOR, 'linewidth': 8.0},
        33: {'label': 'Inflow', 'color': INFLOW_COLOR, 'linewidth': 8.0},
        34: {'label': 'Outflow', 'color': OUTFLOW_COLOR, 'linewidth': 8.0},
    },
    '10x6 Rectangular Obstacles': {
        184: {'label': 'Walls', 'color': 'black', 'linewidth': 1.3},
        185: {'label': 'Inflow', 'color': INFLOW_COLOR, 'linewidth': 8.0},
        186: {'label': 'Outflow', 'color': OUTFLOW_COLOR, 'linewidth': 8.0},
    },
}

CASES = [
    {
        'key': '10x6 Appartment',
        'title': 'Scenario A: Apartment',
        'msh_path': Path('/app/scenarios/10x6_appartment/geometry/appartment_2d.msh'),
        'csv_path': Path('/app/scenarios/10x6_appartment/wind_gt.csv'),
    },
    {
        'key': '10x6 Labyrinth',
        'title': 'Scenario B: Corridor',
        'msh_path': Path('/app/scenarios/10x6_labyrinth/geometry/labyrinth_2d.msh'),
        'csv_path': Path('/app/scenarios/10x6_labyrinth/wind_gt.csv'),
    },
    {
        'key': '10x6 Rectangular Obstacles',
        'title': 'Scenario C: Obstacles',
        'msh_path': Path('/app/scenarios/10x6_multiple_obstacles/geometry/multiple_obstacles_rect_2d.msh'),
        'csv_path': Path('/app/scenarios/10x6_multiple_obstacles/wind_gt.csv'),
    },
]


def load_case(case: dict) -> dict:
    msh_path = case['msh_path']
    csv_path = case['csv_path']
    domain, _, facet_tags = dio.gmshio.read_from_msh(str(msh_path), MPI.COMM_WORLD, gdim=2)
    elem_u = element('Lagrange', domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    V = fem.functionspace(domain, elem_u)

    velocity = fem.Function(V)
    csv_to_function(csv_path, SLICE_HEIGHT, Z_TOL, velocity)

    values = velocity.x.array.reshape(-1, V.dofmap.index_map_bs)[:, :2]
    return {
        **case,
        'domain': domain,
        'function_space': V,
        'velocity': velocity,
        'facet_tags': facet_tags,
        'speed': np.linalg.norm(values, axis=1),
    }


def add_boundary_facets(ax, domain, facet_tags, tag_styles: dict[int, dict]) -> None:
    domain.topology.create_connectivity(1, 0)
    f2v = domain.topology.connectivity(1, 0)
    coords = domain.geometry.x[:, :2]

    for tag in np.unique(facet_tags.values):
        facets = facet_tags.indices[facet_tags.values == tag]
        if len(facets) == 0:
            continue

        style = tag_styles.get(int(tag), {})
        color = style.get('color', 'black')
        linewidth = style.get('linewidth', 1.6)
        segments = []
        for facet in facets:
            vertices = f2v.links(int(facet))
            if len(vertices) == 2:
                segments.append(coords[np.asarray(vertices, dtype=np.int32)])

        if segments:
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=color,
                    linewidths=linewidth,
                    zorder=4,
                )
            )


def plot_case(ax, case: dict, norm: Normalize):
    V = case['function_space']
    velocity = case['velocity']
    coords = V.tabulate_dof_coordinates()[:, :2]
    values = velocity.x.array.reshape(-1, V.dofmap.index_map_bs)[:, :2]
    speed = np.linalg.norm(values, axis=1)

    quiver = ax.quiver(
        coords[:, 0],
        coords[:, 1],
        values[:, 0],
        values[:, 1],
        speed,
        cmap='coolwarm',
        norm=norm,
        angles='xy',
        scale_units='xy',
        scale=None,
        width=0.0022,
        pivot='tail',
    )
    add_boundary_facets(
        ax,
        case['domain'],
        case['facet_tags'],
        CASE_STYLES.get(case['key'], {}),
    )

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(case['title'])
    return quiver


def streamplot_case(ax, case: dict, norm: Normalize):
    domain = case['domain']
    V = case['function_space']
    velocity = case['velocity']

    domain.topology.create_connectivity(domain.topology.dim, 0)
    conn = domain.topology.connectivity(domain.topology.dim, 0).array
    cells = conn.reshape(len(conn) // 3, 3)
    points = domain.geometry.x[:, :2]

    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    xg = np.linspace(x_min, x_max, 260)
    yg = np.linspace(y_min, y_max, 160)
    xx, yy = np.meshgrid(xg, yg)
    query_xy = np.column_stack([xx.ravel(), yy.ravel()])

    tri = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    inside = tri.get_trifinder()(query_xy[:, 0], query_xy[:, 1]) >= 0

    dof_xy = V.tabulate_dof_coordinates()[:, :2]
    dof_uv = velocity.x.array.reshape(-1, V.dofmap.index_map_bs)[:, :2]
    tree = cKDTree(dof_xy)
    _, nn = tree.query(query_xy[inside], k=1, workers=-1)

    u = np.full(query_xy.shape[0], np.nan)
    v = np.full(query_xy.shape[0], np.nan)
    u[inside] = dof_uv[nn, 0]
    v[inside] = dof_uv[nn, 1]

    u = u.reshape(xx.shape)
    v = v.reshape(xx.shape)
    speed = np.sqrt(u**2 + v**2)
    outside = ~inside.reshape(xx.shape)

    stream = ax.streamplot(
        xg,
        yg,
        np.ma.array(u, mask=outside),
        np.ma.array(v, mask=outside),
        color=np.ma.array(speed, mask=outside),
        cmap='coolwarm',
        norm=norm,
        density=1.8,
        linewidth=0.8,
        arrowsize=0.7,
        minlength=0.14,
    )
    add_boundary_facets(
        ax,
        domain,
        case['facet_tags'],
        CASE_STYLES.get(case['key'], {}),
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(case['title'])
    return stream.lines


def boundary_legend_handles() -> list[Line2D]:
    legend_styles = {
        'Walls': {'color': 'black', 'linewidth': 1.8},
        'Inflow': {'color': 'tab:green', 'linewidth': 8.0},
        'Outflow': {'color': 'tab:red', 'linewidth': 8.0},
    }
    return [
        Line2D([0], [0], label=label, **style)
        for label, style in legend_styles.items()
    ]


def add_shared_axis_labels(fig, axes, xlabel: str, ylabel: str) -> Bbox:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    plot_area = Bbox.union([ax.get_position() for ax in axes])
    labeled_area = Bbox.union(
        [
            ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
            for ax in axes
        ]
    )

    pad_px = 8
    x_pad = pad_px / (fig.get_size_inches()[0] * fig.dpi)
    y_pad = pad_px / (fig.get_size_inches()[1] * fig.dpi)

    fig.text(
        plot_area.x0 + 0.5 * plot_area.width,
        labeled_area.y0 - y_pad,
        xlabel,
        ha='center',
        va='top',
    )
    fig.text(
        labeled_area.x0 - x_pad,
        plot_area.y0 + 0.5 * plot_area.height,
        ylabel,
        ha='right',
        va='center',
        rotation='vertical',
    )
    return plot_area


def main() -> None:
    cases = [load_case(case) for case in CASES]
    norm = Normalize(
        vmin=0.0,
        vmax=COLORBAR_VMAX,
        clip=True,
    )

    fig, axes = plt.subplots(1, len(cases), figsize=(14, 3.8), dpi=300)
    mappable = None
    for ax, case in zip(axes, cases):
        if PLOT_MODE == 'streamplot':
            mappable = streamplot_case(ax, case, norm)
        else:
            mappable = plot_case(ax, case, norm)

    bottom = 0.14 if SHOW_BOUNDARY_LEGEND else 0.02
    fig.tight_layout(rect=(0.0, bottom, 0.92, 1.0))
    cbar = fig.colorbar(
        mappable,
        ax=axes,
        pad=0.015,
        shrink=0.6,
        label='Velocity magnitude (m/s)',
    )
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plot_area = add_shared_axis_labels(fig, axes, 'x (m)', 'y (m)')
    if SHOW_BOUNDARY_LEGEND:
        fig.legend(
            handles=boundary_legend_handles(),
            loc='lower center',
            ncol=3,
            frameon=False,
            bbox_to_anchor=(plot_area.x0 + 0.5 * plot_area.width, 0.01),
        )
    fig.savefig(OUTPUT_PATH, bbox_inches='tight', pad_inches=0.01)
    fig.savefig(OUTPUT_PATH_PDF, bbox_inches='tight', pad_inches=0.01)
    print(f'Saved figure: {OUTPUT_PATH}')
    print(f'Saved figure: {OUTPUT_PATH_PDF}')


if __name__ == '__main__':
    main()
