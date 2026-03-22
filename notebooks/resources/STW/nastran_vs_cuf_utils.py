"""
Shared constants, plotting configuration, and helper functions for the
STW Nastran-vs-CUF comparison notebooks.

Import this module at the top of each comparison notebook to avoid
repeating boilerplate code:

    from resources import stw_comparison_utils as cu
    cu.configure_plotting()
    stw_bdf, stw_layout = cu.create_stw_model()
"""

import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from resources import plot_utils
from resources import pynastran_utils
from resources.STW import wingGeometry


# ── Analysis Nastran IDs ──────────────────────────────────────────────────────

FIRST_SUBCASE_ID = 1    # ID of the static-load subcase in the case control deck
SECOND_SUBCASE_ID = 2   # ID of the eigenvalue subcase (SOL 105 only)
METHOD_SET_ID = 5        # ID of the eigenvalue method set (EIGRL card)
NO_EIGENVALUES = 20      # number of KLLRH eigenvalues computed per arc-length step
UPPER_EIGENVALUE = 1e5   # upper bound for the Lanczos solver [N/m]


# ── Analysis directory ────────────────────────────────────────────────────────

# Anchor the analysis directory to this file's location (notebooks/resources/)
# so the path is correct regardless of where Jupyter was started from.
_STW_DIR = os.path.dirname(os.path.abspath(__file__))  # notebooks/resources/STW/
_RESOURCES_DIR = os.path.dirname(_STW_DIR)  # notebooks/resources/
_NOTEBOOKS_ROOT = os.path.dirname(_RESOURCES_DIR)  # notebooks/

ANALYSIS_DIRECTORY_PATH = os.path.join(
    _NOTEBOOKS_ROOT, "analyses", "Simple_Transonic_Wing_Nastran_vs_CUF"
)

# ── Figure geometry ───────────────────────────────────────────────────────────

TEXTWIDTH_INCHES = 6.52437527778          # LaTeX textwidth [in]
FIG_WIDTH_INCHES = TEXTWIDTH_INCHES * 0.75
FIG_HEIGHT_INCHES = FIG_WIDTH_INCHES * (4.8 / 6.4)


# ── Colour / marker defaults ──────────────────────────────────────────────────

MARKERS = list(Line2D.markers.keys())[2:]
DEFAULT_MARKER_SIZE = 2.5
_all_colors = list(plot_utils.COLORS)       # tol-bright palette (do not mutate)
UNSTABLE_COLOR = _all_colors[1]             # red  – reserved for unstable branches
GLASS_CEILING_COLOR = _all_colors[2]        # green – linear buckling load line
COLORS = [c for i, c in enumerate(_all_colors) if i not in (1, 2)]


# ── Functions ─────────────────────────────────────────────────────────────────

def configure_plotting() -> None:
    """Apply shared matplotlib rcParams for all comparison notebooks."""
    plt.rcParams.update({
        "lines.markersize": DEFAULT_MARKER_SIZE,
        "legend.handlelength": 2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.dpi": 150,
    })


def adjust_3d_plot(axes) -> None:
    """Adjust tick density and label padding for 3-D wing-box plots."""
    axes.locator_params(axis="z", nbins=2)
    axes.tick_params(axis="x", which="major", pad=-1)
    axes.tick_params(axis="z", which="major", pad=-2)
    axes.xaxis.labelpad = -1
    axes.yaxis.labelpad = 8
    axes.zaxis.labelpad = -4


def read_cuf_data(filepath: str, max_disp_node_id: int = None) -> dict:
    """
    Read a CUF equilibrium-curve .dat file into a dictionary.

    The file is expected to contain a header line of the form::

        P[kN] w_tip[m] node <id> u[m] node <id> v[m] node <id> w[m] ...

    or with ``P[N]`` as the first column label.  Only the first five columns
    (P, w_tip, u, v, w) are returned.  The load column is always converted to
    Newtons regardless of the unit indicated in the header.

    Parameters
    ----------
    filepath : str
        Path to the CUF .dat file.
    max_disp_node_id : int, optional
        Node ID used for the local-displacement columns.  When *None* the ID
        is parsed automatically from the file header.

    Returns
    -------
    dict
        Keys: ``"P[N]"``, ``"w_tip[m]"``,
        ``f"node {node_id} u[m]"``, ``f"node {node_id} v[m]"``,
        ``f"node {node_id} w[m]"``, ``"node_id"``.
    """
    with open(filepath) as f:
        header = f.readline()

    load_in_kn = header.lstrip().startswith("P[kN]")

    if max_disp_node_id is None:
        match = re.search(r"node (\d+)", header)
        if match:
            max_disp_node_id = int(match.group(1))

    col_names = [
        "P[N]",
        "w_tip[m]",
        f"node {max_disp_node_id} u[m]",
        f"node {max_disp_node_id} v[m]",
        f"node {max_disp_node_id} w[m]",
    ]
    data = np.loadtxt(filepath, skiprows=2)
    result = {col_names[i]: data[:, i] for i in range(len(col_names))}
    if load_in_kn:
        result["P[N]"] = result["P[N]"] * 1e3
    result["node_id"] = max_disp_node_id
    return result


def save_equilibrium_path_csv(
    filepath: str,
    load: np.ndarray,
    tip_displacement: np.ndarray,
    u_z: np.ndarray,
    u_y: np.ndarray,
    local_node_id: int,
    local_node_coords,
) -> None:
    """
    Save a Nastran SOL 106 equilibrium path to a CSV file.

    The file has one metadata comment line followed by a header row with column
    names, then one data row per arc-length step.

    Parameters
    ----------
    filepath : str
        Output CSV file path.
    load : ndarray, shape (N,)
        Total applied load at each arc-length step [N].
    tip_displacement : ndarray, shape (N,)
        Wing-tip out-of-plane displacement at each step [m].
    u_z : ndarray, shape (N,)
        Out-of-plane (z) displacement at the local monitoring node [m].
    u_y : ndarray, shape (N,)
        In-plane (y) displacement at the local monitoring node [m].
    local_node_id : int
        Nastran ID of the local monitoring node.
    local_node_coords : array-like, length 3
        (x, y, z) coordinates of the local monitoring node [m].
    """
    # Extract coordinates
    x, y, z = local_node_coords
    
    # Define csv column names
    col_names = [
        "total_applied_load_N",
        "tip_displacement_m",
        f"u_z_node{local_node_id}_m",
        f"u_y_node{local_node_id}_m",
    ]
    
    # Stack data columns into a 2-D array for easier iteration
    data = np.column_stack([load, tip_displacement, u_z, u_y])
    
    # Write to CSV file
    with open(filepath, "w") as f:
        # Write metadata comment line with local node ID and coordinates
        f.write(
            f"# local node ID: {local_node_id}, "
            f"coordinates [m]: x={x:.6f}, y={y:.6f}, z={z:.6f}\n"
        )
        f.write(",".join(col_names) + "\n")
        
        # Iterate rows of 2-D data array and write formatted values
        for row in data:
            f.write(",".join(f"{v:.6e}" for v in row) + "\n")


def setup_sol_106_bdf(
    base_bdf,
    load_scale_factor: float,
    force_set_id: int,
    analysis_directory_path: str,
    conv: str = "PU",
    eps_p: float = 5e-6,
    eps_u: float = 5e-6,
    eps_w: float = 0.01,
    max_iter: int = 3,
    max_bisect: int = 20,
    minalr: float = 1e-5,
    desiter: int = 4,
    mxinc: int = 100,
):
    """
    Create a SOL 106 arc-length BDF from a base static-load BDF.

    Parameters
    ----------
    base_bdf : BDF
        BDF with the static-load subcase already configured (FORCE cards
        assigned to *force_set_id*).
    load_scale_factor : float
        Scale factor applied to the force set defined by force_set_id.
    force_set_id : int
        SID of the FORCE cards that define the reference unit load.
    analysis_directory_path : str
        Directory used by the KLLRH eigenvalue DMAP include statement.
    conv : str
        NLPARM convergence criterion (``"PU"``, ``"W"``, …).
    eps_p, eps_u, eps_w : float
        Load, displacement, and work convergence tolerances.
    max_iter : int
        Maximum number of Newton–Raphson iterations per increment.
    max_bisect : int
        Maximum number of bisections (load-step reductions) per increment.
    minalr : float
        Minimum arc-length ratio (NLPCI MINALR).
    desiter : int
        Desired number of iterations per increment (NLPCI DESITER).
    mxinc : int
        Maximum number of increments (NLPCI MXINC).

    Returns
    -------
    BDF
        Configured SOL 106 BDF ready to be passed to ``NastranAnalysis``.
    """
    sol_106_bdf = base_bdf.__deepcopy__({})

    load_set_id = force_set_id + 1
    sol_106_bdf.add_load(
        sid=load_set_id,
        scale=1.,
        scale_factors=[load_scale_factor],
        load_ids=[force_set_id],
    )
    sol_106_bdf.subcases[FIRST_SUBCASE_ID].params["LOAD"][0] = load_set_id

    pynastran_utils.set_up_arc_length_method(
        bdf=sol_106_bdf,
        max_iter=max_iter,
        conv=conv,
        eps_p=eps_p,
        eps_u=eps_u,
        eps_w=eps_w,
        max_bisect=max_bisect,
        minalr=minalr,
        desiter=desiter,
        maxinc=mxinc,
    )
    pynastran_utils.set_up_sol_106_with_kllrh_eigenvalues(
        bdf=sol_106_bdf,
        method_set_id=METHOD_SET_ID,
        analysis_directory_path=analysis_directory_path,
        no_eigenvalues=NO_EIGENVALUES,
        upper_eig=UPPER_EIGENVALUE,
    )
    sol_106_bdf.case_control_deck.subcases[FIRST_SUBCASE_ID].add_result_type(
        "NLSTRESS", "ALL", ["PLOT"]
    )
    return sol_106_bdf
