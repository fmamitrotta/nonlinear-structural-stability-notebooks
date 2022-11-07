import os
import time
from pyNastran.bdf.bdf import BDF
from pyNastran.utils.nastran_utils import run_nastran
from pyNastran.op2.op2 import OP2
import numpy as np
import re
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
from numpy import ndarray
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.pyplot import Figure


def wait_nastran(directory_path: str):
    """
    Suspends execution of current thread from the start to the end of a Nastran run.

    Parameters
    ----------
    directory_path : str
        string with path to the directory where input file is run
    """
    # Wait for the analysis to start (when bat file appears)
    while not any([file.endswith('.bat') for file in os.listdir(directory_path)]):
        time.sleep(0.1)
    # Wait for the analysis to finish (when bat file disappears)
    while any([file.endswith('.bat') for file in os.listdir(directory_path)]):
        time.sleep(0.1)


def run_analysis(directory_path: str, bdf_object: BDF, bdf_filename: str, run_flag: bool = True):
    """
    Write .bdf input file from BDF object and execute Nastran analysis.

    Parameters
    ----------
    directory_path : str
        string with path to the directory where input file is run
    bdf_object: BDF
        pyNastran object representing the bdf input file
    bdf_filename: str
        name of the bdf input file
    run_flag: bool
        flag to enable or disable the actual execution of Nastran
    """
    # Create analysis directory
    os.makedirs(directory_path, exist_ok=True)
    # Write bdf file
    bdf_filepath = os.path.join(directory_path, bdf_filename + '.bdf')
    bdf_object.write_bdf(bdf_filepath)
    # Run Nastran
    nastran_path = 'C:\\Program Files\\MSC.Software\\MSC_Nastran\\2021.4\\bin\\nastranw.exe'
    run_nastran(bdf_filename=bdf_filepath, nastran_cmd=nastran_path, run_in_bdf_dir=True, run=run_flag)
    # If Nastran is actually executed wait until analysis is done
    if run_flag:
        wait_nastran(directory_path)


def create_static_load_subcase(bdf_object: BDF, subcase_id: int, load_set_id: int):
    """
    Define a subcase in the input BDF object for the application of a static load.

    Parameters
    ----------
    bdf_object : BDF
        pyNastran object representing Nastran input file
    subcase_id: int
        id of the subcase
    load_set_id: int
        id of the load set assigned to the subcase
    """
    # Create subcase
    bdf_object.create_subcases(subcase_id)
    # Add load set id to case control statement of created subcase
    bdf_object.case_control_deck.subcases[subcase_id].add_integer_type('LOAD', load_set_id)


def read_load_displacement_history_from_op2(op2_object: OP2, displacement_node_id: int = 1) -> \
        Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
    """
    Read history of total applied load and displacements at the indicated node from a OP2 object.

    Parameters
    ----------
    op2_object : OP2
        pyNastran object including the results of a nonlinear analysis (SOL 106)
    displacement_node_id: int
        id of the node where the displacements are read

    Returns
    -------
    load_steps: dict
        dictionary with a vector of the load steps for each subcase
    displacements: dict
        dictionary with subcase ids as keys and arrays of the xyz displacements at the indicated node as values
    loads: dict
        dictionary with subcase ids as keys and arrays of the applied loads as values
    """
    # Initialize dictionaries where the quantities of interest will be saved
    load_steps = {}
    displacements = {}
    loads = {}
    # Iterate through the subcases found in the op2 file
    valid_subcase_ids = [subcase_id for subcase_id in op2_object.load_vectors if
                         hasattr(op2_object.load_vectors[subcase_id], 'lftsfqs')]
    for subcase_id in valid_subcase_ids:
        # Save load steps of current subcase
        load_steps[subcase_id] = op2_object.load_vectors[subcase_id].lftsfqs
        # Save loads summation of current subcase
        loads[subcase_id] = np.apply_along_axis(np.sum, 1, op2_object.load_vectors[subcase_id].data[:, :, 0:3])
        # Save displacements of indicated node id and current subcase
        displacements[subcase_id] = op2_object.displacements[subcase_id].data[:, displacement_node_id - 1, :]
    # Return output data
    return load_steps, loads, displacements


def read_nonlinear_buckling_load_from_f06(f06_filepath: str, op2_object: OP2) -> Tuple[ndarray, ndarray]:
    """
    Return nonlinear buckling loads and critical buckling factors by reading the .f06 and .op2 files including the
    results of the analyis with the nonlinear buckling method.

    Parameters
    ----------
    f06_filepath: str
        string with path to .f06 file
    op2_object: OP2
        pyNastran object including the results of a nonlinear analysis (SOL 106) with nonlinear buckling method

    Returns
    -------
    nonlinear_buckling_load: ndarray
        array of nonlinear buckling loads calculated as  P + Delta P * alpha
    alphas: ndarray
        array of critical buckling factors, used to verify that absolute value is not greater than unity
    """
    # Read critical buckling factor ALPHA for each subcase
    alphas = []  # empty list of critical buckling factors
    regexp = re.compile('-? *[0-9]+.?[0-9]*(?:[Ee] *[-+]? *[0-9]+)?')  # compiled regular expression pattern
    with open(f06_filepath) as f06_file:
        for line in f06_file:
            if 'ALPHA' in line:
                alphas.append(float(re.findall(regexp, line)[0]))
    # Find valid subcases in the OP2 object (subcases with list of load steps)
    valid_subcase_ids = [subcase_id for subcase_id in op2_object.load_vectors if
                         hasattr(op2_object.load_vectors[subcase_id], 'lftsfqs')]
    # Initialize final load of previous subcase
    final_load_previous_subcase = 0
    # Initialize list of nonlinear buckling loads
    nonlinear_buckling_loads = np.empty(len(alphas))
    # Iterate through the valid subcases
    for i, subcase_id in enumerate(valid_subcase_ids):
        # Find final applied load of current subcase
        final_load = np.linalg.norm(
            np.apply_along_axis(np.sum, 0, op2_object.load_vectors[subcase_id].data[-1, :, 0:3]))
        # Find incremental load applied in current subcase with respect to previous subcase
        applied_load = final_load - final_load_previous_subcase
        # Store load steps of current subcase
        load_steps = op2_object.load_vectors[subcase_id].lftsfqs
        # Find last load increment of current subcase
        last_load_increment = applied_load * (load_steps[-1] - load_steps[-2])
        # Calculate nonlinear buckling load as P+DeltaP*ALPHA
        nonlinear_buckling_loads[i] = final_load + last_load_increment * alphas[i]
        # Update final load of previous subcase
        final_load_previous_subcase = final_load
    # Return lists of nonlinear buckling loads and critical buckling factors
    return nonlinear_buckling_loads, np.array(alphas)


def read_kllrh_lowest_eigenvalues_from_f06(f06_filepath: str) -> ndarray:
    """
    Return a list with the lowest eigenvalue of the matrix KLLRH (tangent stiffness matrix) for each load increment
    reading a f06 file resulting from a nonlinear analysis run with a proper DMAP.

    Parameters
    ----------
        f06_filepath: str
            string with path to .f06 file

    Returns
    -------
        lowest_eigenvalues: ndarray
            array with the lowest eigenvalue of the KLLRH matrices
    """
    # Initialize list of the lowest eigenvalues
    lowest_eigenvalues = []
    # Compile a regular expression pattern to read eigenvalue in f06 file
    regexp = re.compile('-? *[0-9]+.?[0-9]*(?:[Ee] *[-+]? *[0-9]+)?')
    # Open file and look for matching pattern line by line
    with open(f06_filepath) as f06_file:
        for line in f06_file:
            # When matching pattern is found, read lowest eigenvalue
            if 'KLLRH LOWEST EIGENVALUE' in line:
                lowest_eigenvalues.append(float(re.findall(regexp, line)[0]))
    # Return list of the lowest eigenvalues
    return np.array(lowest_eigenvalues)


def plot_displacements(op2_object: OP2, displacement_data: ndarray) -> Tuple[Figure, Axes3D]:
    """
    Plot the deformed shape coloured by displacement magnitude based on input OP2 object and displacement data.

    Parameters
    ----------
    op2_object: OP2
        pyNastran object created reading an op2 file with the load_geometry option set to True
    displacement_data: ndarray
        array with displacement data to plot

    Returns
    -------
        fig: Figure
            object of the plotted figure
        ax: Axes3D
            object of the plot's axes
    """
    # Create figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Initialize list of displacement magnitude and vertices
    displacements = np.empty(len(op2_object.elements))
    vertices = np.empty((len(op2_object.elements), 4, 3))
    # Define factor to scale displacements
    displacement_scale_factor = 200
    # Iterate through the elements of the structure
    for count, element in enumerate(op2_object.elements.values()):
        # Store ids of the nodes as an array
        node_ids = np.array(element.node_ids)
        # Take the average displacement magnitude over the nodes of each element
        displacements[count] = np.mean(np.apply_along_axis(
            np.linalg.norm, 1, displacement_data[-1, node_ids - 1, 0:3]))
        # Store the coordinates of the nodes of the deformed elements
        vertices[count] = np.vstack(
            [op2_object.nodes[index].xyz + displacement_data[-1, index - 1, 0:3] *
             displacement_scale_factor for index in node_ids])
    # Create 3D polygons to represent the elements
    pc = Poly3DCollection(vertices, linewidths=.05)
    # Create colormap for the displacement magnitude and normalize values
    m = mpl.cm.ScalarMappable(cmap=mpl.cm.jet)
    m.set_array(displacements)
    m.set_clim(vmin=0, vmax=1)
    rgba_array = m.to_rgba(displacements)
    # Color the elements' face by the average displacement magnitude
    pc.set_facecolor([(rgb[0], rgb[1], rgb[2]) for rgb in rgba_array])
    # Set the edge color black
    pc.set_edgecolor('k')
    # Add polygons to the plot
    ax.add_collection3d(pc)
    # Set axes label
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('z [mm]')
    # Set axes limits
    x_coordinates = [func(points[:, 0]) for points in vertices for func in (np.min, np.max)]
    y_coordinates = [func(points[:, 1]) for points in vertices for func in (np.min, np.max)]
    z_coordinates = [func(points[:, 2]) for points in vertices for func in (np.min, np.max)]
    ax.set_xlim(min(x_coordinates), max(x_coordinates))
    ax.set_ylim(min(y_coordinates), max(y_coordinates))
    ax.set_zlim(min(z_coordinates), max(z_coordinates))
    # Set aspect ratio of the axes
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    # Add colorbar
    cb = fig.colorbar(mappable=m, label='Nondimensional displacement magnitude', pad=0.15)
    # Return axes object
    return fig, ax


def plot_buckling_mode(op2_object: OP2, subcase_id: [int, tuple]) -> Tuple[Figure, Axes3D]:
    """
    Plot the buckling shape using the eigenvectors of the input OP2 object.

    Parameters
    ----------
        op2_object: OP2
            pyNastran object created reading an op2 file with the load_geometry option set to True
        subcase_id: int, tuple
            key of the eigenvectors' dictionary in the OP2 object corresponding to the selected subcase

    Returns
    -------
        fig: Figure
            object of the plotted figure
        ax: Axes3D
            object of the plot's axes
    """
    # Choose eigenvectors as displacement data
    displacement_data = op2_object.eigenvectors[subcase_id].data
    # Call plotting function
    fig, ax = plot_displacements(op2_object, displacement_data)
    # Return axes object
    return fig, ax


def plot_static_deformation(op2_object: OP2, subcase_id: [int, tuple] = 1) -> Tuple[Figure, Axes3D]:
    """
    Plot the buckling shape using the eigenvectors of the input OP2 object.

    Parameters
    ----------
        op2_object: OP2
            pyNastran object created reading an op2 file with the load_geometry option set to True
        subcase_id: int, tuple
            key of the displacements' dictionary in the OP2 object corresponding to the selected subcase

    Returns
    -------
        fig: Figure
            object of the plotted figure
        ax: Axes3D
            object of the plot's axes
    """
    # Choose static displacements as displacement data
    displacement_data = op2_object.displacements[subcase_id].data
    # Call plotting function
    fig, ax = plot_displacements(op2_object, displacement_data)
    # Return axes object
    return fig, ax
