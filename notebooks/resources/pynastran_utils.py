"""
This file is part of the GitHub repository nonlinear-structural-stability-notebooks, created by Francesco M. A. Mitrotta.
Copyright (C) 2023 Francesco Mario Antonio Mitrotta

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.pyplot import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from numpy import ndarray
import os
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2, read_op2
from pyNastran.utils.nastran_utils import run_nastran
import re
from typing import Tuple, Dict, Any, Union
import tol_colors as tc

# Register Tol's color-blind friendly colormaps
plt.cm.register_cmap("sunset", tc.tol_cmap("sunset"))
plt.cm.register_cmap("rainbow_PuRd", tc.tol_cmap("rainbow_PuRd"))


def run_analysis(directory_path: str, bdf_object: BDF, filename: str, run_flag: bool = True):
    """
    Write .bdf input file from BDF object and execute Nastran analysis.

    Parameters
    ----------
    directory_path : str
        string with path to the directory where input file is run
    bdf_object: BDF
        pyNastran object representing the bdf input file
    filename: str
        name of the input file
    run_flag: bool
        flag to enable or disable the actual execution of Nastran
    """
    # Create analysis directory
    os.makedirs(directory_path, exist_ok=True)
    # Write bdf file
    bdf_filename = filename + '.bdf'
    bdf_filepath = os.path.join(directory_path, bdf_filename)
    bdf_object.write_bdf(bdf_filepath)
    # Run Nastran
    nastran_path = 'C:\\Program Files\\MSC.Software\\MSC_Nastran\\2021.4\\bin\\nastran.exe'
    run_nastran(bdf_filename=bdf_filepath, nastran_cmd=nastran_path, run_in_bdf_dir=True, run=run_flag)
    # Read and print wall time of simulation
    log_filepath = os.path.join(directory_path, filename + '.log')
    regexp = re.compile('-? *[0-9]+.?[0-9]*(?:[Ee] *[-+]? *[0-9]+)?')  # compiled regular expression pattern
    with open(log_filepath) as log_file:
        for line in log_file:
            if 'Total' in line:
                wall_time = float(re.findall(regexp, line)[1])
                print(f'Nastran job {bdf_filename} completed\nWall time: {wall_time:.1f} s')
                break


def create_static_load_subcase(bdf_object: BDF, subcase_id: int, load_set_id: int, nlparm_id: int = None):
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
    nlparm_id: int
        id of the NLPARM card assigned to the subcase
    """
    # Create subcase
    bdf_object.create_subcases(subcase_id)
    # Add load set id to case control statement of created subcase
    bdf_object.case_control_deck.subcases[subcase_id].add_integer_type('LOAD', load_set_id)
    # If provided, add NLPARM id to case control statement of created subcase
    if nlparm_id:
        bdf_object.case_control_deck.subcases[subcase_id].add_integer_type('NLPARM', nlparm_id)


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
        node_index = np.where(op2_object.displacements[subcase_id].node_gridtype[:, 0] == displacement_node_id)[0][0]
        displacements[subcase_id] = op2_object.displacements[subcase_id].data[:, node_index, :]
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
        # Calculate the magnitude of the total applied laod at end of current subcase
        final_load = np.linalg.norm(np.apply_along_axis(np.sum, 0, op2_object.load_vectors[subcase_id].data[-1, :, 0:3]))
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
    # Initialize the list of the lowest eigenvalues
    eigenvalue_list = []
    # Compile a regular expression pattern to read eigenvalue in f06 file
    regexp = re.compile('-? *[0-9]+.?[0-9]*(?:[Ee] *[-+]? *[0-9]+)?')
    # Open file iterate line by line
    with open(f06_filepath) as f06_file:
        for line in f06_file:
            # When NLOOP is found append an empty list inside the eigenvalues list
            if 'NLOOP =' in line:
                eigenvalue_list.append([])
            # When KLLRH EIGENVALUE is found append the eigenvalue to the last list of the eigenvalues list
            if 'KLLRH EIGENVALUE' in line:
                raw_results = re.findall(regexp, line)
                eigenvalue_list[-1].append(float(raw_results[1]))
    # Convert list of lists to array padded with nan (we put nan for the iterations where we miss one or more eigenvalues)
    lengths = np.array([len(item) for item in eigenvalue_list])
    mask = lengths[:, None] > np.arange(lengths.max())
    eigenvalue_array = np.full(mask.shape, np.nan)
    eigenvalue_array[mask] = np.concatenate(eigenvalue_list)
    # Return array of eigenvalues in the form: number of eigenvalues x number of iterations
    return eigenvalue_array.T


def plot_displacements(op2_object: OP2, displacement_data: ndarray, node_ids: ndarray, displacement_component: str = "magnitude",
                       displacement_unit_scale_factor: float = 1., coordinate_unit_scale_factor:float =1.,
                       displacement_amplification_factor: float = 1., colormap: str = "rainbow_PuRd", clim: Union[list, ndarray] = None,
                       length_unit: str = "mm") -> Tuple[Figure, Axes3D, ScalarMappable]:
    """
    Plot the deformed shape coloured by displacement magnitude based on input OP2 object and displacement data.

    Parameters
    ----------
    op2_object: OP2
        pyNastran object created reading an op2 file with the load_geometry option set to True
    displacement_data: ndarray
        array with displacement data to plot
    node_ids: ndarray
        array with nodes' identification numbers
    displacement_component: str
        name of the displacement component used for the colorbar
    displacement_unit_scale_factor: float
        unit scale factor for displacements
    coordinate_unit_scale_factor: float
        unit scale factor for nodes coordinates
    displacement_amplification_factor: float
        amplification factor for displacements
    colormap: str
        name of the colormap used for the displacement colorbar
    clim: Union[list, ndarray]:
        colorbar values limits
    length_unit: str
        measurement unit of coordinates and displacements, used in the label of the axes

    Returns
    -------
    fig: Figure
        object of the plotted figure
    ax: Axes3D
        object of the plot's axes
    m: ScalarMappable
        mappable object for the colorbar
    """
    # Dictionary mapping displacement component names to indices of displacement data array
    component_dict = {'tx': 0,
                      'ty': 1,
                      'tz': 2,
                      'rx': 3,
                      'ry': 4,
                      'rz': 5,
                      'magnitude': np.array([0, 1, 2])[:, np.newaxis]}
    # Create figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Initialize array of displacements and vertices
    no_elements = len(op2_object.elements)
    if displacement_component == 'magnitude':
        nodes_displacement = np.empty((no_elements, 3, 4))  # 3 components for each node to calculate magnitude
    else:
        nodes_displacement = np.empty((no_elements, 4))  # 1 component for each node
    vertices = np.empty((no_elements, 4, 3))
    # Iterate through the elements of the structure
    for count, element in enumerate(op2_object.elements.values()):
        # Store ids of the nodes as an array
        element_node_ids = np.array(element.node_ids)
        # Find indexes of the element node ids into the global node ids array
        node_indexes = np.nonzero(np.in1d(node_ids, element_node_ids))[0]
        # Find the displacement of the nodes along the indicated component
        nodes_disp_component = displacement_data[node_indexes, component_dict[displacement_component]]
        # Apply unit conversion to displacements and not to rotations
        if displacement_component not in ["rx", "ry", "rz"]:
            nodes_disp_component *= displacement_unit_scale_factor
        # Store the displacements in the appropriate array
        nodes_displacement[count] = nodes_disp_component
        # Store the coordinates of the nodes of the deformed elements
        vertices[count] = np.vstack([
            op2_object.nodes[node_id].xyz * coordinate_unit_scale_factor +
            displacement_data[np.where(node_ids == node_id)[0], 0:3] * displacement_unit_scale_factor * displacement_amplification_factor
            for node_id in element_node_ids
        ])
    # Calculate displacement magnitude if requested
    if displacement_component == 'magnitude':
        nodes_displacement = np.linalg.norm(nodes_displacement, axis=1)
    # Calculate average displacement for each element
    elements_mean_displacement = np.mean(nodes_displacement, axis=1)
    # Create 3D polygons to represent the elements
    pc = Poly3DCollection(vertices, linewidths=.05)
    # Create colormap for the displacement magnitude
    m = ScalarMappable(cmap=colormap)
    m.set_array(elements_mean_displacement)
    # Set colormap min and max values and displacement values to colors
    if clim is None:
        m.set_clim(vmin=np.amin(nodes_displacement), vmax=np.amax(nodes_displacement))
    else:
        m.set_clim(vmin=clim[0], vmax=clim[1])
    rgba_array = m.to_rgba(elements_mean_displacement)
    # Color the elements' face by the average displacement magnitude
    pc.set_facecolor([(rgb[0], rgb[1], rgb[2]) for rgb in rgba_array])
    # Set the edge color black
    pc.set_edgecolor('k')
    # Add polygons to the plot
    ax.add_collection3d(pc)
    # Set axes label
    ax.set_xlabel(f"$x,\,\mathrm{{{length_unit}}}$")
    ax.set_ylabel(f"$y,\,\mathrm{{{length_unit}}}$")
    ax.set_zlabel(f"$z,\,\mathrm{{{length_unit}}}$")
    # Set axes limits
    x_coords = vertices[..., 0].ravel()
    y_coords = vertices[..., 1].ravel()
    z_coords = vertices[..., 2].ravel()
    ax.set_xlim(x_coords.min(), x_coords.max())
    ax.set_ylim(y_coords.min(), y_coords.max())
    ax.set_zlim(z_coords.min(), z_coords.max())
    # Set aspect ratio of the axes
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    # Return axes object
    return fig, ax, m


def plot_buckling_mode(op2_object: OP2, subcase_id: Union[int, tuple], mode_number: int = 1, displacement_component: str = "magnitude",
                       unit_scale_factor: float = 1., displacement_amplification_factor: float = 200., colormap: str = "rainbow_PuRd",
                       length_unit: str = "mm") -> Tuple[Figure, Axes3D, Colorbar]:
    """
    Plot the buckling shape using the eigenvectors of the input OP2 object.

    Parameters
    ----------
    op2_object: OP2
        pyNastran object created reading an op2 file with the load_geometry option set to True
    subcase_id: int, tuple
        key of the eigenvectors' dictionary in the OP2 object corresponding to the selected subcase
    mode_number: int
        number of the buckling mode to be plotted
    displacement_component: str
        name of the displacement component used for the colorbar
    unit_scale_factor: float
        scale factor for unit conversion
    displacement_amplification_factor: float
        scale factor applied to the displacements
    colormap: str
        name of the colormap used for the displacement colorbar
    length_unit: str
        measurement unit of coordinates and displacements, used in the label of axes and colormap

    Returns
    -------
    fig: Figure
        figure object
    ax: Axes3D
        axes object
    cbar: Colorbar
        colorbar object
    """
    # Choose eigenvectors as displacement data
    displacement_data = op2_object.eigenvectors[subcase_id].data[mode_number - 1, :, :]
    # Call plotting function
    fig, ax, m = plot_displacements(op2_object=op2_object, displacement_data=displacement_data,
                                    node_ids=op2_object.eigenvectors[subcase_id].node_gridtype[:, 0],
                                    displacement_component=displacement_component, displacement_unit_scale_factor=1.,
                                    coordinate_unit_scale_factor=unit_scale_factor,
                                    displacement_amplification_factor=displacement_amplification_factor, colormap=colormap,
                                    length_unit=length_unit)
    # Add colorbar
    label_dict = {'tx': 'Nondimensional $u_x$',
                  'ty': 'Nondimensional $u_y$',
                  'tz': 'Nondimensional $u_z$',
                  'rx': 'Nondimensional rotation about $x$',
                  'ry': 'Nondimensional rotation about $y$',
                  'rz': 'Nondimensional rotation about $z$',
                  'magnitude': 'Nondimensional displacement magnitude'}
    cbar = fig.colorbar(mappable=m, label=label_dict[displacement_component])
    # Set whitespace to 0
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # Return axes object
    return fig, ax, cbar


def plot_static_deformation(op2_object: OP2, subcase_id: Union[int, tuple] = 1, load_step: int = 0,
                            displacement_component: str = "magnitude", unit_scale_factor: float = 1.,
                            displacement_amplification_factor:float = 1., colormap: str = "rainbow_PuRd", clim: Union[list, ndarray] = None,
                            length_unit: str = "mm") -> Tuple[Figure, Axes3D, Colorbar]:
    """
    Plot the buckling shape using the eigenvectors of the input OP2 object.

    Parameters
    ----------
    op2_object: OP2
        pyNastran object created reading an op2 file with the load_geometry option set to True
    subcase_id: int, tuple
        key of the displacements' dictionary in the OP2 object corresponding to the selected subcase
    load_step: int
        integer representing the load step of a nonlinear or a time-dependent analysis, default is 0 so that last step
        is chosen
    displacement_component: str
        name of the displacement component used for the colorbar
    unit_scale_factor: float
        scale factor for unit conversion
    displacement_amplification_factor: float
        scale factor applied to the displacements
    colormap: str
        name of the colormap used for the displacement colorbar
    clim: Union[list, ndarray]:
        colorbar values limits
    length_unit: str
        measurement unit of coordinates and displacements, used in the label of axes and colormap
    
    Returns
    -------
    fig: Figure
        figure object
    ax: Axes3D
        axes object
    cbar: Colorbar
        colorbar object
    """
    # Choose static displacements as displacement data
    displacement_data = op2_object.displacements[subcase_id].data[load_step - 1, :, :]
    # Call plotting function
    fig, ax, m = plot_displacements(op2_object=op2_object, displacement_data=displacement_data,
                                    node_ids=op2_object.displacements[subcase_id].node_gridtype[:, 0],
                                    displacement_component=displacement_component, displacement_unit_scale_factor=unit_scale_factor,
                                    coordinate_unit_scale_factor=unit_scale_factor,
                                    displacement_amplification_factor=displacement_amplification_factor, colormap=colormap,
                                    clim=clim, length_unit=length_unit)
    # Add colorbar
    label_dict = {'tx': f'$u_x$, {length_unit}',
                  'ty': f'$u_y$, {length_unit}',
                  'tz': f'$u_z$, {length_unit}',
                  'rx': '$\\theta_x,\,\mathrm{rad}$',
                  'ry': '$\\theta_y,\,\mathrm{rad}$',
                  'rz': '$\\theta_z,\,\mathrm{rad}$',
                  'magnitude': f'Displacement magnitude, {length_unit}'}
    cbar = fig.colorbar(mappable=m, label=label_dict[displacement_component])
    # Set whitespace to 0
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # Return axes object
    return fig, ax, cbar


def add_unitary_force(bdf_object: BDF, nodes_ids: Union[list, ndarray], set_id: int, direction_vector: Union[list, ndarray]):
    """
    Apply a uniform force over the indicated nodes such that the total magnitude is 1 N.

    Parameters
    ----------
    bdf_object: BDF
        pyNastran object representing a bdf input file
    nodes_ids: [list, ndarray]
        array with ids of the nodes where the force is applied
    set_id: int
        set id of the force card
    direction_vector: [list, ndarray]
        vector with the force components along the x, y and z axes
    """
    # Define force magnitude so that the total magnitude is 1 N
    force_magnitude = 1 / len(nodes_ids)
    # Add a force card for each input node id
    for node_id in nodes_ids:
        bdf_object.add_force(sid=set_id, node=node_id, mag=force_magnitude, xyz=direction_vector)


def set_up_newton_method(bdf_object: BDF, nlparm_id: int = 1, ninc: int = None, max_iter: int = 25, conv: str = 'PW',
                         eps_u: float = 0.01, eps_p: float = 0.01, eps_w: float = 0.01, max_bisect: int = 5,
                         subcase_id: int = 0):
    """
    Assign SOL 106 as solution sequence, add parameter to consider large displacement effects and add NLPARM to set up
    the full Newton method.

    Parameters
    ----------
    bdf_object: BDF
        pyNastran object representing a bdf input file
    nlparm_id: int
        identification number of NLPARM card
    ninc: int
        number of increments
    max_iter: int
        limit on number of iterations for each load increment
    conv: str
        flag to select convergence criteria
    eps_u: float
        error tolerance for displacement criterion
    eps_p: float
        error tolerance for load criterion
    eps_w: float
        error tolerance for work criterion
    max_bisect: int
        maximum number of bisections allowed for each load increment
    subcase_id: int
        identification number of the subcase where the NLPARM card is applied
    """
    # Assign SOL 106 as solution sequence
    bdf_object.sol = 106
    # Add parameter for large displacement effects
    bdf_object.add_param('LGDISP', [1])
    # Define parameters for the nonlinear iteration strategy with full Newton method
    bdf_object.add_nlparm(nlparm_id=nlparm_id, ninc=ninc, kmethod='ITER', kstep=1, max_iter=max_iter, conv=conv,
                          int_out='YES', eps_u=eps_u, eps_p=eps_p, eps_w=eps_w, max_bisect=max_bisect)
    # Add NLPARM id to the control case commands
    bdf_object.case_control_deck.subcases[subcase_id].add_integer_type('NLPARM', nlparm_id)


def set_up_arc_length_method(bdf_object: BDF, nlparm_id: int = 1, ninc: int = None, max_iter: int = 25,
                             conv: str = "PW", eps_u: float = 0.01, eps_p: float = 0.01, eps_w: float = 0.01,
                             max_bisect: int = 5, subcase_id: int = 0, constraint_type: str = "CRIS",
                             minalr: float = 0.25, maxalr: float = 4., desiter: int = 12, maxinc: int = 20):
    """
    Assign SOL 106 as solution sequence, add parameter to consider large displacement effects and add NLPARM and NLPCI
    to set up the arc-length method.

    Parameters
    ----------
    bdf_object: BDF
        pyNastran object representing a bdf input file
    nlparm_id: int
        identification number of NLPARM card
    ninc: int
        number of increments
    max_iter: int
        limit on number of iterations for each load increment
    conv: str
        flag to select convergence criteria
    eps_u: float
        error tolerance for displacement criterion
    eps_p: float
        error tolerance for load criterion
    eps_w: float
        error tolerance for work criterion
    max_bisect: int
        maximum number of bisections allowed for each load increment
    subcase_id: int
        identification number of the subcase where the NLPARM card is applied
    constraint_type: str
        type of constraint used in the arc-length method
    minalr: float
        minimum allowable arc-length adjustment ratio
    maxalr: float
        maximum allowable arc-length adjustment ratio
    desiter: int
        desired nuber of iteration for convergence
    maxinc: int
        maximum number of controlled increment steps allowed within a subcase
    """
    # Set up basic nonlinear analysis
    set_up_newton_method(bdf_object=bdf_object, nlparm_id=nlparm_id, ninc=ninc, max_iter=max_iter, conv=conv,
                         eps_u=eps_u, eps_p=eps_p, eps_w=eps_w, max_bisect=max_bisect, subcase_id=subcase_id)
    # Define parameters for the arc-length method
    bdf_object.add_nlpci(nlpci_id=nlparm_id, Type=constraint_type, minalr=minalr, maxalr=maxalr, desiter=desiter, mxinc=maxinc)


def run_sol_105_buckling_analysis(bdf_object: BDF, static_load_set_id: int, analysis_directory_path: str,
                                  input_name: str, no_eigenvalues:int = 1, run_flag: bool = True) -> OP2:
    """
    Returns the OP2 object representing the results of SOL 105 analysis. The function defines subcase 1 to apply the
    load set associated to the input set idenfitication number and subcase 2 to calculate the critical eigenvalue
    using the EIGRL card.

    Parameters
    ----------
    bdf_object: BDF
        pyNastran object representing the bdf input of the box beam model
    static_load_set_id: int
        set id of the static load applied in the first subcase
    analysis_directory_path: str
        string with the path to the directory where the analysis is run
    input_name: str
        string with the name that will be given to the input file
    no_eigenvalues: int
        number of calculated buckling loads
    run_flag: bool
        boolean indicating whether Nastran analysis is actually run

    Returns
    -------
    op2_output: OP2
        object representing the op2 file produced by SOL 105
    """
    # Set SOL 105 as solution sequence (linear buckling analysis)
    bdf_object.sol = 105
    # Create first subcase for the application of the static load
    load_application_subcase_id = 1
    create_static_load_subcase(bdf_object=bdf_object, subcase_id=load_application_subcase_id,
                               load_set_id=static_load_set_id)
    # Add EIGRL card to define the parameters for the eigenvalues calculation
    eigrl_set_id = static_load_set_id + 1
    bdf_object.add_eigrl(sid=eigrl_set_id, v1=0., nd=no_eigenvalues)  # calculate the first nd positive eigenvalues
    # Create second subcase for the calculation of the eigenvalues
    eigenvalue_calculation_subcase_id = 2
    bdf_object.create_subcases(eigenvalue_calculation_subcase_id)
    bdf_object.case_control_deck.subcases[eigenvalue_calculation_subcase_id].add_integer_type('METHOD', eigrl_set_id)
    # Run analysis
    run_analysis(directory_path=analysis_directory_path, bdf_object=bdf_object, filename=input_name, run_flag=run_flag)
    # Read op2 file
    op2_filepath = os.path.join(analysis_directory_path, input_name + '.op2')
    op2_output = read_op2(op2_filename=op2_filepath, load_geometry=True, debug=None)
    # Return OP2 object
    return op2_output


def run_nonlinear_buckling_method(bdf_object: BDF, method_set_id: int, analysis_directory_path: str, input_name: str,
                                  calculate_tangent_stiffness_matrix_eigenvalues: bool = False,
                                  no_eigenvalues: int = 1, run_flag: bool = True) -> OP2:
    """
    Returns the OP2 object representing the results of SOL 106 analysis employing the nonlinear buckling method. The
    function requires the subcases with the associated load sets to be already defined. It applies the nonlinear
    buckling method to all subcases using the PARAM,BUCKLE,2 command and the EIGRL card.

    Parameters
    ----------
    bdf_object: BDF
        pyNastran object representing the bdf input of the box beam model
    method_set_id: int
        identification number of the EIGRL card that is defined for the eigenvalue calculation
    analysis_directory_path: str
        string with the path to the directory where the analysis is run
    input_name: str
        string with the name that will be given to the input file
    calculate_tangent_stiffness_matrix_eigenvalues: bool
        boolean indicating whether lowest eigenvalues of tangent stiffness matrix will be calculated
    no_eigenvalues: int
        number of eigenvalues of the tangent stiffness matrix that will be calculated
    run_flag: bool
        boolean indicating whether Nastran analysis is actually run

    Returns
    -------
    op2_output: OP2
        object representing the op2 file produced by SOL 106
    """
    # Set SOL 106 as solution sequence (nonlinear analysis)
    bdf_object.sol = 106
    # Define parameters for nonlinear buckling method
    bdf_object.add_param('BUCKLE', [2])
    bdf_object.add_eigrl(sid=method_set_id, nd=no_eigenvalues)  # calculate lowest eigenvalues in magnitude
    bdf_object.case_control_deck.subcases[0].add_integer_type('METHOD', method_set_id)  # add EIGRL id to case control
    # Define parameters to calculate lowest eigenvalues of tangent stiffness matrix if requested
    if calculate_tangent_stiffness_matrix_eigenvalues:
        bdf_object.executive_control_lines[1:1] = [
            'include \'' + os.path.join(os.pardir, os.pardir, 'resources', 'kllrh_lowest_eigenvalues.dmap') + '\'']
        bdf_object.add_param('BMODES', [no_eigenvalues])
    # Run analysis
    run_analysis(directory_path=analysis_directory_path, bdf_object=bdf_object, filename=input_name, run_flag=run_flag)
    # Read op2 file
    op2_filepath = os.path.join(analysis_directory_path, input_name + '.op2')
    op2_output = read_op2(op2_filename=op2_filepath, load_geometry=True, debug=None)
    # Return OP2 object
    return op2_output


def run_tangent_stiffness_matrix_eigenvalue_calculation(bdf_object: BDF, method_set_id: int,
                                                        analysis_directory_path: str, input_name: str,
                                                        no_eigenvalues: int = 1, run_flag: bool = True) -> OP2:
    """
    Returns the OP2 object representing the results of SOL 106 analysis employing the nonlinear buckling method. The
    function requires the subcases with the associated load sets to be already defined. It applies the nonlinear
    buckling method to all subcases using the PARAM,BUCKLE,2 command and the EIGRL card.

    Parameters
    ----------
    bdf_object: BDF
        pyNastran object representing the bdf input of the box beam model
    method_set_id: int
        identification number of the EIGRL card that is defined for the eigenvalue calculation
    analysis_directory_path: str
        string with the path to the directory where the analysis is run
    input_name: str
        string with the name that will be given to the input file
    no_eigenvalues: int
        number of eigenvalues of the tangent stiffness matrix that will be calculated
    run_flag: bool
        boolean indicating whether Nastran analysis is actually run

    Returns
    -------
    op2_output: OP2
        object representing the op2 file produced by SOL 106
    """
    # Set SOL 106 as solution sequence (nonlinear analysis)
    bdf_object.sol = 106
    # Define parameters to calculate lowest eigenvalues of tangent stiffness matrix
    bdf_object.add_param('BUCKLE', [2])
    bdf_object.add_eigrl(sid=method_set_id, nd=no_eigenvalues)  # calculate lowest eigenvalues in magnitude
    bdf_object.case_control_deck.subcases[0].add_integer_type('METHOD', method_set_id)  # add EIGRL id to case control
    bdf_object.executive_control_lines[1:1] = [
        'include \'' + os.path.join(os.pardir, os.pardir, 'resources', 'kllrh_lowest_eigenvalues_nobuckle.dmap') + '\'']
    bdf_object.add_param('BMODES', [no_eigenvalues])
    # Run analysis
    run_analysis(directory_path=analysis_directory_path, bdf_object=bdf_object, filename=input_name, run_flag=run_flag)
    # Read op2 file
    op2_filepath = os.path.join(analysis_directory_path, input_name + '.op2')
    op2_output = read_op2(op2_filename=op2_filepath, load_geometry=True, debug=None)
    # Return OP2 object
    return op2_output
