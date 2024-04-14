"""
This file is part of the GitHub repository nonlinear-structural-stability-notebooks, created by Francesco M. A. Mitrotta.
Copyright (C) 2022 Francesco Mario Antonio Mitrotta

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
plt.cm.register_cmap('sunset', tc.tol_cmap('sunset'))
plt.cm.register_cmap('rainbow_PuRd', tc.tol_cmap('rainbow_PuRd'))
# Set resources folder path
RESOURCES_PATH = os.path.dirname(os.path.abspath(__file__))


def run_analysis(directory_path: str, bdf_object: BDF, filename: str, run_flag: bool = True,
                 parallel:bool = False, no_cores:int = 6):
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
    parallel: bool
        flag to enable or disable the parallel execution of Nastran
    no_cores: int
        number of cores used for the parallel execution of Nastran
    """
    # Create analysis directory if it does not exist
    os.makedirs(directory_path, exist_ok=True)
    # Write bdf file
    bdf_filename = filename + '.bdf'
    bdf_filepath = os.path.join(directory_path, bdf_filename)
    bdf_object.write_bdf(bdf_filepath, is_double=True)  # write bdf file with double precision
    # Create keywords list for parallel execution
    if parallel:
        keywords_list = ['scr=yes', 'bat=no', 'old=no', 'news=no', 'notify=no', f"smp={no_cores:d}"]
    else:
        keywords_list = None
    # Run Nastran
    nastran_path = 'C:\\Program Files\\MSC.Software\\MSC_Nastran\\2021.4\\bin\\nastran.exe'
    run_nastran(bdf_filename=bdf_filepath, nastran_cmd=nastran_path, run_in_bdf_dir=True, run=run_flag, keywords=keywords_list)
    # Read and print wall time of simulation
    log_filepath = os.path.join(directory_path, filename + '.log')
    regexp = re.compile('-? *[0-9]+.?[0-9]*(?:[Ee] *[-+]? *[0-9]+)?')  # compiled regular expression pattern
    with open(log_filepath) as log_file:
        for line in log_file:
            if 'Total' in line:
                wall_time = float(re.findall(regexp, line)[1])
                print(f"Nastran job {bdf_filename} completed\nWall time: {wall_time:.1f} s")
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
        loads[subcase_id] = np.sum(op2_object.load_vectors[subcase_id].data[:, :, 0:3], axis=1)
        # Save displacements of indicated node id and current subcase
        node_index = np.where(op2_object.displacements[subcase_id].node_gridtype[:, 0] == displacement_node_id)[0][0]
        displacements[subcase_id] = op2_object.displacements[subcase_id].data[:, node_index, :]
    # Return output data
    return load_steps, loads, displacements


def read_nonlinear_buckling_load_from_f06(f06_filepath: str, op2_object: OP2) -> Tuple[ndarray, ndarray]:
    """
    Return nonlinear buckling load vector and critical buckling factors by reading the .f06 and .op2 files of a
    SOL 106 analyis with the nonlinear buckling method.

    Parameters
    ----------
    f06_filepath: str
        string with path to .f06 file
    op2_object: OP2
        pyNastran object including the results of a nonlinear analysis (SOL 106) with nonlinear buckling method

    Returns
    -------
    nonlinear_buckling_load_vectors: ndarray
        array of the nonlinear buckling load vectors, calculated as P + Delta P * alpha, dimensions (number of subcases, number of nodes, 6)
    alphas: ndarray
        array of critical buckling factors, dimensions (number of subcases)
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
    # Return None if no valid subcases are found
    if valid_subcase_ids == []:
        nonlinear_buckling_load_vectors = None
    else:
        # Initialize list of nonlinear buckling load vectors
        nonlinear_buckling_load_vectors = np.empty(tuple([len(alphas)]) + np.shape(op2_object.load_vectors[valid_subcase_ids[0]].data[-1, :, :]))
        # Iterate through the valid subcases
        for i, subcase_id in enumerate(valid_subcase_ids):
            # Find the final load vector of current subcase
            final_load_vector = op2_object.load_vectors[subcase_id].data[-1, :, :]
            # Find last increment vector of current subcase
            last_increment_vector = final_load_vector - op2_object.load_vectors[subcase_id].data[-2, :, :]
            # Calculate nonlinear buckling load as P+DeltaP*ALPHA
            nonlinear_buckling_load_vectors[i] = final_load_vector + last_increment_vector * alphas[i]
    # Return lists of nonlinear buckling loads and critical buckling factors
    return nonlinear_buckling_load_vectors, np.array(alphas)


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


def plot_displacements(op2_object: OP2, displacement_data: ndarray, displacement_component: str = "magnitude",
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
    # Extract mapping from node ids to indexes and undeformed coordinates
    node_id_to_index = {}
    undeformed_coordinates_array = np.empty((len(op2_object.nodes), 3))
    for node_index, (node_id, node) in enumerate(op2_object.nodes.items()):
        node_id_to_index[node_id] = node_index
        undeformed_coordinates_array[node_index] = node.xyz
    # Apply scalings and find deformed coordinates
    undeformed_coordinates_array *= coordinate_unit_scale_factor  # apply unit conversion to undeformed coordinates
    displacement_data[:, :3] = displacement_data[:, :3] * displacement_unit_scale_factor  # apply unit conversion to displacements
    deformed_coordinates_array = undeformed_coordinates_array + displacement_data[:, :3] * displacement_amplification_factor  # apply amplification factor to displacements and calculate deformed coordinates
    # Find node indexes for each element
    element_node_indexes = np.array([[node_id_to_index[node_id] for node_id in element.node_ids] for element in op2_object.elements.values()])
    # Find vertices coordinates for Poly3DCollection
    vertices = deformed_coordinates_array[element_node_indexes]
    # Create Poly3DCollection
    pc = Poly3DCollection(vertices, linewidths=.05, edgecolor='k')
    # Handle displacement component and calculate nodal displacement array of interest
    if displacement_component == 'magnitude':
        nodal_displacement_array = np.linalg.norm(displacement_data[:, :3], axis=1)  # calculate displacement magnitude
    else:
        component_dict = {'tx': 0, 'ty': 1, 'tz': 2, 'rx': 3, 'ry': 4, 'rz': 5}
        nodal_displacement_array = displacement_data[:, component_dict[displacement_component]]  # select displacement component
    # Calculate average displacement for each element
    fringe_data = np.mean(nodal_displacement_array[element_node_indexes], axis=1)
    # Create colormap for the displacement magnitude
    m = ScalarMappable(cmap=colormap)
    m.set_array(fringe_data)
    # Set colormap min and max values and displacement values to colors
    if clim is None:
        m.set_clim(vmin=np.amin(nodal_displacement_array), vmax=np.amax(nodal_displacement_array))
    else:
        m.set_clim(vmin=clim[0], vmax=clim[1])
    rgba_array = m.to_rgba(fringe_data)
    # Color the elements' face by the average displacement of interest
    pc.set_facecolor([(rgb[0], rgb[1], rgb[2]) for rgb in rgba_array])
    # Initialize figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
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
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in 'xyz')])
    # Return axes object
    return fig, ax, m


def plot_buckling_mode(op2_object: OP2, subcase_id: Union[int, tuple], mode_number: int = 1, displacement_component: str = 'magnitude',
                       unit_scale_factor: float = 1., displacement_amplification_factor: float = 200., colormap: str = 'rainbow_PuRd',
                       length_unit: str = 'mm') -> Tuple[Figure, Axes3D, Colorbar]:
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
    fig, ax, m = plot_displacements(op2_object=op2_object, displacement_data=displacement_data.copy(),
                                    displacement_component=displacement_component, displacement_unit_scale_factor=1.,
                                    coordinate_unit_scale_factor=unit_scale_factor,
                                    displacement_amplification_factor=displacement_amplification_factor, colormap=colormap,
                                    length_unit=length_unit)
    # Add colorbar
    label_dict = {'tx': "Nondimensional $u_x$",
                  'ty': "Nondimensional $u_y$",
                  'tz': "Nondimensional $u_z$",
                  'rx': "Nondimensional $\\theta_x$",
                  'ry': "Nondimensional $\\theta_y$",
                  'rz': "Nondimensional $\\theta_z$",
                  'magnitude': "Nondimensional $\|u\|$"}
    cbar = fig.colorbar(mappable=m, label=label_dict[displacement_component])
    # Set whitespace to 0
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # Return axes object
    return fig, ax, cbar


def plot_static_deformation(op2_object: OP2, subcase_id: Union[int, tuple] = 1, load_step: int = 0,
                            displacement_component: str = "magnitude", unit_scale_factor: float = 1.,
                            displacement_amplification_factor:float = 1., colormap: str = "rainbow_PuRd",
                            clim: Union[list, ndarray] = None, length_unit: str = "mm") -> Tuple[Figure, Axes3D, Colorbar]:
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
    fig, ax, m = plot_displacements(op2_object=op2_object, displacement_data=displacement_data.copy(),
                                    displacement_component=displacement_component, displacement_unit_scale_factor=unit_scale_factor,
                                    coordinate_unit_scale_factor=unit_scale_factor,
                                    displacement_amplification_factor=displacement_amplification_factor, colormap=colormap,
                                    clim=clim, length_unit=length_unit)
    # Add colorbar
    label_dict = {'tx': f"$u_x$, {length_unit}",
                  'ty': f"$u_y$, {length_unit}",
                  'tz': f"$u_z$, {length_unit}",
                  'rx': "$\\theta_x,\,\mathrm{rad}$",
                  'ry': "$\\theta_y,\,\mathrm{rad}$",
                  'rz': "$\\theta_z,\,\mathrm{rad}$",
                  'magnitude': f"\|u\|, {length_unit}"}
    cbar = fig.colorbar(mappable=m, label=label_dict[displacement_component])
    # Set whitespace to 0
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # Return axes object
    return fig, ax, cbar


def add_uniform_force(bdf_object: BDF, nodes_ids: Union[list, ndarray], set_id: int, direction_vector: Union[list, ndarray], force_magnitude: float = 1):
    """
    Apply a uniformly distributed force over the indicated nodes.

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
    force_magnitude: float
        total magnitude of the applied force, to be distributed uniformly over the nodes
    """
    # Define force magnitude so that the total magnitude is equal to the input force magnitude
    nodal_force = force_magnitude / len(nodes_ids)
    # Add a force card for each input node id
    for node_id in nodes_ids:
        bdf_object.add_force(sid=set_id, node=node_id, mag=nodal_force, xyz=direction_vector)


def add_uniform_pressure(bdf_object: BDF, elements_ids: Union[list, ndarray], set_id: int, force_magnitude: float = 1):
    """
    Apply a uniformly distributed pressure over the indicated elements.

    Parameters
    ----------
    bdf_object: BDF
        pyNastran object representing a bdf input file
    elements_ids: [list, ndarray]
        array with ids of the elements where the pressure is applied
    set_id: int
        set id of the force card
    force_magnitude: float
        total magnitude of the applied force, to be distributed uniformly as a pressure over the elements
    """
    # Find pressure value by dividing the force magnitude by the sum of the areas of the elements
    pressure = force_magnitude / np.sum([bdf_object.elements[element_id].Area() for element_id in elements_ids])
    # Add PLOAD2 card to define the pressure over the elements
    bdf_object.add_pload2(sid=set_id, pressure=pressure, eids=elements_ids)


def set_up_newton_method(bdf_object: BDF, nlparm_id: int = 1, ninc: int = None, kstep: int = -1, max_iter: int = 25, conv: str = 'PW',
                         eps_u: float = 0.01, eps_p: float = 0.01, eps_w: float = 0.01, max_bisect: int = 5,
                         subcase_id: int = 0):
    """
    Assign SOL 106 as solution sequence, add parameter to consider large displacement effects and add NLPARM to set up
    the load control method with full Newton iteration.

    Parameters
    ----------
    bdf_object: BDF
        pyNastran object representing a bdf input file
    nlparm_id: int
        identification number of NLPARM card
    ninc: int
        number of increments
    kstep: int
        number of iterations before the stiffness update
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
    if 'LGDISP' not in bdf_object.params:
        bdf_object.add_param('LGDISP', [1])
    # Define parameters for the nonlinear iteration strategy with full Newton method (update tangent stiffness matrix after every converged iteration)
    bdf_object.add_nlparm(nlparm_id=nlparm_id, ninc=ninc, kmethod='ITER', kstep=kstep, max_iter=max_iter, conv=conv,
                          int_out='YES', eps_u=eps_u, eps_p=eps_p, eps_w=eps_w, max_bisect=max_bisect)
    # Add NLPARM id to case control deck of the indicated subcase
    if 'NLPARM' not in bdf_object.case_control_deck.subcases[subcase_id].params:
        bdf_object.case_control_deck.subcases[subcase_id].add_integer_type('NLPARM', nlparm_id)  # add new NLPARM command if not present
    else:
        bdf_object.case_control_deck.subcases[subcase_id].params['NLPARM'][0] = nlparm_id  # overwrite existing NLPARM id if command is already present


def set_up_arc_length_method(bdf_object: BDF, nlparm_id: int = 1, ninc: int = None, kstep: int = -1, max_iter: int = 25,
                             conv: str = 'PW', eps_u: float = 0.01, eps_p: float = 0.01, eps_w: float = 0.01,
                             max_bisect: int = 5, subcase_id: int = 0, constraint_type: str = 'CRIS',
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
    kstep: int
        number of iterations before the stiffness update
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
    set_up_newton_method(bdf_object=bdf_object, nlparm_id=nlparm_id, ninc=ninc, kstep=kstep, max_iter=max_iter, conv=conv,
                         eps_u=eps_u, eps_p=eps_p, eps_w=eps_w, max_bisect=max_bisect, subcase_id=subcase_id)
    # Define parameters for the arc-length method
    bdf_object.add_nlpci(nlpci_id=nlparm_id, Type=constraint_type, minalr=minalr, maxalr=maxalr, desiter=desiter, mxinc=maxinc)


def set_up_sol_105(bdf_object: BDF, static_load_set_id: int, no_eigenvalues:int = 1):
    """
    Set up a SOL 105 analysis. The function defines subcase 1 to apply the load set associated to the input load set id and subcase 2
    to calculate the buckling eigenvalues using the EIGRL card.

    Parameters
    ----------
    bdf_object: BDF
        pyNastran object representing the bdf input of the box beam model
    static_load_set_id: int
        set id of the static load applied in the first subcase
    no_eigenvalues: int
        number of calculated buckling loads
    """
    # Set SOL 105 as solution sequence (linear buckling analysis)
    bdf_object.sol = 105
    # Create first subcase for the application of the static load
    load_application_subcase_id = 1
    create_static_load_subcase(bdf_object=bdf_object, subcase_id=load_application_subcase_id,
                               load_set_id=static_load_set_id)
    # Add EIGRL card to define the parameters for the eigenvalue calculation
    eigrl_set_id = static_load_set_id + 1
    bdf_object.add_eigrl(sid=eigrl_set_id, v1=0., nd=no_eigenvalues)  # calculate the first nd positive eigenvalues
    # Create second subcase for the calculation of the buckling eigenvalues
    eigenvalue_calculation_subcase_id = 2
    bdf_object.create_subcases(eigenvalue_calculation_subcase_id)
    bdf_object.case_control_deck.subcases[eigenvalue_calculation_subcase_id].add_integer_type('METHOD', eigrl_set_id)  # add EIGRL id to case control deck of second subcase


def run_sol_105(bdf_object: BDF, static_load_set_id: int, analysis_directory_path: str, input_name: str, no_eigenvalues:int = 1,
                run_flag: bool = True) -> OP2:
    """
    Set up and run a SOL 105 analysis and return the resulting OP2 object. This function calls set_up_sol_105 to define the
    subcases and cards for the analysis and run_analysis to execute the analysis.

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
    # Set up SOL 105 to run linear buckling analysis
    set_up_sol_105(bdf_object=bdf_object, static_load_set_id=static_load_set_id, no_eigenvalues=no_eigenvalues)
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
    # Define cards for nonlinear buckling method
    bdf_object.add_param('BUCKLE', [2])
    bdf_object.add_eigrl(sid=method_set_id, nd=no_eigenvalues)  # calculate lowest eigenvalues in magnitude
    bdf_object.case_control_deck.subcases[0].add_integer_type('METHOD', method_set_id)  # add EIGRL id to case control
    # Include DMAP to calculate eigenvalues of tangent stiffness matrix
    relative_path_to_resources = os.path.relpath(RESOURCES_PATH, analysis_directory_path)  # relative path to resources folder
    if calculate_tangent_stiffness_matrix_eigenvalues:
        bdf_object.executive_control_lines[1:1] = [
            "include '" + os.path.join(relative_path_to_resources, "kllrh_eigenvalues.dmap") + "'"]
        bdf_object.add_param('BMODES', [no_eigenvalues])
    # Run analysis
    run_analysis(directory_path=analysis_directory_path, bdf_object=bdf_object, filename=input_name, run_flag=run_flag)
    # Read op2 file
    op2_filepath = os.path.join(analysis_directory_path, input_name + '.op2')
    op2_output = read_op2(op2_filename=op2_filepath, load_geometry=True, debug=None)
    # Return OP2 object
    return op2_output


def set_up_sol_106_with_kllrh_eigenvalues(bdf_object: BDF, analysis_directory_path: str, method_set_id: int,
                                          no_eigenvalues: int = 1, lower_eig: float = -1.e32, upper_eig: float = 1.e32,
                                          eigenvectors_flag: bool = False) -> OP2:
    """
    Set up a SOL 106 analysis with the calculation of the eigenvalues of the tangent stiffness matrix.

    Parameters
    ----------
    bdf_object: BDF
        pyNastran object representing the bdf input of the box beam model
    analysis_directory_path: str
        string with the path to the directory where the analysis is run
    method_set_id: int
        identification number of the EIGRL card that is defined for the eigenvalue calculation
    no_eigenvalues: int
        number of eigenvalues of the tangent stiffness matrix that will be calculated
    lower_eig: float
        lower bound of the eigenvalues to be calculated
    upper_eig: float
        upper bound of the eigenvalues to be calculated
    eigenvectors_flag: bool
        boolean indicating whether eigenvectors will be calculated

    Returns
    -------
    op2_output: OP2
        object representing the op2 file produced by SOL 106
    """
    # Set SOL 106 as solution sequence (nonlinear analysis)
    bdf_object.sol = 106
    # Define cards to calculate smallest magnitude eigenvalues of tangent stiffness matrix
    bdf_object.add_param('BUCKLE', [2])
    bdf_object.add_eigrl(sid=method_set_id, nd=no_eigenvalues)  # calculate lowest eigenvalues in magnitude
    bdf_object.case_control_deck.subcases[0].add_integer_type('METHOD', method_set_id)  # add EIGRL id to case control
    # Include DMAP to calculate eigenvalues of tangent stiffness matrix
    relative_path_to_resources = os.path.relpath(RESOURCES_PATH, analysis_directory_path)  # relative path to resources folder
    if eigenvectors_flag:
        bdf_object.executive_control_lines[1:1] = [
            "include '" + os.path.join(relative_path_to_resources, "kllrh_eigenvectors.dmap") + "'"]  # include DMAP to calculate eigenvectors
    else:
        bdf_object.executive_control_lines[1:1] = [
            "include '" + os.path.join(relative_path_to_resources, "kllrh_eigenvalues_nobuckle.dmap") + "'"]  # include DMAP to calculate only eigenvalues
    # Define parameters to calculate lowest eigenvalues of tangent stiffness matrix
    if no_eigenvalues > 1:
        bdf_object.add_param('BMODES', [no_eigenvalues])  # add PARAM BMODES if more than one eigenvalue is calculated
    if lower_eig > -1.e32:
        if lower_eig < 0:  # if negative convert absolute value of eigenvalue to cycle
            bdf_object.add_param('LOWEREIG', [-np.sqrt(np.abs(lower_eig))/(2*np.pi)])
        else:  # if positive convert eigenvalue to cycle
            bdf_object.add_param('LOWEREIG', [np.sqrt(lower_eig)/(2*np.pi)])
    if upper_eig < 1.e32:
        if upper_eig < 0:  # if negative convert absolute value of eigenvalue to cycle
            bdf_object.add_param('UPPEREIG', [-np.sqrt(np.abs(upper_eig))/(2*np.pi)])
        else:  # if positive convert eigenvalue to cycle
            bdf_object.add_param('UPPEREIG', [np.sqrt(upper_eig)/(2*np.pi)])


def run_sol_106_with_kllrh_eigenvalues(bdf_object: BDF, method_set_id: int, analysis_directory_path: str, input_name: str,
                                       no_eigenvalues: int = 1, lower_eig: float = -1.e32, upper_eig: float = 1.e32,
                                       eigenvectors_flag: bool = False, run_flag: bool = True) -> OP2:
    """
    Set up and run a SOL 106 analysis with the calculation of the eigenvalues of the tangent stiffness matrix.

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
    lower_eig: float
        lower bound of the eigenvalues to be calculated
    upper_eig: float
        upper bound of the eigenvalues to be calculated
    eigenvectors_flag: bool
        boolean indicating whether eigenvectors will be calculated
    run_flag: bool
        boolean indicating whether Nastran analysis is actually run

    Returns
    -------
    op2_output: OP2
        object representing the op2 file produced by SOL 106
    """
    # Set up SOL 106 analyis with the calculation of the eigenvalues of the tangent stiffness matrix
    set_up_sol_106_with_kllrh_eigenvalues(bdf_object=bdf_object, analysis_directory_path=analysis_directory_path,
                                          method_set_id=method_set_id, no_eigenvalues=no_eigenvalues, lower_eig=lower_eig,
                                          upper_eig=upper_eig, eigenvectors_flag=eigenvectors_flag)
    # Run analysis
    run_analysis(directory_path=analysis_directory_path, bdf_object=bdf_object, filename=input_name, run_flag=run_flag)
    # Read op2 file
    op2_filepath = os.path.join(analysis_directory_path, input_name + '.op2')
    op2_output = read_op2(op2_filename=op2_filepath, load_geometry=True, debug=None)
    # Return OP2 object
    return op2_output
