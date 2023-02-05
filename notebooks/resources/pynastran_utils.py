import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from numpy import ndarray
import os
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
from pyNastran.op2.op2 import read_op2
from pyNastran.utils.nastran_utils import run_nastran
import re
import time
from typing import Tuple, Dict, Any


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
    nastran_path = 'C:\\Program Files\\MSC.Software\\MSC_Nastran\\2021.4\\bin\\nastranw.exe'
    run_nastran(bdf_filename=bdf_filepath, nastran_cmd=nastran_path, run_in_bdf_dir=True, run=run_flag)
    # If Nastran is actually executed wait until analysis is done
    if run_flag:
        wait_nastran(directory_path)
    # Read and print wall time of simulation
    log_filepath = os.path.join(directory_path, filename + '.log')
    regexp = re.compile('-? *[0-9]+.?[0-9]*(?:[Ee] *[-+]? *[0-9]+)?')  # compiled regular expression pattern
    with open(log_filepath) as log_file:
        for line in log_file:
            if 'Total' in line:
                wall_time = float(re.findall(regexp, line)[1])
                print(f'Nastran job {bdf_filename} completed\nWall time: {wall_time:.1f} s')
                break


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


def plot_displacements(op2_object: OP2, displacement_data: ndarray, displacement_component: str = 'magnitude',
                       displacement_scale_factor: float = 1., colormap: str = 'jet') -> \
        Tuple[Figure, Axes3D, ScalarMappable]:
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
    displacement_scale_factor: float
        scale factor for displacements (used to plot buckling modes)
    colormap: str
        name of the colormap used for the displacement colorbar

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
    if displacement_component == 'magnitude':
        nodes_displacement = np.empty((len(op2_object.elements), 3, 4))
    else:
        nodes_displacement = np.empty((len(op2_object.elements), 4))
    vertices = np.empty((len(op2_object.elements), 4, 3))
    # Iterate through the elements of the structure
    for count, element in enumerate(op2_object.elements.values()):
        # Store ids of the nodes as an array
        node_ids = np.array(element.node_ids)
        # Store indicated displacement component for the current nodes
        nodes_displacement[count] = displacement_data[-1, node_ids - 1, component_dict[displacement_component]]
        # Store the coordinates of the nodes of the deformed elements
        vertices[count] = np.vstack(
            [op2_object.nodes[index].xyz + displacement_data[-1, index - 1, 0:3] *
             displacement_scale_factor for index in node_ids])
    # Calculate displacement magnitude if requested
    if displacement_component == 'magnitude':
        nodes_displacement = np.apply_along_axis(np.linalg.norm, 1, nodes_displacement)
    # Calculate average displacement for each element
    elements_mean_displacement = np.apply_along_axis(np.mean, 1, nodes_displacement)
    # Create 3D polygons to represent the elements
    pc = Poly3DCollection(vertices, linewidths=.05)
    # Create colormap for the displacement magnitude
    m = ScalarMappable(cmap=colormap)
    m.set_array(elements_mean_displacement)
    # Set colormap min and max values and displacement values to colors
    m.set_clim(vmin=np.amin(nodes_displacement), vmax=np.amax(nodes_displacement))
    rgba_array = m.to_rgba(elements_mean_displacement)
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
    # Return axes object
    return fig, ax, m


def plot_buckling_mode(op2_object: OP2, subcase_id: [int, tuple], displacement_component: str = 'magnitude',
                       colormap: str = 'jet') -> Tuple[Figure, Axes3D]:
    """
    Plot the buckling shape using the eigenvectors of the input OP2 object.

    Parameters
    ----------
    op2_object: OP2
        pyNastran object created reading an op2 file with the load_geometry option set to True
    subcase_id: int, tuple
        key of the eigenvectors' dictionary in the OP2 object corresponding to the selected subcase
    displacement_component: str
        name of the displacement component used for the colorbar
    colormap: str
        name of the colormap used for the displacement colorbar

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
    displacement_scale_factor = 200
    fig, ax, m = plot_displacements(op2_object, displacement_data, displacement_component, displacement_scale_factor,
                                    colormap)
    # Add colorbar
    label_dict = {'tx': 'Nondimensional displacement along $x$',
                  'ty': 'Nondimensional displacement along $y$',
                  'tz': 'Nondimensional displacement along $z$',
                  'rx': 'Nondimensional rotation about $x$',
                  'ry': 'Nondimensional rotation about $y$',
                  'rz': 'Nondimensional rotation about $z$',
                  'magnitude': 'Nondimensional displacement magnitude'}
    fig.colorbar(mappable=m, label=label_dict[displacement_component], pad=0.15)
    # Return axes object
    return fig, ax


def plot_static_deformation(op2_object: OP2, subcase_id: [int, tuple] = 1, displacement_component: str = 'magnitude',
                            colormap: str = 'jet') -> Tuple[Figure, Axes3D]:
    """
    Plot the buckling shape using the eigenvectors of the input OP2 object.

    Parameters
    ----------
    op2_object: OP2
        pyNastran object created reading an op2 file with the load_geometry option set to True
    subcase_id: int, tuple
        key of the displacements' dictionary in the OP2 object corresponding to the selected subcase
    displacement_component: str
        name of the displacement component used for the colorbar
    colormap: str
        name of the colormap used for the displacement colorbar

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
    fig, ax, m = plot_displacements(op2_object, displacement_data, displacement_component, colormap=colormap)
    # Add colorbar
    label_dict = {'tx': 'Displacement along $x$ [mm]',
                  'ty': 'Displacement along $y$ [mm]',
                  'tz': 'Displacement along $z$ [mm]',
                  'rx': 'Rotation about $x$ [rad]',
                  'ry': 'Rotation about $y$ [rad]',
                  'rz': 'Rotation about $z$ [rad]',
                  'magnitude': 'Displacement magnitude [mm]'}
    fig.colorbar(mappable=m, label=label_dict[displacement_component], pad=0.15)
    # Return axes object
    return fig, ax


def add_unitary_force(bdf_object: BDF, nodes_ids: [list, ndarray], set_id: int, direction_vector: [list, ndarray]):
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
                             conv: str = 'PW', eps_u: float = 0.01, eps_p: float = 0.01, eps_w: float = 0.01,
                             max_bisect: int = 5, subcase_id: int = 0, minalr: float = 0.25, maxalr: float = 4.,
                             desiter: int = 12, maxinc: int = 20):
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
    bdf_object.add_nlpci(nlpci_id=nlparm_id, Type='CRIS', minalr=minalr, maxalr=maxalr, desiter=desiter, mxinc=maxinc)


def run_sol_105_buckling_analysis(bdf_object: BDF, static_load_set_id: int, analysis_directory_path: str,
                                  input_name: str, run_flag: bool = True) -> OP2:
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
    bdf_object.add_eigrl(sid=eigrl_set_id, v1=0., nd=1)  # calculate only the first positive eigenvalue
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


def run_sol_106_buckling_analysis(bdf_object: BDF, method_set_id: int, analysis_directory_path: str, input_name: str,
                                  run_flag: bool = True) -> OP2:
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
    run_flag: bool
        boolean indicating whether Nastran analysis is actually run

    Returns
    -------
    op2_output: OP2
        object representing the op2 file produced by SOL 106
    """
    # Set SOL SOL6 as solution sequence (nonlinear analysis)
    bdf_object.sol = 106
    # Define parameters for nonlinear buckling method
    bdf_object.add_param('BUCKLE', [2])
    bdf_object.add_eigrl(sid=method_set_id, nd=1)  # calculate lowest eigenvalue, either positive or negative
    bdf_object.case_control_deck.subcases[0].add_integer_type('METHOD', method_set_id)
    # Run analysis
    run_analysis(directory_path=analysis_directory_path, bdf_object=bdf_object, filename=input_name, run_flag=run_flag)
    # Read op2 file
    op2_filepath = os.path.join(analysis_directory_path, input_name + '.op2')
    op2_output = read_op2(op2_filename=op2_filepath, load_geometry=True, debug=None)
    # Return OP2 object
    return op2_output
