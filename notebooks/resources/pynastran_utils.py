"""
This file is part of the GitHub repository nonlinear-structural-stability-notebooks, created by Francesco M. A. Mitrotta.
Copyright (C) 2022 Francesco Mario Antonio Mitrotta

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from numpy import ndarray
import os
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2, read_op2
from pyNastran.utils.nastran_utils import run_nastran
import re
from typing import Tuple, Dict, Any, Union
import subprocess


# Constant variables
NASTRAN_PATH = r"C:\Program Files\MSC.Software\MSC_Nastran\2021.4\bin\nastran.exe"  # path to nastran executable - change to your path
RESOURCES_PATH = os.path.dirname(os.path.abspath(__file__))  # set resources folder path


def run_analysis(
    directory_path: str, filename: str, bdf: BDF, parallel: bool = False,
    no_cores: int = 6, run_flag: bool = True):
    """
    Write .bdf input file from BDF object and execute Nastran analysis.

    Parameters
    ----------
    directory_path : str
        string with path to the directory where input file is run
    filename: str
        name of the input file without extension
    bdf: BDF
        pyNastran object representing Nastran bdf input
    parallel: bool
        flag to enable or disable the parallel execution of Nastran
    no_cores: int
        number of cores used for the parallel execution of Nastran
    run_flag: bool
        flag to enable or disable the actual execution of Nastran
    """
    # Create analysis directory if it does not exist
    os.makedirs(directory_path, exist_ok=True)
    
    # Write bdf file
    bdf_filename = filename + '.bdf'
    bdf_filepath = os.path.join(directory_path, bdf_filename)
    bdf.write_bdf(bdf_filepath, is_double=True)  # use double precision
    
    # Create default keywords list
    keywords_list = ["scr=yes", "bat=no", "old=no", "news=no", "notify=no"]
    
    # Add smp keyword for parallel execution
    if parallel:
        keywords_list.append(f"smp={no_cores:d}")
    
    # Call Nastran process
    run_nastran(
        bdf_filename=bdf_filepath, nastran_cmd=NASTRAN_PATH,
        run_in_bdf_dir=True, run=run_flag, keywords=keywords_list)
        
    # Read and print wall time of simulation
    log_filepath = os.path.join(directory_path, filename + '.log')
    regexp = re.compile('-? *[0-9]+.?[0-9]*(?:[Ee] *[-+]? *[0-9]+)?')  # compiled regular expression pattern
    with open(log_filepath) as log_file:
        for line in log_file:
            if 'Total' in line:
                wall_time = float(re.findall(regexp, line)[1])
                print(f"Nastran job {bdf_filename} completed\nWall time: {wall_time:.1f} s")
                break


def create_static_load_subcase(
    bdf: BDF, subcase_id: int, load_set_id: int, nlparm_id: int = None):
    """
    Define a subcase for the application of an external static load.

    Parameters
    ----------
    bdf : BDF
        pyNastran object representing Nastran input file
    subcase_id: int
        id of the subcase
    load_set_id: int
        id of the load set assigned to the subcase
    nlparm_id: int
        id of the nonlinear analysis settings assigned to the subcase
    """
    # Create subcase with input id
    bdf.create_subcases(subcase_id)
    
    # Add load set id to case control statement of subcase
    bdf.case_control_deck.subcases[subcase_id].add_integer_type(
        'LOAD', load_set_id)
    
    # If provided, add NLPARM id to case control statement of subcase
    if nlparm_id:
        bdf.case_control_deck.subcases[subcase_id].add_integer_type(
            'NLPARM', nlparm_id)
        

def create_real_eigenvalue_extraction_subcase(
    bdf: BDF, subcase_id: int, method_set_id: int, no_eigenvalues: int = 1):
    """
    Define a subcase for the application of an external static load.

    Parameters
    ----------
    bdf : BDF
        pyNastran object representing Nastran input file
    subcase_id: int
        id of the subcase
    method_set_id: int
        id of the real eigenvalue extraction parameters assigned to the
        subcase
    no_eigenvalues: int
        number of real eigenvalues to be calculated
    """
    # Create subcase with input id
    bdf.create_subcases(subcase_id)
    
    # Add method set id to case control statement of subcase
    bdf.case_control_deck.subcases[subcase_id].add_integer_type(
        'METHOD', method_set_id)
    
    # If provided set id isn't already in the bdf, add EIGRL card
    if method_set_id not in bdf.methods:
        bdf.add_eigrl(sid=method_set_id, v1=0., nd=no_eigenvalues)
    else:
        # Else, update parameters of the existing EIGRL card
        bdf.methods[method_set_id].v1 = 0.  # only positive eigenvalues
        bdf.methods[method_set_id].nd = no_eigenvalues


def read_displacement_from_op2(
    op2: OP2, node_ids: list[int] = [1]) -> Dict[Any, Dict[int, ndarray]]:
    """
    Read displacements at the indicated nodes from a linear analysis OP2
    object.

    Parameters
    ----------
    op2 : OP2
        pyNastran object including the results read from an op2 file
    node_ids: list[int]
        list of ids of the nodes where the displacements are read

    Returns
    -------
    displacements: dict
        dictionary of dictionaries with node ids as primary keys,
        subcase ids as secondary keys and numpy arrays of displacements 
        and rotations at the indicated node as values
    """
    # Initialize dictionary where the displacements will be saved
    displacements = {id: {} for id in node_ids}
    
    # Loop through the subcases
    for subcase_id in op2.displacements:
        # Loop through the node ids
        for id in node_ids:
            # Save displacements of current subcase and current node id
            node_index = np.where(
                op2.displacements[subcase_id].node_gridtype[:, 0] == id)[0][0]
            displacements[id][subcase_id] = op2.displacements[
                subcase_id].data[:, node_index, :]
    
    # Return output data
    return displacements


def read_load_displacement_history_from_op2(
    op2: OP2, node_ids: list[int] = [1]) -> \
        Tuple[Dict[Any, ndarray], Dict[Any, ndarray], Dict[Any, ndarray]]:
    """
    Read history of total applied loads and of displacements at the
    indicated nodes from a nonlinear analysis OP2 object.

    Parameters
    ----------
    op2 : OP2
        pyNastran object including the results of a nonlinear analysis
    node_ids: list[int]
        list of ids of the nodes where the displacements are read

    Returns
    -------
    load_steps: dict
        dictionary with a vector of the load steps for each subcase
    displacements: dict
        dictionary of dictionaries with node ids as primary keys,
        subcase ids as secondary keys and numpy arrays of displacements 
        and rotations at the indicated node as values
    loads: dict
        dictionary with subcase ids as keys and arrays of the applied
        loads as values
    """
    # Initialize dictionaries where the quantities of interest will be saved
    load_steps = {}
    displacements = {id: {} for id in node_ids}
    loads = {}
    
    # Iterate through the subcases found in the op2 file
    valid_subcase_ids = [
        subcase_id for subcase_id in op2.load_vectors if hasattr(
            op2.load_vectors[subcase_id], 'lftsfqs')]
    for subcase_id in valid_subcase_ids:
        # Save load steps of current subcase
        load_steps[subcase_id] = op2.load_vectors[subcase_id].lftsfqs
        
        # Save loads summation of current subcase
        loads[subcase_id] = np.sum(
            op2.load_vectors[subcase_id].data[:, :, 0:3], axis=1)
        
        # Loop through the node ids
        for id in node_ids:
            # Save displacements of current subcase and current node id
            node_index = np.where(
                op2.displacements[subcase_id].node_gridtype[:, 0] == id)[0][0]
            displacements[id][subcase_id] = op2.displacements[
                subcase_id].data[:, node_index, :]
    
    # Return output data
    return load_steps, loads, displacements


def read_nonlinear_buckling_load_from_f06(f06_filepath: str, op2: OP2) -> Tuple[ndarray, ndarray]:
    """
    Return nonlinear buckling load vector and critical buckling factors by reading the .f06 and .op2 files of a
    SOL 106 analyis with the nonlinear buckling method.

    Parameters
    ----------
    f06_filepath: str
        string with path to .f06 file
    op2: OP2
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
    valid_subcase_ids = [subcase_id for subcase_id in op2.load_vectors if
                         hasattr(op2.load_vectors[subcase_id], 'lftsfqs')]
    # Return None if no valid subcases are found
    if valid_subcase_ids == []:
        nonlinear_buckling_load_vectors = None
    else:
        # Initialize list of nonlinear buckling load vectors
        nonlinear_buckling_load_vectors = np.empty(tuple([len(alphas)]) + np.shape(op2.load_vectors[valid_subcase_ids[0]].data[-1, :, :]))
        # Iterate through the valid subcases
        for i, subcase_id in enumerate(valid_subcase_ids):
            # Find the final load vector of current subcase
            final_load_vector = op2.load_vectors[subcase_id].data[-1, :, :]
            # Find last increment vector of current subcase
            last_increment_vector = final_load_vector - op2.load_vectors[subcase_id].data[-2, :, :]
            # Calculate nonlinear buckling load as P+DeltaP*ALPHA
            nonlinear_buckling_load_vectors[i] = final_load_vector + last_increment_vector * alphas[i]
    # Return lists of nonlinear buckling loads and critical buckling factors
    return nonlinear_buckling_load_vectors, np.array(alphas)


def read_kllrh_lowest_eigenvalues_from_f06(f06_filepath: str) -> ndarray:
    """
    Return a list with the lowest eigenvalues of the matrix KLLRH (tangent stiffness matrix) for each load increment
    reading a f06 file resulting from a nonlinear analysis run with a proper DMAP.

    Parameters
    ----------
        f06_filepath: str
            string with path to .f06 file

    Returns
    -------
        eigenvalue_array: ndarray
            array with the eigenvalues of the KLLRH matrices for each converged increment, dimensions (number of eigenvalues, number of increments)

    Raises
    ------
        ValueError
            If no line containing 'NLOOP' is found in the file
        FileNotFoundError
            If the specified file does not exist
    """
    # Initialize the list of the lowest eigenvalues
    eigenvalue_list = []
    # Flag to check if NLOOP was found
    nloop_found = False
    # Compile a regular expression pattern to read eigenvalue in f06 file
    regexp = re.compile('-? *[0-9]+.?[0-9]*(?:[Ee] *[-+]? *[0-9]+)?')
    
    # Open file iterate line by line
    with open(f06_filepath) as f06_file:
        for line in f06_file:
            # When NLOOP is found append an empty list inside the eigenvalues list
            if 'NLOOP =' in line:
                nloop_found = True
                eigenvalue_list.append([])
            # When KLLRH EIGENVALUE is found append the eigenvalue to the last list of the eigenvalues list
            if 'KLLRH EIGENVALUE' in line:
                raw_results = re.findall(regexp, line)
                eigenvalue_list[-1].append(float(raw_results[1]))
    
    # Check if NLOOP was found
    if not nloop_found:
        raise ValueError("No 'NLOOP' line found in the f06 file. Please check if the input bdf file contains the expected DMAP.")
    
    # Convert list of lists to array padded with nan (we put nan for the iterations where we miss one or more eigenvalues)
    lengths = np.array([len(item) for item in eigenvalue_list])
    mask = lengths[:, None] > np.arange(lengths.max())
    eigenvalue_array = np.full(mask.shape, np.nan)
    eigenvalue_array[mask] = np.concatenate(eigenvalue_list)
    
    # Return array of eigenvalues in the form: number of eigenvalues x number of iterations
    return eigenvalue_array.T


def read_kllrh_lowest_eigenvalues_from_op2(op2: OP2) -> ndarray:
    """
    Return the array of the lowest N eigenvalues of the matrix KLLRH (tangent stiffness matrix) for each converged increment
    of the nonlinear analysis represented by the input OP2 object.

    Parameters
    ----------
        op2: OP2
            pyNastran object including the results of a nonlinear analysis (SOL 106) with the calculation of the eigenvalues of the tangent stiffness matrix

    Returns
    -------
        eigenvalue_array: ndarray
            array with the eigenvalues of the KLLRH matrices for each converged increment, dimensions (number of eigenvalues, number of increments)
    """
    # Find the key of the first eigenvectors object in the OP2 object
    eigenvectors_key = next(iter(op2.eigenvectors))
    # Find max number of eigenvalues calculated for each converged increment
    max_no_eigenvalues = op2.eigenvectors[eigenvectors_key].lsdvmns.max().astype(int)
    # Find the start and end indices of each converged increment
    sequence_starts = np.where(op2.eigenvectors[eigenvectors_key].lsdvmns == 1)[0]
    sequence_ends = np.roll(sequence_starts, -1)
    # Pad the last end index with the length of the eigenvectors list
    sequence_ends[-1] = len(op2.eigenvectors[eigenvectors_key].lsdvmns)
    # Create a max_no_eigenvalues x number of converged increments array with the eigenvalues of the KLLRH matrix
    eigenvalues = np.column_stack([
        np.pad(op2.eigenvectors[eigenvectors_key].eigrs[start:end], (0, max_no_eigenvalues - (end - start)), mode='constant', constant_values=np.nan)  # store the eigenvalues from start to end index and pad the rest of the column with nan
        for start, end in zip(sequence_starts, sequence_ends)])  # iterate through the start and end indices of the converged increments
    # Return array
    return eigenvalues


def add_uniform_force(bdf: BDF, nodes_ids: Union[list, ndarray], set_id: int, direction_vector: Union[list, ndarray], force_magnitude: float = 1):
    """
    Apply a uniformly distributed force over the indicated nodes.

    Parameters
    ----------
    bdf: BDF
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
        bdf.add_force(sid=set_id, node=node_id, mag=nodal_force, xyz=direction_vector)


def add_uniform_pressure(bdf: BDF, elements_ids: Union[list, ndarray], set_id: int, force_magnitude: float = 1):
    """
    Apply a uniformly distributed pressure over the indicated elements.

    Parameters
    ----------
    bdf: BDF
        pyNastran object representing a bdf input file
    elements_ids: [list, ndarray]
        array with ids of the elements where the pressure is applied
    set_id: int
        set id of the force card
    force_magnitude: float
        total magnitude of the applied force, to be distributed uniformly as a pressure over the elements
    """
    # Find pressure value by dividing the force magnitude by the sum of the areas of the elements
    pressure = force_magnitude / np.sum([bdf.elements[element_id].Area() for element_id in elements_ids])
    # Add PLOAD2 card to define the pressure over the elements
    bdf.add_pload2(sid=set_id, pressure=pressure, eids=elements_ids)


def set_up_newton_method(bdf: BDF, nlparm_id: int = 1, ninc: int = None, kstep: int = -1, max_iter: int = 25, conv: str = 'PW',
                         eps_u: float = 0.01, eps_p: float = 0.01, eps_w: float = 0.01, max_bisect: int = 5,
                         subcase_id: int = 0):
    """
    Assign SOL 106 as solution sequence, add parameter to consider large displacement effects and add NLPARM to set up
    the load control method with full Newton iteration.

    Parameters
    ----------
    bdf: BDF
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
    bdf.sol = 106
    # Add parameter for large displacement effects
    if 'LGDISP' not in bdf.params:
        bdf.add_param('LGDISP', [1])
    # Define parameters for the nonlinear iteration strategy with full Newton method (update tangent stiffness matrix after every converged iteration)
    bdf.add_nlparm(nlparm_id=nlparm_id, ninc=ninc, kmethod='ITER', kstep=kstep, max_iter=max_iter, conv=conv,
                   int_out='YES', eps_u=eps_u, eps_p=eps_p, eps_w=eps_w, max_bisect=max_bisect)
    # Add NLPARM id to case control deck of the indicated subcase
    if 'NLPARM' not in bdf.case_control_deck.subcases[subcase_id].params:
        bdf.case_control_deck.subcases[subcase_id].add_integer_type('NLPARM', nlparm_id)  # add new NLPARM command if not present
    else:
        bdf.case_control_deck.subcases[subcase_id].params['NLPARM'][0] = nlparm_id  # overwrite existing NLPARM id if command is already present


def set_up_arc_length_method(bdf: BDF, nlparm_id: int = 1, ninc: int = None, kstep: int = -1, max_iter: int = 25,
                             conv: str = 'PW', eps_u: float = 0.01, eps_p: float = 0.01, eps_w: float = 0.01,
                             max_bisect: int = 5, subcase_id: int = 0, constraint_type: str = 'CRIS',
                             minalr: float = 0.25, maxalr: float = 4., desiter: int = 12, maxinc: int = 20):
    """
    Assign SOL 106 as solution sequence, add parameter to consider large displacement effects and add NLPARM and NLPCI
    to set up the arc-length method.

    Parameters
    ----------
    bdf: BDF
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
    set_up_newton_method(bdf=bdf, nlparm_id=nlparm_id, ninc=ninc, kstep=kstep, max_iter=max_iter, conv=conv,
                         eps_u=eps_u, eps_p=eps_p, eps_w=eps_w, max_bisect=max_bisect, subcase_id=subcase_id)
    # Define parameters for the arc-length method
    bdf.add_nlpci(nlpci_id=nlparm_id, Type=constraint_type, minalr=minalr, maxalr=maxalr, desiter=desiter, mxinc=maxinc)


def set_up_sol_105(
    bdf: BDF, static_load_set_id: int = None, method_set_id: int = None,
    no_eigenvalues:int = 1):
    """
    Set up a SOL 105 analysis. The function defines subcase 1 to apply the load set associated to the input load set id and subcase 2
    to calculate the buckling eigenvalues using the EIGRL card.

    Parameters
    ----------
    bdf: BDF
        pyNastran object representing the Nastran bdf input
    static_load_set_id: int
        set id of the static load applied in the first subcase
    method_set_id: int
        set id of the real eigenvalue extraction parameters used in the
        second subcase
    no_eigenvalues: int
        number of real eigenvalues to be calculated
    """
    # Set SOL 105 as solution sequence (linear buckling analysis)
    bdf.sol = 105
    
    # If set id of the static load is provided, create first subcase for the
    # application of the static load
    if static_load_set_id is not None:
        load_application_subcase_id = 1
        create_static_load_subcase(
            bdf=bdf, subcase_id=load_application_subcase_id,
            load_set_id=static_load_set_id)
    
    # If set id of eigenvalue extraction parameters is provided, create second
    # subcase for the calculation of the buckling eigenvalues
    if method_set_id is not None:
        eigenvalue_calculation_subcase_id = 2
        create_real_eigenvalue_extraction_subcase(
            bdf=bdf, subcase_id=eigenvalue_calculation_subcase_id,
            method_set_id=method_set_id, no_eigenvalues=no_eigenvalues)


def run_sol_105(bdf: BDF, input_name: str, analysis_directory_path: str,
                static_load_set_id: int = None, method_set_id: int = None,
                no_eigenvalues:int = 1, run_flag: bool = True) -> OP2:
    """
    Set up and run a SOL 105 analysis and return the resulting OP2
    object. This function calls set_up_sol_105 to define the subcases
    for the analysis and run_analysis to execute the analysis.

    Parameters
    ----------
    bdf: BDF
        pyNastran object representing the Nastran bdf input
    input_name: str
        string with the name that will be given to the input file
    analysis_directory_path: str
        string with the path to the directory where the analysis is run
    static_load_set_id: int
        set id of the static load applied in the first subcase
    method_set_id: int
        set id of the real eigenvalue extraction parameters used in the
        second subcase
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
    set_up_sol_105(
        bdf=bdf, static_load_set_id=static_load_set_id,
        method_set_id=method_set_id, no_eigenvalues=no_eigenvalues)
    
    # Run analysis
    run_analysis(
        directory_path=analysis_directory_path, bdf=bdf, filename=input_name,
        run_flag=run_flag)
    
    # Read op2 file and return OP2 object
    op2_filepath = os.path.join(analysis_directory_path, input_name + '.op2')
    op2_output = read_op2(
        op2_filename=op2_filepath, load_geometry=True, debug=None)
    return op2_output


def run_nonlinear_buckling_method(bdf: BDF, method_set_id: int, analysis_directory_path: str, input_name: str,
                                  calculate_tangent_stiffness_matrix_eigenvalues: bool = False,
                                  no_eigenvalues: int = 1, run_flag: bool = True) -> OP2:
    """
    Returns the OP2 object representing the results of SOL 106 analysis employing the nonlinear buckling method. The
    function requires the subcases with the associated load sets to be already defined. It applies the nonlinear
    buckling method to all subcases using the PARAM,BUCKLE,2 command and the EIGRL card.

    Parameters
    ----------
    bdf: BDF
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
    bdf.sol = 106
    # Define cards for nonlinear buckling method
    bdf.add_param('BUCKLE', [2])
    bdf.add_eigrl(sid=method_set_id, nd=no_eigenvalues)  # calculate lowest eigenvalues in magnitude
    bdf.case_control_deck.subcases[0].add_integer_type('METHOD', method_set_id)  # add EIGRL id to case control
    # Include DMAP to calculate eigenvalues of tangent stiffness matrix
    relative_path_to_resources = os.path.relpath(RESOURCES_PATH, analysis_directory_path)  # relative path to resources folder
    if calculate_tangent_stiffness_matrix_eigenvalues:
        bdf.executive_control_lines[1:1] = [
            "include '" + os.path.join(relative_path_to_resources, "kllrh_eigenvalues.dmap") + "'"]
        bdf.add_param('BMODES', [no_eigenvalues])
    # Run analysis
    run_analysis(directory_path=analysis_directory_path, bdf=bdf, filename=input_name, run_flag=run_flag)
    # Read op2 file
    op2_filepath = os.path.join(analysis_directory_path, input_name + '.op2')
    op2_output = read_op2(op2_filename=op2_filepath, load_geometry=True, debug=None)
    # Return OP2 object
    return op2_output


def eig_to_cycles(eig: float) -> float:
    """
    Convert the eigenvalue unit to cycles.

    Parameters
    ----------
    eig: float
        eigenvalue of the tangent stiffness matrix

    Returns
    -------
    cycles: float
        number of cycles corresponding to the input eigenvalue
    """
    if eig < 0:  # if negative convert absolute value of eigenvalue to cycles and then make the final number negative
        cycles = -np.sqrt(np.abs(eig))/(2*np.pi)
    else:  # if positive convert eigenvalue to cycles
        cycles = np.sqrt(eig)/(2*np.pi)
    return cycles


def set_up_sol_106_with_kllrh_eigenvalues(
    bdf: BDF, method_set_id: int, analysis_directory_path: str,
    no_eigenvalues: int = 1, dmap_option: str = None,
    lower_eig: float = -1.e32, upper_eig: float = 1.e32) -> OP2:
    """
    Set up a SOL 106 analysis with the calculation of the eigenvalues of
    the tangent stiffness matrix.

    Parameters
    ----------
    bdf: BDF
        pyNastran object representing the bdf input
    method_set_id: int
        identification number of the EIGRL card that is defined for the
        eigenvalue calculation
    analysis_directory_path: str
        string with the path to the directory where the analysis is run
    no_eigenvalues: int
        number of eigenvalues of the tangent stiffness matrix that will
        be calculated
    dmap_option: str
        string indicating the additional task to be performed by the
        DMAP. "eigenvectors" will calculate eigenvectors, "stop" will
        stop the analysis after the first negative eigenvalue. If None,
        the default DMAP will be used.
    lower_eig: float
        lower bound of the eigenvalues to be calculated
    upper_eig: float
        upper bound of the eigenvalues to be calculated

    Returns
    -------
    op2_output: OP2
        object representing the op2 file produced by SOL 106
    """
    # Set SOL 106 as solution sequence (nonlinear analysis)
    bdf.sol = 106
    
    # Define cards to calculate smallest magnitude eigenvalues of tangent
    # stiffness matrix
    bdf.add_param('BUCKLE', [2])  # add PARAM BUCKLE
    bdf.add_eigrl(sid=method_set_id, nd=no_eigenvalues)  # add EIGRL card
    bdf.case_control_deck.subcases[0].add_integer_type(
        'METHOD', method_set_id)  # add EIGRL id to case control
    
    # Select DMAP file to calculate eigenvalues of tangent stiffness matrix
    if dmap_option is None:
        dmap_filename = "kllrh_eigenvalues_nobuckle.dmap"
    elif dmap_option == "eigenvectors":
        dmap_filename = "kllrh_eigenvectors.dmap"
    elif dmap_option == "stop":
        dmap_filename = "kllrh_eigenvalues_stop.dmap"
    else:
        raise ValueError(
            "Invalid DMAP option. Choose 'eigenvectors' or 'stop'.")
    
    # Include DMAP file in the bdf input
    relative_path_to_resources = os.path.relpath(
        RESOURCES_PATH, analysis_directory_path)
    bdf.executive_control_lines.insert(
        -1, "include '" + os.path.join(
            relative_path_to_resources, dmap_filename) + "'")
    
    # Define parameters to calculate lowest eigenvalues of tangent stiffness
    # matrix
    if no_eigenvalues > 1:
        bdf.add_param('BMODES', [no_eigenvalues])  # add PARAM BMODES if more than one eigenvalue is calculated
    if lower_eig > -1.e32:
        bdf.add_param('LOWEREIG', [eig_to_cycles(lower_eig)])  # add PARAM LOWEREIG if lower bound is defined
    if upper_eig < 1.e32:
        bdf.add_param('UPPEREIG', [eig_to_cycles(upper_eig)])  # add PARAM UPPEREIG if upper bound is defined


def run_sol_106_with_kllrh_eigenvalues(
    bdf: BDF, method_set_id: int, analysis_directory_path: str,
    input_name: str, no_eigenvalues: int = 1, dmap_option: str = None,
    lower_eig: float = -1.e32, upper_eig: float = 1.e32,
    run_flag: bool = True) -> OP2:
    """
    Set up and run a SOL 106 analysis with the calculation of the
    eigenvalues of the tangent stiffness matrix.

    Parameters
    ----------
    bdf: BDF
        pyNastran object representing the bdf input
    method_set_id: int
        identification number of the EIGRL card that is defined for the
        eigenvalue calculation
    analysis_directory_path: str
        string with the path to the directory where the analysis is run
    input_name: str
        string with the name that will be given to the input file
    no_eigenvalues: int
        number of eigenvalues of the tangent stiffness matrix that will
        be calculated
    dmap_option: str
        string indicating the additional task to be performed by the
        DMAP. "eigenvectors" will calculate eigenvectors, "stop" will
        stop the analysis after the first negative eigenvalue. If None,
        the default DMAP will be used.
    lower_eig: float
        lower bound of the eigenvalues to be calculated
    upper_eig: float
        upper bound of the eigenvalues to be calculated
    run_flag: bool
        boolean indicating whether Nastran analysis is actually run

    Returns
    -------
    op2_output: OP2
        object representing the op2 file produced by SOL 106
    """
    # Set up SOL 106 analyis with the calculation of the eigenvalues of the tangent stiffness matrix
    set_up_sol_106_with_kllrh_eigenvalues(
        bdf=bdf, method_set_id=method_set_id,
        analysis_directory_path=analysis_directory_path,
        no_eigenvalues=no_eigenvalues, dmap_option=dmap_option,
        lower_eig=lower_eig, upper_eig=upper_eig)
    
    # Run analysis
    run_analysis(
        directory_path=analysis_directory_path, bdf=bdf, filename=input_name,
        run_flag=run_flag)
    
    # Read op2 file
    op2_filepath = os.path.join(analysis_directory_path, input_name + '.op2')
    op2_output = read_op2(
        op2_filename=op2_filepath, load_geometry=True, debug=None)
    
    # Return OP2 object
    return op2_output


def read_monitor_point_from_f06(
    f06_path: str, monitor_point_name: str) -> Dict[str, np.ndarray]:
    """
    Read the data of a monitor point from a Nastran .f06 file.
    
    Parameters
    ----------
    f06_path: str
        path to the .f06 file
    monitor_point_name: str
        name of the monitor point to be read
    
    Returns
    -------
    data_dict: dict
        dictionary with axis labels as keys and numpy arrays with the
        data as values
    """
    # Open the file and read all lines into a list for easy access by index
    with open(f06_path, 'r') as file:
        lines = file.readlines()
    
    # Initialize control variables
    start_table = False  # flag to indicate the start of the desired table
    data_dict = {}       # dictionary to store parsed table data
    
    # Step 1: Locate the monitor point section by searching for the specified
    # monitor point name
    for i, line in enumerate(lines):
        if f"MONITOR POINT NAME = {monitor_point_name}" in line:
            # Once the monitor point is found, start searching for the table
            # header
            for j in range(i, len(lines)):
                # Look for the table header line, which marks the start of the
                # data table
                if "AXIS      RIGID AIR       ELASTIC REST." in lines[j]:
                    start_table = True
                    # Set table start two lines below the header
                    table_start = j + 2
                    break
        # Exit the outer loop if the table start has been found
        if start_table:
            break
    
    # If the monitor point or table is not found, raise an error
    if not start_table:
        raise ValueError(
            f"Monitor point {monitor_point_name} not found in the file.")
    
    # Step 2: Read the table data line by line, starting from the identified
    # table start line
    for k in range(table_start, len(lines)):
        # Check for an empty line or "PAGE" to mark the end of the table
        # section
        if lines[k].strip() == "" or "PAGE" in lines[k]:
            break
        else:
            # Split the line into parts: first part is the axis label, rest are
            # numerical values
            line_data = lines[k].split()
            axis = line_data[0]  # extract axis label (first item in line)
            # Convert remaining items to floats and store in a numpy array
            data = np.array([float(val) for val in line_data[1:]])
            # Store the axis and data array in the dictionary
            data_dict[axis] = data
    
    # Return the dictionary with axis labels as keys and data arrays as values
    return data_dict
