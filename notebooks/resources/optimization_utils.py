"""
This file is part of the GitHub repository nonlinear-structural-stability-notebooks, created by Francesco M. A. Mitrotta.
Copyright (C) 2024 Francesco Mario Antonio Mitrotta

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from numpy import ndarray  # import ndarray type from NumPy library
import numpy as np  # import NumPy library
import openmdao.api as om  # make available the most common OpenMDAO classes and functions
from pyNastran.bdf.bdf import BDF  # pyNastran BDF class
from pyNastran.bdf.mesh_utils.mass_properties import mass_properties  # pyNastran function to calculate mass properties
from resources import pynastran_utils  # custom module to deal with pyNastran objects
import os  # import os module to interact with the operating system
from pyNastran.op2.op2 import OP2, read_op2  # import OP2 object and function to read op2 file
import matplotlib.pyplot as plt  # import matplotlib library for plotting
from matplotlib.ticker import MaxNLocator  # import MaxNLocator class for axis ticks formatting
from matplotlib.figure import Figure  # class for figure objects
from matplotlib.axes import Axes  # class for axes objects
import warnings  # import warnings module to handle warnings


# Constants for Nastran analysis subcase ids
FIRST_SUBCASE_ID = 1  # id of the first subcase, used in both SOL 105 and SOL 106 analyses
SECOND_SUBCASE_ID = 2  # id of the second subcase, specifically for SOL 105 analyses


def read_linear_stresses(op2:OP2) -> ndarray:
    """
    Return array of linear von Mises stresses from OP2 object.

    Parameters
    ----------
    op2 : OP2
        OP2 object containing the analysis results.

    Returns
    -------
    ndarray
        Array of linear stresses for each element in the model.
    """
    stress_list = []
    
    if op2.op2_results.stress.cbar_stress:
        sa_max_array = op2.op2_results.stress.cbar_stress[FIRST_SUBCASE_ID].data[0, :, 5]  # maximum combined bending and axial stress at end A
        sa_min_array = op2.op2_results.stress.cbar_stress[FIRST_SUBCASE_ID].data[0, :, 6]  # minimum combined bending and axial stress at end A
        sb_max_array = op2.op2_results.stress.cbar_stress[FIRST_SUBCASE_ID].data[0, :, 12]  # maximum combined bending and axial stress at end A
        sb_min_array = op2.op2_results.stress.cbar_stress[FIRST_SUBCASE_ID].data[0, :, 13]  # minimum combined bending and axial stress at end A
        von_mises_stress_a = np.sqrt(sa_max_array**2 + sa_min_array**2 - sa_max_array*sa_min_array)  # von Mises stress at end A
        von_mises_stress_b = np.sqrt(sb_max_array**2 + sb_min_array**2 - sb_max_array*sb_min_array)  # von Mises stress at end B
        stress_list.append(np.concatenate((von_mises_stress_a, von_mises_stress_b)))  # append von Mises stresses for CBAR elements
    
    if op2.op2_results.stress.cquad4_stress:
        stress_list.append(op2.op2_results.stress.cquad4_stress[FIRST_SUBCASE_ID].data[0, :, 7])  # von Mises stress is in the 8th row of the stress array
    
    stress_array = np.concatenate(stress_list)  # concatenate stress arrays for all element types
    return stress_array


def compute_ks_function(g:ndarray, rho:float=100., upper:float=0., lower_flag:bool=False):
    """
    Computes the Kreisselmeier-Steinhauser (KS) function for a given array of constraint values.

    The KS function is used for aggregating constraint violations into a single scalar value, facilitating constraint handling in optimization problems.

    Parameters
    ----------
    g : ndarray
        Array of constraint values. By default negative means satisfied and positive means violated. Behavior is modified with upper and lower_flag.
    rho : float, optional
        Constraint aggregation factor. Default is 100.
    upper : float, optional
        Upper bound for the constraints. If lower_flag is True, then this is the lower bound. Default is 0.
    lower_flag : bool, optional
        Flag to indicate if the constraints are lower-bounded. Default is False.

    Returns
    -------
    float
        The computed value of the KS function.
    """
    con_val = g - upper  # adjust constraint values based on upper bound
    if lower_flag:
        con_val = -con_val  # invert constraint values if they are lower-bounded
    g_max = np.max(np.atleast_2d(con_val), axis=-1)[:, np.newaxis]  # find maximum constraint value
    g_diff = con_val - g_max  # subtract maximum constraint value
    exponents = np.exp(rho * g_diff)  # calculate exponential terms for aggregation
    summation = np.sum(exponents, axis=-1)[:, np.newaxis]  # sum the exponential terms
    KS = g_max + 1.0 / rho * np.log(summation)  # calculate the final KS function value
    return KS


class NastranSolver(om.ExplicitComponent):
    """
    An OpenMDAO component that performs a finite element analysis using a Nastran BDF model.

    This component can run linear buckling (SOL 105) or nonlinear static analysis (SOL 106) based on the defined BDF object and extract relevant analysis results.

    Attributes
    ----------
    options : dict
        A dictionary of options for configuring the Nastran analysis.
    
    Methods
    -------
    initialize()
        Declare options for the component.
    setup()
        Define the component's inputs and outputs.
    setup_partials()
        Declare partial derivatives for the component.
    compute(inputs, outputs, discrete_inputs, discrete_outputs)
        Executes the Nastran analysis with the provided inputs and populates the outputs based on the analysis results.
    """

    def initialize(self):
        """
        Declare options for the component.

        Options
        -------
        bdf : BDF
            The BDF object representing the model.
        analysis_directory_path : str
            Path to the directory where the analysis files are stored.
        input_name : str
            Name of the input file for the analysis.
        run_flag : bool
            Flag to indicate if the analysis should be run.
        yield_strength : float
            Yield strength of the material.
        eigenvalue_scale_factor : float
            Scaling factor applied to the eigenvalues of the tangent stiffness matrix before calculating the KS function.
        """
        self.options.declare('bdf', types=BDF, desc='BDF object representing the Nastran model.')
        self.options.declare('analysis_directory_path', types=str, desc='Directory path for storing analysis files.')
        self.options.declare('input_name', types=str, desc='Name for the analysis input file.')
        self.options.declare('run_flag', types=bool, default=True, desc='Flag to control whether the analysis should be executed.')
        self.options.declare('yield_strength', types=float, desc='Yield strength of the material.')
        self.options.declare('eigenvalue_scale_factor', types=float, default=1., desc='Scaling factor applied to the eigenvalues of the tangent stiffness matrix before calculating the KS function.')
        

    def setup(self):
        """
        Define the inputs and outputs for the Nastran analysis component. This includes defining the shape and type of analysis results to be expected.
        """
        # Define inputs
        self.add_input('t_val', shape_by_conn=True, desc='Thickness values for the elements in the model.')
        # Define outputs
        self.add_output('mass', desc='Total mass of the structure.')
        self.add_output('ks_stress', desc='Kreisselmeier-Steinhauser aggregated stress value.')
        self.add_discrete_output('op2', val=None, desc='OP2 object containing the analysis results.')
        # Define outputs based on analysis type
        if self.options['bdf'].sol == 105 or self.options['bdf'].sol == 144:
            self.add_output('ks_buckling', desc='KS aggregated buckling load factor.')
        elif self.options['bdf'].sol == 106:
            self.add_output('ks_stability', desc='KS aggregated stability metric.')
            self.add_output('load_factor', desc='Ratio of applied load to prescribed load in the final state of the analysis.')
        else:
            raise ValueError("Unsupported solution sequence. Must be SOL 105 or SOL 144 for buckling or SOL 106 for nonlinear static analysis.")

    def setup_partials(self):
        """
        Declare partial derivatives for the component using finite difference method.
        """
        # Finite difference all partials
        self.declare_partials(of='*', wrt='*', method='fd', step=1e-5)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """
        Conducts the Nastran analysis based on the inputs and updates the outputs with the results. This includes running the analysis, reading results, and aggregating outcomes.

        Parameters
        ----------
        inputs : dict
            Dictionary containing the input values.
        outputs : dict
            Dictionary containing the output values.
        discrete_inputs : dict
            Dictionary containing the discrete input values.
        discrete_outputs : dict
            Dictionary containing the discrete output values.
        """
        # Extract options for convenience
        bdf = self.options['bdf']
        analysis_directory_path = self.options['analysis_directory_path']
        input_name = self.options['input_name']
        run_flag = self.options['run_flag']
        yield_strength = self.options['yield_strength']
        eigenvalue_scale_factor = self.options['eigenvalue_scale_factor']
        
        # Get thickness values from inputs and loop over properties
        t_array = inputs['t_val']
        for i, pid in enumerate(bdf.properties):
            
            # Asssign thickness to PSHELL properties
            if bdf.properties[pid].type == 'PSHELL':
                bdf.properties[pid].t = t_array[0, i]
                bdf.properties[pid].z1 = None
                bdf.properties[pid].z2 = None
                
            # Assign thickness to PBARL properties
            elif bdf.properties[pid].type == 'PBARL':
                bdf.properties[pid].dim[2] = t_array[0, i]
                bdf.properties[pid].dim[3] = t_array[0, i]
        # Calculate mass
        outputs['mass'] = mass_properties(bdf)[0]
        # Run analysis
        pynastran_utils.run_analysis(directory_path=analysis_directory_path, bdf=bdf, filename=input_name, run_flag=run_flag)
        # Read op2 file and assign to discrete output
        op2_filepath = os.path.join(analysis_directory_path, input_name + '.op2')
        op2 = read_op2(op2_filepath, load_geometry=True, debug=None)
        discrete_outputs['op2'] = op2
        
        # Extract results based on analysis type
        if bdf.sol == 105:
            # Read linear von Mises stresses and aggregate with KS function
            stresses = read_linear_stresses(op2)  # find von Mises stresses for all elements
            outputs['ks_stress'] = compute_ks_function(stresses, upper=yield_strength)  # aggregate stresses with KS function using yield strength as upper bound
            
            # Read buckling load factors and aggregate with KS function
            outputs['ks_buckling'] = compute_ks_function(
                np.array(op2.eigenvectors[SECOND_SUBCASE_ID].eigrs), lower_flag=True, upper=1.)  # buckling load factor must be greater than 1 for the structure not to buckle
        
        elif bdf.sol == 106:
            # Read nonlinear von Mises stresses and aggregate with KS function
            subcase_id = next(iter(op2.nonlinear_cquad4_stress))  # get id of first subcase
            stresses = op2.nonlinear_cquad4_stress[subcase_id].data[-1, :, 5]  # von Mises stress is in the 6th column of the nonlinear stress array
            outputs['ks_stress'] = compute_ks_function(stresses, upper=yield_strength)  # von Mises stress must be less than yield stress for the material not to yield
            
            # Read eigenvalues of tangent stiffness matrix and aggregate with KS function
            try:  # try to read eigenvalues from f06 file
                f06_filepath = os.path.splitext(op2_filepath)[0] + '.f06'  # path to .f06 file
                eigenvalues = pynastran_utils.read_kllrh_lowest_eigenvalues_from_f06(f06_filepath)
            except ValueError:  # if eigenvalues are not in f06 file, read them from op2 file
                eigenvalues = pynastran_utils.read_kllrh_lowest_eigenvalues_from_op2(op2)
            outputs['ks_stability'] = compute_ks_function(
                eigenvalues[~np.isnan(eigenvalues)].flatten()*eigenvalue_scale_factor, lower_flag=True)  # nan values are discarded and eigenvalues are scaled before aggregation
            
            # Read final applied load factor
            load_factors, _, _ = pynastran_utils.read_load_displacement_history_from_op2(op2=op2)  # read load factors from op2 file
            outputs['load_factor'] = load_factors[subcase_id][-1]  # find load factor at the last converged increment
        
        elif bdf.sol == 144:
            # Create a deep copy of the original BDF object
            sol_105_bdf = bdf.__deepcopy__({})
            
            # Include the pch file with the forces obtained from the SOL 144
            # analysis
            pch_filepath = input_name + '.pch'
            sol_105_bdf.add_include_file(pch_filepath)
            
            # By default, the pch file generates FORCE cards with set ids 1 and
            # 2. The force set of interest is that with id 1, and we use id 3
            # for the method set id
            force_set_id = 1
            method_set_id = 3
            
            # Change the input name to avoid overwriting the original input file
            input_name = "sol_105_" + input_name
            
            # Run SOL 105 analysis and assign OP2 object to discrete output
            sol_105_op2 = pynastran_utils.run_sol_105(
                bdf=sol_105_bdf, input_name=input_name,
                analysis_directory_path=analysis_directory_path, 
                static_load_set_id=force_set_id, method_set_id=method_set_id,
                run_flag=run_flag)
            discrete_outputs['op2'] = sol_105_op2
            
            # Read von mises stresses and aggregate with KS function
            stresses = read_linear_stresses(op2)
            outputs['ks_stress'] = compute_ks_function(stresses, upper=yield_strength)  # von Mises stress must be less than yield stress for the material not to yield
            
            # Read buckling load factors and aggregate with KS function
            outputs['ks_buckling'] = compute_ks_function(
                np.array(sol_105_op2.eigenvectors[SECOND_SUBCASE_ID].eigrs), lower_flag=True, upper=1.)  # buckling load factor must be greater than 1 for the structure not to buckle
        
        else:
            raise ValueError("Invalid solution sequence number. Must be 105, 106 or SOL 144.")


class NastranGroup(om.Group):
    """
    An OpenMDAO Group that encapsulates the NastranSolver component. This allows for easy integration of the solver into larger OpenMDAO models or workflows.

    Attributes
    ----------
    options : dict
        Configuration options for the Nastran analysis, passed to the NastranSolver component.

    Methods
    -------
    initialize()
        Declares options for the group, including all necessary settings for running a Nastran analysis.
    setup()
        Configures the group, adding the NastranSolver component and setting up connections between inputs and outputs.
    """
    def initialize(self):
        """
        Initialize options for the Nastran analysis group. This includes specifying the BDF object, material properties, and other analysis configurations.
        """
        self.options.declare('bdf', types=BDF, desc='BDF object representing the Nastran model.')
        self.options.declare('analysis_directory_path', types=str, desc='Directory path for storing analysis files.')
        self.options.declare('input_name', types=str, desc='Name for the analysis input file.')
        self.options.declare('run_flag', types=bool, default=True, desc='Flag to control whether the analysis should be executed.')
        self.options.declare('yield_strength', types=float, desc='Yield strength of the material in MPa.')
        self.options.declare('eigenvalue_scale_factor', types=float, default=1., desc='Scaling factor applied to the eigenvalues of the tangent stiffness matrix before calculating the KS function.')

    def setup(self):
        """
        Setup the Nastran analysis group by adding the NastranSolver subsystem and configuring its options based on the group's settings.
        """
        self.add_subsystem('nastran_solver', NastranSolver(
            bdf=self.options['bdf'],
            analysis_directory_path=self.options['analysis_directory_path'],
            input_name=self.options['input_name'],
            run_flag=self.options['run_flag'],
            yield_strength=self.options['yield_strength'],
            eigenvalue_scale_factor=self.options['eigenvalue_scale_factor']))
        

def plot_optimization_history(
    recorder_filepath: str,
    variable_names: list[str],
    y_labels: list[str] | None = None
) -> tuple[Figure, Axes, dict[str, ndarray]]:
    """
    Plots the history of optimization variables and objectives from an
    OpenMDAO recorder file.

    Parameters
    ----------
    recorder_filepath : str
        Path to the OpenMDAO recorder file containing optimization
        history.
    variable_names : list
        List of variable names to plot.
    y_labels : list, optional
        List of y-axis labels for the plots. If provided, must have the
        same length as variable_names. Labels corresponding to invalid
        variable names will be ignored. If not provided, variable names
        will be used as labels.

    Returns
    -------
    tuple
        Contains:
        - Figure: matplotlib figure object
        - Axes: matplotlib axes object
        - dict: Dictionary containing the histories of the input
        variables

    Raises
    ------
    ValueError
        If none of the provided variable names exists in the
        optimization history or if y_labels is provided but has
        different length than variable_names
    """
    
    # Initialize the CaseReader object
    cr = om.CaseReader(recorder_filepath)
    
    # Extract driver cases without recursing into system or solver cases
    driver_cases = cr.get_cases('driver', recurse=False)
    
    # Get available output keys
    output_keys = list(driver_cases[0].outputs.keys())
    
    # Validate variable names
    valid_names = []
    invalid_names = []
    valid_labels = []  # Store labels for valid variables
    
    # Validate y_labels length if provided
    if y_labels is not None and len(y_labels) != len(variable_names):
        raise ValueError(
            f"Length of y_labels ({len(y_labels)}) must match length of "
            f"variable_names ({len(variable_names)})"
        )
    
    # Process variable names and corresponding labels
    for i, name in enumerate(variable_names):
        if name in output_keys:
            valid_names.append(name)
            # If y_labels is provided, use corresponding label, otherwise use
            # variable name
            if y_labels is not None:
                valid_labels.append(y_labels[i])
            else:
                valid_labels.append(name)
        else:
            invalid_names.append(name)
            
    # Warn about invalid names if any
    if invalid_names:
        warnings.warn(
            f"The following variable names were not found in the optimization "
            f"history and will be skipped: {invalid_names}"
        )
    
    # Raise error if no valid names
    if not valid_names:
        raise ValueError(
            f"None of the provided variable names {variable_names} was found "
            f"in the optimization history. Available variables are: "
            f"{output_keys}"
        )
    
    # Prepare data structures for plotting using only valid names
    histories = {
        key: np.array([case[key] for case in driver_cases]) 
        for key in valid_names
    }
    
    # Create figure and axes for subplots
    no_outputs = len(valid_names)
    fig, axes = plt.subplots(
        no_outputs, 1, sharex=True, figsize=(6.4, 4.8/4*no_outputs))
    fig.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    
    # Handle case of single subplot (axes is not array)
    if no_outputs == 1:
        axes = [axes]
    
    # Plot each history with corresponding label
    iterations_array = np.arange(len(next(iter(histories.values()))))
    for i, (key, label) in enumerate(zip(valid_names, valid_labels)):
        axes[i].plot(iterations_array, histories[key])
        axes[i].set(ylabel=label)
        axes[i].grid()
        axes[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        
    # Set x-axis label for the last subplot
    axes[-1].set(xlabel="Iteration")
    
    # Print final values of the optimization variables
    print("Design variables, constraints and objective at last iteration:")
    for key in histories:
        print(f"- {key}: {histories[key][-1]}")
    
    return fig, axes, histories
