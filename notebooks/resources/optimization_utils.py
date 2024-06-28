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


# Constants for Nastran analysis subcase ids
FIRST_SUBCASE_ID = 1  # id of the first subcase, used in both SOL 105 and SOL 106 analyses
SECOND_SUBCASE_ID = 2  # id of the second subcase, specifically for SOL 105 analyses


def find_linear_stresses(op2:OP2) -> ndarray:
    """
    Extracts linear stresses from the OP2 object.

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
        sigma_y : float
            Yield strength of the material.
        analysis_directory_path : str
            Path to the directory where the analysis files are stored.
        input_name : str
            Name of the input file for the analysis.
        run_flag : bool
            Flag to indicate if the analysis should be run.
        """
        self.options.declare('bdf', types=BDF, desc='BDF object representing the Nastran model.')
        self.options.declare('sigma_y', types=float, desc='Yield strength of the material.')
        self.options.declare('analysis_directory_path', types=str, desc='Directory path for storing analysis files.')
        self.options.declare('input_name', types=str, desc='Name for the analysis input file.')
        self.options.declare('run_flag', types=bool, default=True, desc='Flag to control whether the analysis should be executed.')

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
        if self.options['bdf'].sol == 105:
            self.add_output('ks_buckling', desc='KS aggregated buckling load factor.')
        elif self.options['bdf'].sol == 106:
            self.add_output('ks_stability', desc='KS aggregated stability metric.')
            self.add_output('applied_load', desc='Magnitude of the applied load in the final state of the analysis.')
        else:
            raise ValueError("Unsupported solution sequence. Must be SOL 105 for buckling or SOL 106 for nonlinear static analysis.")

    def setup_partials(self):
        """
        Declare partial derivatives for the component using finite difference method.
        """
        # Finite difference all partials
        self.declare_partials('*', '*', method='fd', step=1e-6)

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
        sigma_y = self.options['sigma_y']
        analysis_directory_path = self.options['analysis_directory_path']
        input_name = self.options['input_name']
        run_flag = self.options['run_flag']
        # Assign thickness values to the property cards
        t_array = inputs['t_val']
        for i, pid in enumerate(bdf.properties):
            if bdf.properties[pid].type == 'PSHELL':
                bdf.properties[pid].t = t_array[0, i]
                bdf.properties[pid].z1 = -t_array[0, i]/2
                bdf.properties[pid].z2 = t_array[0, i]/2
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
            # Read von mises stresses and aggregate with KS function
            stresses = find_linear_stresses(op2)
            outputs['ks_stress'] = compute_ks_function(stresses, upper=sigma_y)  # von Mises stress must be less than yield stress for the material not to yield
            # Read buckling load factors and aggregate with KS function
            outputs['ks_buckling'] = compute_ks_function(np.array(op2.eigenvectors[SECOND_SUBCASE_ID].eigrs),
                                                        lower_flag=True, upper=1.)  # buckling load factor must be greater than 1 for the structure not to buckle
        elif bdf.sol == 106:
            # Find von mises stresses and aggregate with KS function
            stresses = op2.nonlinear_cquad4_stress[FIRST_SUBCASE_ID].data[-1, :, 5]
            outputs['ks_stress'] = compute_ks_function(stresses, upper=sigma_y)  # von Mises stress must be less than yield stress for the material not to yield
            # Read eigenvalues of tangent stiffness matrix and aggregate with KS function
            f06_filepath = os.path.splitext(op2_filepath)[0] + '.f06'  # path to .f06 file
            eigenvalues = pynastran_utils.read_kllrh_lowest_eigenvalues_from_f06(f06_filepath)  # read eigenvalues from f06 files
            outputs['ks_stability'] = compute_ks_function(eigenvalues[~np.isnan(eigenvalues)].flatten()*1e3, lower_flag=True)  # nan values are discarded and eigenvalues are converted from N/mm to N/m
            # Calculate final applied load magnitude
            _, applied_loads, _ = pynastran_utils.read_load_displacement_history_from_op2(op2=op2)
            outputs['applied_load'] = np.linalg.norm(applied_loads[FIRST_SUBCASE_ID][-1, :])  # calculate magnitude of applied load at last converged increment of the analysis
        else:
            raise ValueError("Invalid solution sequence number. Must be 105 or 106.")


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
        self.options.declare('sigma_y', types=float, desc='Yield strength of the material in MPa.')
        self.options.declare('analysis_directory_path', types=str, desc='Directory path for storing analysis files.')
        self.options.declare('input_name', types=str, desc='Name for the analysis input file.')
        self.options.declare('run_flag', types=bool, default=True, desc='Flag to control whether the analysis should be executed.')

    def setup(self):
        """
        Setup the Nastran analysis group by adding the NastranSolver subsystem and configuring its options based on the group's settings.
        """
        self.add_subsystem('nastran_solver', NastranSolver(
            bdf=self.options['bdf'],
            sigma_y=self.options['sigma_y'],
            analysis_directory_path=self.options['analysis_directory_path'],
            input_name=self.options['input_name'],
            run_flag=self.options['run_flag']))
        

def plot_optimization_history(recorder_filepath:str):
    """
    Plots the history of optimization variables and objectives from an OpenMDAO recorder file.

    Parameters
    ----------
    recorder_filepath : str
        Path to the OpenMDAO recorder file containing optimization history.

    Returns
    -------
    dict
        A dictionary containing the optimization history for each variable and objective.
    """
    
    # Initialize the CaseReader object
    cr = om.CaseReader(recorder_filepath)
    
    # Extract driver cases without recursing into system or solver cases
    driver_cases = cr.get_cases('driver', recurse=False)
    
    # Prepare data structures for plotting
    output_keys = list(driver_cases[0].outputs.keys())
    no_outputs = len(output_keys)
    histories = {key: np.array([case[key] for case in driver_cases]) for key in output_keys}  # retrieve histories of the functions
    
    # Setup plot labels
    y_labels = ["$t$, mm"]
    if no_outputs == 4:
        y_labels = y_labels + ["$KS_{BLF}$", "$KS_{\sigma}$, MPa", "$m$, ton"]  # add labels for linear buckling optimization problem
    else:
        y_labels = y_labels + ["$P/P_\mathrm{design}$", "$KS_{\lambda}$, N/m", "$KS_{\sigma}$, MPa","$m$, ton"]  # add labels for nonlinear structural stability optimization problem
        histories['nastran_solver.applied_load'] = histories['nastran_solver.applied_load']/histories['nastran_solver.applied_load'][0]  # normalize applied load by initial value
            
    # Create figure and axes for subplots
    fig, axes = plt.subplots(no_outputs, 1, sharex=True)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    
    # Plot each history and show plot
    iterations_array = np.arange(len(next(iter(histories.values()))))
    for i, key in enumerate(histories):
        axes[i].plot(iterations_array, histories[key])
        axes[i].set(ylabel=y_labels[i])
        axes[i].grid()
    axes[-1].set(xlabel="Iteration")
    plt.show()
    
    # Print final values of the optimization variables and objectives
    print("Design variables, constraints and objective at last iteration:")
    for key in histories:
        print(f"- {key}: {histories[key][-1]}")
    
    # Return the figure and histories
    return fig, histories
