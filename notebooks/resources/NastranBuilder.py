from pyNastran.bdf.bdf import BDF
from numpy import ndarray
import numpy as np
import openmdao.api as om
from pyNastran.op2.op2 import read_op2
from resources import pynastran_utils
import os
from pyNastran.bdf.mesh_utils.mass_properties import mass_properties  # pyNastran function to calculate mass properties
from mpi4py import MPI  # import MPI module from mpi4py library
from mphys import Builder  # import Builder class from mphys library
from resources.optimization_utils import compute_ks_function  # function to compute KS function


FIRST_SUBCASE_ID = 1   # ID of the first subcase in the op2 file
SECOND_SUBCASE_ID = 2  # ID of the second subcase in the op2 file
FULL_VEHICLE_MONITOR_POINT = "AEROSG2D"  # name of the monitor point


def get_original_nodes_coordinates(bdf: BDF) -> ndarray:
    """
    Function to extract the original coordinates of the nodes from a BDF
    object.

    Parameters
    ----------
    bdf : BDF
        BDF object representing the Nastran input file.

    Returns
    -------
    ndarray
        Array containing the original coordinates of the nodes.
    """
    if bdf is not None:
        xpts = np.array(
            [node.get_position() for node in bdf.nodes.values()]).flatten()
    else: 
        xpts = np.zeros(0)
    return xpts


class NastranModel:
    """
    Class to store the Nastran model and analysis settings.
    """
    def __init__(
        self, analysis_directory_path, input_name, bdf, 
        run_flag, trim_id=None, comm=MPI.COMM_WORLD):
        """
        Initialize the Nastran model with the analysis settings.
        """
        # Assign input values to attributes
        self.analysis_directory_path = analysis_directory_path
        self.input_name = input_name
        self.bdf = bdf
        self.run_flag = run_flag
        self.trim_id = trim_id
        self.comm = comm
        self.n_dof = 6
        
    def run_nastran(self):
        """
        Run Nastran analysis.
        """
        # Assign design variables (wall thickness)
        for i, pid in enumerate(self.bdf.properties):
            self.bdf.properties[pid].t = self.dv_struct[i]
            self.bdf.properties[pid].z1 = None
            self.bdf.properties[pid].z2 = None
            
        # Assign angle of attack if SOL 144 is used
        if self.bdf.sol == 144:
            self.bdf.trims[self.trim_id].uxs[0] = np.deg2rad(self.aoa[0])
        
        # Run Nastran analysis
        pynastran_utils.run_analysis(
            directory_path=self.analysis_directory_path,
            filename=self.input_name, bdf=self.bdf, run_flag=self.run_flag)
        
        # Read OP2 file and store OP2 object in solver
        op2_filepath = os.path.join(
            self.analysis_directory_path, self.input_name + '.op2')
        self.op2 = read_op2(
            op2_filename=op2_filepath, load_geometry=True, debug=None)
        
        # Return structural displacements
        displacements = self.op2.displacements[FIRST_SUBCASE_ID].data[-1, :, :]
        return displacements.flatten()


class NastranMesh(om.IndepVarComp):
    """
    Component to read the initial grid coordinates
    """

    def initialize(self):
        self.options.declare(
            "bdf",
            default=None,
            desc="BDF object representing Nastran input",
            recordable=False,
        )

    def setup(self):
        bdf = self.options["bdf"]
        xpts = get_original_nodes_coordinates(bdf)
        self.add_output(
            "x_struct0",
            distributed=True,
            val=xpts,
            shape=xpts.size,
            desc="structural node coordinates",
            tags=["mphys_coordinates"],
        )


class NastranSolver(om.ExplicitComponent):
    """
    Component to evaluate run a Nastran analysis based on the defined bdf object.

    Attributes
    ----------
    options : dict
        A dictionary of options for the component.

    Methods
    -------
    initialize()
        Declare options for the component.
    setup()
        Define the component's inputs and outputs.
    setup_partials()
        Declare partial derivatives for the component.
    compute(inputs, outputs, discrete_inputs, discrete_outputs)
        Run SOL 105 and calculate output functions.
    """

    def initialize(self):
        self.options.declare('solver')

    def setup(self):
        # Assign component options
        self.solver = self.options['solver']
        # Define inputs and outputs
        self.add_input('dv_struct', shape_by_conn=True, tags=['mphys_input'])
        self.add_input(
            'x_struct0', shape_by_conn=True, tags=['mphys_coordinates'])
        self.add_output(
            'u_struct', np.zeros(6*len(self.options['solver'].bdf.nodes)),
            tags=['mphys_coupling'])
        # Add input for angle of attack if SOL 144 is used
        if self.solver.bdf.sol == 144:
            self.add_input('aoa', tags=['mphys_input'], units='deg')

    def compute(self, inputs, outputs):
        """
        Assign wall thickness value, run SOL 106 and calculate output functions.

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
        # Assign input values to solver attributes
        self.solver.dv_struct = inputs['dv_struct']
        self.solver.xyz = inputs['x_struct0']
        # Assign angle of attack if SOL 144 is used
        if self.solver.bdf.sol == 144:
            self.solver.aoa = inputs['aoa']
        
        # Run Nastran analysis
        outputs['u_struct'] = self.solver.run_nastran()


class NastranMass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('solver')

    def setup(self):
        # Assign component options
        self.solver = self.options['solver']
        # Define inputs and outputs
        self.add_input(
            'dv_struct', shape_by_conn=True, tags=['mphys_input'])
        self.add_input(
            'x_struct0', shape_by_conn=True, tags = ['mphys_coordinates'])
        self.add_output('mass', tags=['mphys_result'])
    
    def setup_partials(self):
        """
        Declare partial derivatives for the component using finite
        difference method.
        """
        # Finite difference all partials
        self.declare_partials('*', '*', method='fd', step=1e-6)

    def compute(self, inputs, outputs):
        outputs['mass'] = mass_properties(self.solver.bdf)[0]


class NastranFunctions(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('solver')

    def setup(self):
        # Assign component options
        self.solver = self.options['solver']
        
        # Define inputs
        self.add_input(
            'dv_struct', shape_by_conn=True, tags=['mphys_input'])
        self.add_input(
            'x_struct0', shape_by_conn=True, tags = ['mphys_coordinates'])
        self.add_input(
            'yield_strength', shape_by_conn=True, tags=['mphys_input'])
        
        # Define outputs based on the solution sequence number
        if self.solver.bdf.sol in [105, 106]:
            self.add_output('ks_stress', tags=['mphys_result'])
            if self.solver.bdf.sol == 105:
                self.add_output('ks_buckling', tags=['mphys_result'])
            elif self.solver.bdf.sol == 106:
                self.add_output('ks_stability', tags=['mphys_result'])
                self.add_output(
                    'load_factor',
                    desc='Ratio of the applied load to the target load at the \
                        final load step of the nonlinear analysis',
                    tags=['mphys_result'])
        # Define input and output for SOL 144
        elif self.solver.bdf.sol == 144:
            self.add_input('aoa', tags=['mphys_input'], units='deg')
            self.add_output('lift', tags=['mphys_result'])
        # Raise error if solution sequence number is invalid
        else:
            raise ValueError("Invalid solution sequence number. Must be 105,\
                106 or 144.")
    
    def setup_partials(self):
        """
        Declare partial derivatives for the component using finite
        difference method.
        """
        # Finite difference all partials
        self.declare_partials('*', '*', method='fd', step=1e-6)

    def compute(self, inputs, outputs):
        # Calculate output functions for SOL 105
        if self.solver.bdf.sol == 105:
            # Read buckling load factors and aggregate with KS function
            blf_values = self.solver.op2.eigenvectors[SECOND_SUBCASE_ID].eigrs
            outputs['ks_buckling'] = compute_ks_function(
                np.array(blf_values), lower_flag=True, upper=1.)  # buckling load factor must be greater than 1 for the structure not to buckle
            
            # Read von Mises stresses and aggregate with KS function
            stresses = self.solver.op2.op2_results.stress.cquad4_stress[
                FIRST_SUBCASE_ID].data[0, :, 7]  # von Mises stress is in the 8th column of the stress array
            outputs['ks_stress'] = compute_ks_function(
                stresses, upper=inputs['yield_strength'])  # von Mises stress must be less than yield stress for the material not to yield
        
        # Calculate output functions for SOL 106
        elif self.solver.bdf.sol == 106:
            # Read eigenvalues of tangent stiffness matrix and aggregate with
            # KS function
            f06_filepath = os.path.join(
                self.solver.analysis_directory_path,
                self.solver.input_name + '.f06')  # path to .f06 file
            eigenvalues =\
                pynastran_utils.read_kllrh_lowest_eigenvalues_from_f06(
                    f06_filepath)  # read eigenvalues from f06 files
            outputs['ks_stability'] = compute_ks_function(eigenvalues[
                ~np.isnan(eigenvalues)].flatten(), lower_flag=True)  # nan values are discarded
            
            # Read nonlinear von Mises stresses and aggregate with KS function
            stresses = self.solver.op2.nonlinear_cquad4_stress[
                FIRST_SUBCASE_ID].data[-1, :, 5]  # von Mises stress is in the 6th column of the stress array
            outputs['ks_stress'] = compute_ks_function(
                stresses, upper=inputs['yield_strength'])  # von Mises stress must be less than yield stress for the material not to yield
            
            # Output final load step
            load_steps, _, _ =\
                pynastran_utils.read_load_displacement_history_from_op2(
                    op2=self.solver.op2)
            outputs['load_factor'] = load_steps[FIRST_SUBCASE_ID][-1]
            
        # Calculate output functions for SOL 144
        elif self.solver.bdf.sol == 144:
            # Read total normal force from f06 file
            f06_filepath = os.path.join(
                self.solver.analysis_directory_path,
                self.solver.input_name + '.f06')  # path to .f06 file
            aero_loads_dict = pynastran_utils.read_monitor_point_from_f06(
                f06_path=f06_filepath,
                monitor_point_name=FULL_VEHICLE_MONITOR_POINT)
            normal_force = aero_loads_dict['CZ'][1]
            # Calculate lift force - multiply by 2 to account for symmetry
            outputs['lift'] =\
                2*normal_force*np.cos(np.deg2rad(inputs['aoa'][0]))
            
        # Raise error if solution sequence number is invalid
        else:
            raise ValueError("Invalid solution sequence number. Must be 105, \
                106 or 144.")


class NastranCouplingGroup(om.Group):
    def initialize(self):
        self.options.declare('solver')

    def setup(self):
        self.add_subsystem('solver', NastranSolver(
            solver=self.options['solver']), promotes=['*'])


class NastranPostcouplingGroup(om.Group):
    def initialize(self):
        self.options.declare('solver')

    def setup(self):
        self.add_subsystem(
            'struct_mass', NastranMass(solver=self.options['solver']),
            promotes=['*'])

        self.add_subsystem(
            'struct_function', NastranFunctions(solver=self.options['solver']),
            promotes=['*'])


class NastranBuilder(Builder):
    def __init__(
        self, analysis_directory_path, input_name, bdf, run_flag=True,
        trim_id=None):
        """
        Initialize the builder with the Nastran model and analysis settings.
        """
        self.analysis_directory_path = analysis_directory_path
        self.input_name = input_name
        self.bdf = bdf
        self.run_flag = run_flag
        self.trim_id = trim_id
    
    def initialize(self, comm=MPI.COMM_WORLD):
        """
        Initialize the builder with the MPI communicator.
        """
        self.solver = NastranModel(
            analysis_directory_path=self.analysis_directory_path,
            input_name=self.input_name,
            bdf=self.bdf,
            run_flag=self.run_flag,
            trim_id=self.trim_id,
            comm=comm)

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        """
        The subsystem that contains the subsystem that will return the
        mesh coordinates

        Parameters
        ----------
        scenario_name : str, optional
            The name of the scenario calling the builder.

        Returns
        -------
        mesh : openmdao.api.Group
            The openmdao subsystem that has an output of coordinates.
        """
        return NastranMesh(bdf=self.solver.bdf)
    
    def get_coupling_group_subsystem(self, scenario_name=None):
        """
        Return the coupling group subsystem that performs the Nastran SOL 106 analysis.
        """
        return NastranCouplingGroup(solver=self.solver)

    def get_post_coupling_subsystem(self, scenario_name=None):
        """
        Define any post-processing steps after the structural analysis.
        """
        return NastranPostcouplingGroup(solver=self.solver)

    def get_ndof(self):
        """
        Return the number of degrees of freedom per node, equal to 6 (3 translations and 3 rotations).
        """
        return self.solver.ndof
