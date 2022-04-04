# Import packages
import os
import time
from pyNastran.utils.nastran_utils import run_nastran
from pyNastran.op2.op2 import read_op2
import numpy as np


# Function to wait the Nastran analysis to finish
def wait_nastran(analysis_directory):
    # Wait for the analysis to start (when bat file appears)
    while not any([file.endswith('.bat') for file in os.listdir(analysis_directory)]):
        time.sleep(0.1)
    # Wait for the analysis to finish (when bat file disappears)
    while any([file.endswith('.bat') for file in os.listdir(analysis_directory)]):
        time.sleep(0.1)


# Function to run a Nastran analysis given a directory path, the BDF object, the name of the bdf file to be written and
# a flag whether to really run the analysis or not
def run_analysis(directory_path, bdf_object, bdf_filename, run_flag):
    # Create analysis directory
    os.makedirs(directory_path, exist_ok=True)
    # Write bdf file
    output_bdf_file_path = os.path.join(directory_path, bdf_filename + '.bdf')
    bdf_object.write_bdf(output_bdf_file_path)
    # Run Nastran
    nastran_path = 'C:\\Program Files\\MSC.Software\\MSC_Nastran\\2021.4\\bin\\nastranw.exe'
    run_nastran(output_bdf_file_path, nastran_path, run_in_bdf_dir=True, run=run_flag)
    if run_flag:
        wait_nastran(directory_path)


# Function to define a subcase associated to a certain load set
def create_static_load_subcase(model, subcase_id, subtitle, load_sid):
    model.create_subcases(subcase_id)
    model.case_control_deck.subcases[subcase_id].add('SUBTITLE', subtitle, [], 'STRING-type')
    model.case_control_deck.subcases[subcase_id].add_integer_type('LOAD', load_sid)


# Function to read equilibrium path data including history of load step, a displacement quantity and two load quantities
# from an op2 file
def read_equilibrium_path_data_from_op2(op2_file_path, displacement_node_id, displacement_index, load1_index,
                                        load2_index):
    # Read op2 file
    op2 = read_op2(op2_file_path, build_dataframe=True, debug=False)
    # Initialize dictionaries where the quantities of interest will be saved
    load_step = {}
    displacement = {}
    load1 = {}
    load2 = {}
    # Iterate through the subcases found in the op2 file
    for subcase_id in op2.load_vectors:
        load_step[subcase_id] = op2.load_vectors[subcase_id].lftsfqs
        displacement[subcase_id] = op2.displacements[subcase_id].data[:, displacement_node_id, displacement_index]
        # We add a minus sign so that P_x is positive along the direction defined in the sketches
        load1[subcase_id] = -np.apply_along_axis(np.sum, 1, op2.load_vectors[subcase_id].data[:, :, load1_index])
        load2[subcase_id] = np.apply_along_axis(np.sum, 1, op2.load_vectors[subcase_id].data[:, :, load2_index])
    # Return output data
    return load_step, displacement, load1, load2
