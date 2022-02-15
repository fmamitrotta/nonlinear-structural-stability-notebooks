# Import packages
from pyNastran.bdf.bdf import BDF
import numpy as np
import os
import time


# Function to create the base model of a column with constant cross-sectional properties
def create_column_base_model(young_modulus, poisson_ratio, density, diameter, length, no_elements):

    # Create an instance of the BDF class
    base_model = BDF(debug=False)

    # Define material properties with MAT1 card
    mid = 1
    base_model.add_mat1(mid, young_modulus, G='', nu=poisson_ratio, rho=density)

    # Define element properties with PBEAML card
    pid = 1
    beam_type = 'ROD'
    base_model.add_pbeaml(pid, mid, beam_type, xxb=[0.], dims=[[diameter/2]])

    # Define nodes and beam elements (GRID and CBEAM cards)
    nodes_x_coordinates = np.linspace(0, length, no_elements + 1)
    element_orientation_vector = [0., 0., 1.]
    base_model.add_grid(1, [0., 0., 0.])
    for i in range(no_elements):
        base_model.add_grid(i + 2, [nodes_x_coordinates[i + 1], 0., 0.])
        base_model.add_cbeam(i + 1, pid, [i + 1, i + 2], element_orientation_vector, g0=None)

    # Define permanent single point constraints for all nodes to consider a two-dimensional problem in the xy plane
    base_model.add_grdset(cp='', cd='', ps='345', seid='')

    # Define two single-point constraints with SPC1 cards (pin support at one end and roller support at the other end)
    # and combine these two constraints with a SPCADD card
    pin_support_sid = 1
    roller_support_sid = pin_support_sid + 1
    roller_supported_node_id = no_elements + 1
    spcadd_sid = roller_support_sid + 1
    base_model.add_spc1(pin_support_sid, '12', 1)
    base_model.add_spc1(roller_support_sid, '2', roller_supported_node_id)
    base_model.add_spcadd(spcadd_sid, [pin_support_sid, roller_support_sid])

    # Define parameters
    # Store results in op2 file
    base_model.add_param('POST', [1])
    # Print maximums of applied loads, single-point forces of constraint, multipoint forces of constraint, and
    # displacements in the f06 file
    base_model.add_param('PRTMAXIM', ['YES'])

    # Define defaults for all subcases
    base_model.create_subcases(0)
    # id of single-point constraint to apply
    base_model.case_control_deck.subcases[0].add_integer_type('SPC', spcadd_sid)
    # Request neither sorted nor unsorted Bulk Data to be printed
    base_model.case_control_deck.subcases[0].add('ECHO', 'NONE', [], 'STRING-type')
    # Store displacement data of all nodes in the op2 file
    base_model.case_control_deck.subcases[0].add_result_type('DISPLACEMENT', 'ALL', ['PLOT'])
    # Store form and type of applied load vector
    base_model.case_control_deck.subcases[0].add_result_type('OLOAD', 'ALL', ['PLOT'])

    # Return BDF object
    return base_model


# Function to wait the Nastran analyis to finish
def wait_nastran(analysis_directory):
    # Wait for the analysis to start (when bat file appears)
    while not any([file.endswith('.bat') for file in os.listdir(analysis_directory)]):
        time.sleep(0.1)
    # Wait for the analysis to finish (when bat file disappears)
    while any([file.endswith('.bat') for file in os.listdir(analysis_directory)]):
        time.sleep(0.1)


# Function to define a subcase
def create_column_subcase(model, subcase_id, subtitle, load_sid, force_sids, scale_factors):
    model.create_subcases(subcase_id)
    model.case_control_deck.subcases[subcase_id].add('SUBTITLE', subtitle, [], 'STRING-type')
    model.add_load(load_sid, 1., scale_factors, force_sids)
    model.case_control_deck.subcases[subcase_id].add_integer_type('LOAD', load_sid)
