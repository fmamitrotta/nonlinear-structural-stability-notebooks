from pyNastran.bdf.bdf import BDF
from numpy import ndarray
import numpy as np
from notebooks.resources import pynastran_utils
import os
from pyNastran.op2.op2 import read_op2
import matplotlib.pyplot as plt


def create_base_bdf_input(young_modulus, poisson_ratio, density, shell_thickness, width, span, height, cquad4_length)\
        -> (BDF, ndarray, list, ndarray):
    """
    Returns a BDF object with material properties, nodes, elements, boundary conditions and output files defaults for
    the box beam model.

            Parameters:
                    young_modulus (float): Young's modulus of the column
                    poisson_ratio (float): Poisson's ratio of the column
                    density (float): density of the column
                    shell_thickness (float): thickness of the shell elements composing the box beam
                    width (float): width of the box beam section
                    span (float): span of the box beam
                    height (float): height of the box beam section
                    cquad4_length (float): prescribed linear size of the CQUAD4 elements

            Returns:
                    bdf_input (BDF): BDF object of the box beam model
                    nodes_id_array (ndarray): array with the id of the nodes
                    edge_indices (list): list of indices of the nodes' id array corresponding to the edges of the box
                     beam
                    elements_id_array (ndarray): array with the id of the elements

    """
    # Create BDF object
    bdf_input = BDF(debug=False)
    # Add material card
    material_id = 1
    bdf_input.add_mat1(mid=material_id, E=young_modulus, G='', nu=poisson_ratio, rho=density)
    # Add element property card
    property_id = 1
    bdf_input.add_pshell(pid=property_id, mid1=material_id, t=shell_thickness, mid2=material_id, mid3=material_id)
    # Define coordinates of the vertices of the box based on the input geometrical properties
    vertices_xyz = np.array([(0., 0., height/2),         # vertex 0
                            (width, 0., height/2),       # 1
                            (width, 0., -height/2),      # 2
                            (0., 0., -height/2),         # 3
                            (0., span, height/2),        # 4
                            (width, span, height/2),     # 5
                            (width, span, -height/2),    # 6
                            (0., span, -height/2)])      # 7
    # Define faces by sequence of vertices
    faces = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    # Define number of nodes along the span of the box
    no_nodes_span = int(np.rint((np.linalg.norm(vertices_xyz[4] - vertices_xyz[0])/cquad4_length)))
    # Initialize array with nodes' id
    nodes_id_array = np.empty((no_nodes_span, 0), dtype=int)
    # Initialize variable with id of last defined node
    last_node_id = 0
    # Initailize list with the column indices of the node_ids array associated with the edges of the box
    edge_indices = []
    # Define nodes of the mesh for each face of the box
    for face in faces:
        # Define number of nodes along the width or height of the box
        no_points_chord_height = int(np.rint((np.linalg.norm(vertices_xyz[face[1]] - vertices_xyz[face[0]]) /
                                              cquad4_length)))
        # Find coordinates of nodes along the root and the tip of the current face
        root_nodes_xyz = np.linspace(vertices_xyz[face[0]], vertices_xyz[face[1]], no_points_chord_height)
        tip_nodes_xyz = np.linspace(vertices_xyz[face[3]], vertices_xyz[face[2]], no_points_chord_height)
        # Find coordinates of nodes along the span for each couple of root-tip points of the current face
        for j in range(no_points_chord_height-1):
            nodes_xyz = np.linspace(root_nodes_xyz[j], tip_nodes_xyz[j], no_nodes_span)
            # Generate ids for current set of nodes
            nodes_id_array = np.hstack((nodes_id_array, np.arange(last_node_id+1,
                                                                  last_node_id+no_nodes_span+1)[:, None]))
            # Update id of last defined node
            last_node_id = last_node_id + no_nodes_span
            for row in range(no_nodes_span):
                bdf_input.add_grid(nodes_id_array[row, -1], nodes_xyz[row])
        # Save column index corresponding to the edge of the current face
        edge_indices.append(np.size(nodes_id_array, 1))
    # Add last column of nodes' ids
    nodes_id_array = np.hstack((nodes_id_array, np.arange(1, no_nodes_span+1)[:, None]))
    # Initialize array with elements' id
    elements_id_array = np.empty(tuple(dim - 1 for dim in nodes_id_array.shape), dtype=int)
    # Define shell elements (CQUAD4 cards)
    for col in range(np.size(nodes_id_array, 1) - 1):
        for row in range(np.size(nodes_id_array, 0) - 1):
            elements_id_array[row, col] = 1 + col * np.size(elements_id_array, 0) + row
            bdf_input.add_cquad4(eid=elements_id_array[row, col], pid=property_id, nids=[nodes_id_array[row, col],
                                                                                         nodes_id_array[row + 1, col],
                                                                                         nodes_id_array[row + 1, col + 1],
                                                                                         nodes_id_array[row, col + 1]])
    # Define boundary conditions constraining the nodes at the root
    constraint_set_id = 1
    root_nodes_ids = list(nodes_id_array[0, 0:-2])
    bdf_input.add_spc1(constraint_set_id, '123456', root_nodes_ids)
    bdf_input.create_subcases(0)
    bdf_input.case_control_deck.subcases[0].add_integer_type('SPC', constraint_set_id)
    # Set defaults for output files
    bdf_input.add_param('POST', [1])  # add PARAM card to store results in a op2 file
    bdf_input.case_control_deck.subcases[0].add('ECHO', 'NONE', [],
                                                'STRING-type')  # request no Bulk Data to be printed
    bdf_input.case_control_deck.subcases[0].add_result_type('DISPLACEMENT', 'ALL', [
        'PLOT'])  # store displacement data of all nodes in the op2 file
    bdf_input.case_control_deck.subcases[0].add_result_type('SPCFORCES', 'ALL', [
        'PLOT'])  # store single point constraint forces data in the op2 file
    bdf_input.case_control_deck.subcases[0].add_result_type('OLOAD', 'ALL',
                                                            ['PLOT'])  # store form and type of applied load vector
    # Return BDF object
    return bdf_input, nodes_id_array, edge_indices, elements_id_array


def set_up_sol_106(bdf_input):
    # Assign solution sequence
    bdf_input.sol = 106
    # Add parameter for large displacement effects
    bdf_input.add_param('LGDISP', [1])
    # Define general parameters for the nonlinear iteration strategy
    nlparm_id = 1
    bdf_input.add_nlparm(nlparm_id=nlparm_id, kmethod='ITER', kstep=1, int_out='YES')
    # Define parameters for the arc-length method
    bdf_input.add_nlpci(nlpci_id=nlparm_id, Type='CRIS', mxinc=100)
    # Add NLPARM id to the control case commands
    bdf_input.case_control_deck.subcases[0].add_integer_type('NLPARM', nlparm_id)


def calculate_linear_buckling_load(bdf_input, static_load_set_id, analysis_directory_path, input_name, run_flag=True,
                                   plot_shape=False):
    # Set solution sequence for linear buckling analysis (SOL 105)
    bdf_input.sol = 105
    # Create first subcase for the application of the static load
    load_application_subcase_id = 1
    pynastran_utils.create_static_load_subcase(bdf_object=bdf_input, subcase_id=load_application_subcase_id,
                                               load_set_id=static_load_set_id)
    # Add EIGRL card to define the parameters for the eigenvalues calculation
    eigrl_set_id = static_load_set_id+1
    bdf_input.add_eigrl(sid=eigrl_set_id, v1=0., nd=1)
    # Create second subcase for the calculation of the eigenvalues
    eigenvalue_calculation_subcase_id = 2
    bdf_input.create_subcases(eigenvalue_calculation_subcase_id)
    bdf_input.case_control_deck.subcases[eigenvalue_calculation_subcase_id].add_integer_type('METHOD', eigrl_set_id)
    # Run analysis
    pynastran_utils.run_analysis(directory_path=analysis_directory_path, bdf_object=bdf_input, bdf_filename=input_name,
                                 run_flag=run_flag)
    # Read op2 file
    op2_filepath = os.path.join(analysis_directory_path, input_name + '.op2')
    op2_output = read_op2(op2_filename=op2_filepath, load_geometry=True, debug=False)
    # Find buckling load and print it
    buckling_load = op2_output.eigenvectors[eigenvalue_calculation_subcase_id].eigr
    print(f'Buckling load: {buckling_load:.0f} N')
    # Plot buckling shape if requested
    if plot_shape:
        print(f'Buckling shape:')
        ax = pynastran_utils.plot_buckling_shape(op2_object=op2_output, subcase=eigenvalue_calculation_subcase_id)
        # Adjust number of ticks and distance from axes
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='z', nbins=2)
        ax.tick_params(axis='y', which='major', pad=15)
        plt.show()
    # Return buckling load
    return buckling_load
