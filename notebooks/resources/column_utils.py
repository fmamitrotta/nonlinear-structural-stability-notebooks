from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
import numpy as np
import matplotlib.pyplot as plt


def create_base_bdf(young_modulus: float, poisson_ratio: float, density: float, diameter: float, length: float,
                    no_elements: int) -> BDF:
    """
    Returns a BDF object with material properties, nodes, elements, boundary conditions and applied forces of Euler's
     column.

    Parameters
    ----------
    young_modulus: float
        Young's modulus of the column
    poisson_ratio: float
        Poisson's ratio of the column
    density: float
        density of the column
    diameter: float
        diameter of the column
    length: float
        length of the column
    no_elements: int
        number of beam elements for the discretization of the column

    Returns
    -------
    bdf_input: BDF
        object representing the Nastran input of Euler's column
    """
    # Create an instance of the BDF class without debug or info messages
    bdf_input = BDF(debug=None)
    # Define isotropic material properties with MAT1 card
    material_id = 1
    bdf_input.add_mat1(mid=material_id, E=young_modulus, G='', nu=poisson_ratio, rho=density)
    # Define element properties with PBEAML card
    property_id = 1
    beam_type = 'ROD'
    bdf_input.add_pbeaml(pid=property_id, mid=material_id, beam_type=beam_type, xxb=[0.], dims=[[diameter / 2]])
    # Define nodes and beam elements with GRID and CBEAM cards
    nodes_x_coordinates = np.linspace(0, length, no_elements + 1)
    element_orientation_vector = [0., 0., 1.]
    bdf_input.add_grid(nid=1, xyz=[0., 0., 0.])
    for i in range(no_elements):
        bdf_input.add_grid(nid=i + 2, xyz=[nodes_x_coordinates[i + 1], 0., 0.])
        bdf_input.add_cbeam(eid=i + 1, pid=property_id, nids=[i + 1, i + 2], x=element_orientation_vector, g0=None)
    # Define permanent single point constraints for all nodes to consider a two-dimensional problem in the xy plane
    bdf_input.add_grdset(cp='', cd='', ps='345', seid='')
    # Define single-point constraint for pin support
    pin_support_set_id = 1
    bdf_input.add_spc1(pin_support_set_id, '12', [1])
    # Defins single-point constraint for roller support
    roller_support_set_id = pin_support_set_id + 1
    roller_supported_node_id = no_elements + 1
    bdf_input.add_spc1(roller_support_set_id, '2', [roller_supported_node_id])
    # Combine constraints with a SPCADD card
    spcadd_set_id = roller_support_set_id + 1
    bdf_input.add_spcadd(spcadd_set_id, [pin_support_set_id, roller_support_set_id])
    # Assign constraint to global subcase
    bdf_input.create_subcases(0)
    bdf_input.case_control_deck.subcases[0].add_integer_type('SPC', spcadd_set_id)
    # Define compression force at the roller-supported node
    force_set_id = spcadd_set_id + 1
    force_magnitude = 1.  # [N]
    force_direction = [-1., 0., 0.]
    bdf_input.add_force(sid=force_set_id, node=roller_supported_node_id, mag=force_magnitude, xyz=force_direction)
    # Define parameters to store results in op2 file
    bdf_input.add_param('POST', [1])
    bdf_input.case_control_deck.subcases[0].add_result_type('DISPLACEMENT', 'ALL', ['PLOT'])  # store displacements
    bdf_input.case_control_deck.subcases[0].add_result_type('OLOAD', 'ALL', ['PLOT'])  # store applied load vector
    # Request neither sorted nor unsorted Bulk Data to be printed
    bdf_input.case_control_deck.subcases[0].add('ECHO', 'NONE', [], 'STRING-type')
    # Return BDF object
    return bdf_input


def plot_buckling_mode(op2_object: OP2):
    """
    Plot first buckling mode of the column.

    Parameters
    ----------
    op2_object: OP2
        object representing Nastran output created reading an op2 file with the load_geometry option set to True
    """
    # Create new figure
    fig, ax = plt.subplots()
    # Calculate coordinates of nodes in the buckling shape
    nodes_xy_coordinates = np.vstack([op2_object.nodes[index].xyz[0:2] for index in op2_object.nodes.keys()]) + \
                           np.squeeze([*op2_object.eigenvectors.values()][0].data[0, :, 0:2])
    # Plot nodes
    ax.plot(nodes_xy_coordinates[:, 0], nodes_xy_coordinates[:, 1], '.-')
    # Set axes labels and grid
    plt.xlabel('$x$ [mm]')
    plt.ylabel('Nondimensional displacement along $y$')
    plt.grid()
    # Show plot
    plt.show()
