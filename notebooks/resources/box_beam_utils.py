from pyNastran.bdf.bdf import BDF
from numpy import ndarray
import numpy as np
from pyNastran.op2.op2 import read_op2
import pyvista
from pyvista import PolyData


def mesh_box(width: float, height: float, span: float, edge_length: float) -> tuple[ndarray, ndarray]:
    """
    Discretizes a box with the input dimensions into quadrilateral shell elements returning an array with the xyz
    coordinates of the nodes and an array with their connectivity information.

    Parameters
    ----------
    width: float
        box width
    height: float
        box height
    span: float
        box span
    edge_length: float
     prescribed length of the edges of the shell elements used to discretize the geometry

    Returns
    -------
    nodes_xyz_array: ndarray
        array with the xyz coordinates of the nodes, respectively on the first, second and third column
    nodes_connectivity_matrix: ndarray
        array with the connectivity information of the nodes. Each row contains the combination of the indices of
        nodes_xyz_array indicating the nodes composing one element
    """
    # Define coordinates of the vertices of the box based on the input geometrical properties
    vertices_xyz = np.array([(0., 0., height / 2),  # vertex 0
                             (width, 0., height / 2),  # 1
                             (width, 0., -height / 2),  # 2
                             (0., 0., -height / 2),  # 3
                             (0., span, height / 2),  # 4
                             (width, span, height / 2),  # 5
                             (width, span, -height / 2),  # 6
                             (0., span, -height / 2)])  # 7
    # Define faces by sequence of vertices
    faces = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    # Define number of nodes along the span of the box making sure that there is an odd number of nodes to better
    # approximate the buckling shape
    no_nodes_span = np.ceil(np.linalg.norm(vertices_xyz[4] - vertices_xyz[0]) / edge_length / 2).astype('int') * 2 + 1
    # Initialize array with nodes' coordinates
    nodes_xyz_array = np.empty((no_nodes_span, 0, 3))
    # Initialize array with nodes' scalar indices
    nodes_index_array = np.empty((no_nodes_span, 0), dtype=int)
    # Initialize variable scalar index of last node
    last_node_index = 0
    # Define nodes of the mesh for each face of the box
    for face in faces:
        # Define number of nodes along the width or height of the box making sure that there is an odd number of nodes
        # to better approximate the buckling shape
        no_nodes = np.ceil(np.linalg.norm(vertices_xyz[face[1]] - vertices_xyz[face[0]]) / edge_length / 2).astype(
            'int') * 2 + 1
        # Find coordinates of nodes along the root and the tip of the current face
        root_nodes_xyz = np.linspace(vertices_xyz[face[0]], vertices_xyz[face[1]], no_nodes)
        tip_nodes_xyz = np.linspace(vertices_xyz[face[3]], vertices_xyz[face[2]], no_nodes)
        # Find coordinates and scalar indices of spanwise nodes for each couple of root-tip points of the current face
        for j in range(no_nodes - 1):
            # Concatenate array of nodes' coordinates with coordinates of new nodes found
            nodes_xyz_array = np.concatenate((nodes_xyz_array, np.reshape(np.linspace(
                root_nodes_xyz[j], tip_nodes_xyz[j], no_nodes_span), (no_nodes_span, 1, 3))), axis=1)
            # Concatenate array of nodes' scalar indices with scalar indices of new nodes found
            nodes_index_array = np.concatenate((nodes_index_array, np.reshape(np.arange(
                last_node_index+j*no_nodes_span, last_node_index+(j+1)*no_nodes_span), (no_nodes_span, 1))), axis=1)
        # Update last scalar index
        last_node_index = nodes_index_array[-1, -1]+1
    # Concatenate array of nodes' coordinates and scalar indices with coordinates and scalar indices of first set of
    # spanwise nodes. This is needed for the generation of the connectivity matrix
    nodes_xyz_array = np.concatenate((nodes_xyz_array, np.reshape(nodes_xyz_array[:, 0, :], (no_nodes_span, 1, 3))),
                                     axis=1)
    nodes_index_array = np.concatenate((nodes_index_array, np.reshape(nodes_index_array[:, 0], (no_nodes_span, 1))),
                                       axis=1)
    # Initialize list with nodes connectivity information
    nodes_connectivity_list = []
    # Iterate through the rows and columns of the array of nodes' scalar indices
    for row in range(np.size(nodes_index_array, 0) - 1):
        for col in range(np.size(nodes_index_array, 1) - 1):
            # Append array with the indices of the 4 nodes composing the current element to the list
            nodes_connectivity_list.append(np.array([nodes_index_array[row, col], nodes_index_array[row + 1, col],
                                                     nodes_index_array[row + 1, col + 1], nodes_index_array[row,
                                                                                                            col + 1]]))
    # Transform connectivity list into an array
    nodes_connectivity_matrix = np.vstack(tuple(nodes_connectivity_list))
    # Transform array of node's coordinates into a 2D array neglecting last set of spanwise nodes added previously.
    # order='F is used to reshape the array stacking the sequence of spanwise nodes one below the other
    nodes_xyz_array = np.reshape(nodes_xyz_array[:, :-1, :], (-1, 3), order='F')
    # Return array with nodes' coordinates and array with the connectivity matrix
    return nodes_xyz_array, nodes_connectivity_matrix


def mesh_box_with_pyvista(width: float, span: float, height: float, edge_length: float, root_y_coordinate: float = 0.)\
        -> PolyData:
    """
    Discretizes a box with the input dimensions into quadrilateral shell elements using the pyvista package. Returns a
    PolyData object including the xyz coordinates of the nodes and their connectivity information.

    Parameters
    ----------
    width: float
        box width
    height: float
        box height
    span: float
        box span
    edge_length: float
        prescribed length of the edges of the shell elements used to discretize the geometry y0 (float): y-coordinate of
         the box root
    root_y_coordinate: float
        position of the box root

     Returns
     -------
     cleaned_mesh: PolyData
        pyvista object including the xyz coordinates of the nodes and their connectivity information
    """
    # Find number of elements along each side of the box based on the prescribed edge length. We make sure that there is
    # an even number of elements over each dimension to better approximate the buckling shape
    no_elements = [np.ceil(side/edge_length/2).astype('int')*2 for side in [width, span, height]]
    # Discretize top skin of the box
    top_skin_mesh = pyvista.Plane(center=[width/2, root_y_coordinate+span/2, height/2], direction=[0, 0, 1],
                                  i_size=width, j_size=span, i_resolution=no_elements[0], j_resolution=no_elements[1])
    # Discretize bottom skin of the box
    bottom_skin_mesh = pyvista.Plane(center=[width/2, root_y_coordinate+span/2, -height/2], direction=[0, 0, -1],
                                     i_size=width, j_size=span, i_resolution=no_elements[0],
                                     j_resolution=no_elements[1])
    # Discretize front spar of the box
    front_spar_mesh = pyvista.Plane(center=[0, root_y_coordinate+span/2, 0], direction=[-1, 0, 0], i_size=height,
                                    j_size=span, i_resolution=no_elements[2], j_resolution=no_elements[1])
    # Discretize rear spar of the box
    rear_spar_mesh = pyvista.Plane(center=[width, root_y_coordinate+span/2, 0], direction=[1, 0, 0], i_size=height,
                                   j_size=span, i_resolution=no_elements[2], j_resolution=no_elements[1])
    # Merge different components together
    merged_mesh = top_skin_mesh.merge([bottom_skin_mesh, front_spar_mesh, rear_spar_mesh])
    # Clean obtained mesh merging points closer than indicated tolerance
    cleaned_mesh = merged_mesh.clean(tolerance=edge_length/100)
    # Return cleaned mesh
    return cleaned_mesh


def mesh_rib_with_pyvista(y_coordinate: float, width: float, height: float, edge_length: float, x_0: float = 0.) -> PolyData:
    """
    Discretizes a rib with the input dimensions into quadrilateral shell elements using the pyvista package. Returns a
    PolyData object including the xyz coordinates of the nodes and their connectivity information.

    Parameters
    ----------
    y_coordinate: float
        y-coordinate of the rib
    width: float
        rib width
    height: float
        rib height
    edge_length: float
        prescribed length of the edges of the shell elements used to discretize the geometry
    x_0: float
        x-coordinate of front rib edge

    Returns
    -------
    rib_mesh: PolyData
        pyvista object including the xyz coordinates of the nodes and their connectivity information
    """
    # Find number of elements along each side of the rib
    no_elements = [np.ceil(side/edge_length/2).astype('int')*2 for side in [width, height]]
    # Discretize rib
    rib_mesh = pyvista.Plane(center=[x_0 + width/2, y_coordinate, 0], direction=[0, 1, 0], i_size=width,
                             j_size=height, i_resolution=no_elements[0], j_resolution=no_elements[1])
    # Return discretized geometry
    return rib_mesh


def mesh_box_beam_with_pyvista(ribs_y_coordinates: ndarray, width: float, height: float, edge_length: float) -> PolyData:
    """
    Discretizes a box beam reinforced with ribs into quadrilateral shell elements using the pyvista package. Returns a
    PolyData object including the xyz coordinates of the nodes and their connectivity information.

    Parameters
    ----------
    ribs_y_coordinates: ndarray
        y-coordinates of the ribs
    width: float
        box beam width
    height: float
        box beam height
    edge_length: float
        prescribed length of the edges of the shell elements used to discretize the geometry

    Returns
    -------
    cleaned_box_beam_mesh: PolyData
        pyvista object including the xyz coordinates of the nodes and their connectivity information
    """
    # Initialize lists of the PolyData objects corresponding to the box segments and to the ribs
    box_meshes = []
    rib_meshes = []
    # Iterate through the y-coordinates of the rib, except last one
    for count, y in enumerate(ribs_y_coordinates[:-1]):
        # Discretize box segment between current and next rib and add PolyData object to the list
        box_meshes.append(mesh_box_with_pyvista(width, ribs_y_coordinates[count+1]-y, height, edge_length, y))
        # Discretize current rib and add PolyData object to the list
        rib_meshes.append(mesh_rib_with_pyvista(y, width, height, edge_length))
    # Discretize last rib and add PolyData object to the list
    rib_meshes.append(mesh_rib_with_pyvista(ribs_y_coordinates[-1], width, height, edge_length))
    # Merge all box segments and ribs together
    merged_box_beam_mesh = box_meshes[0].merge(box_meshes[1:] + rib_meshes)
    # Clean obtained mesh merging points closer than indicated tolerance
    cleaned_box_beam_mesh = merged_box_beam_mesh.clean(tolerance=edge_length/100)
    # Return cleaned mesh
    return cleaned_box_beam_mesh


def mesh_stiffened_box_with_pyvista(width: float, span: float, height: float, stiffeners_x_coordinates: ndarray, stiffeners_height: float,
 edge_length: float, root_y_coordinate: float = 0.) -> PolyData:
    """
    Discretizes a box reinforced with stiffeners with the input dimensions into quadrilateral shell elements using the pyvista package.
    Returns a PolyData object including the xyz coordinates of the nodes and their connectivity information.

    Parameters
    ----------
    width: float
        box width
    height: float
        box height
    span: float
        box span
    stiffeners_x_coordinates: ndarray
        array with the x-coordinates of the stiffeners
    stiffeners_height: float
        height of the stiffeners
    edge_length: float
        prescribed length of the edges of the shell elements used to discretize the geometry y0 (float): y-coordinate of
         the box root
    root_y_coordinate: float
        position of the box root

     Returns
     -------
     cleaned_mesh: PolyData
        pyvista object including the xyz coordinates of the nodes and their connectivity information
    """
    # Find number of elements along each side of the box based on the prescribed edge length. We make sure that there is an even number of elements over each dimension to better approximate the buckling shape
    no_elements = [np.ceil(side/edge_length/2).astype('int')*2 for side in [height, span]]
    # Discretize front spar of the stiffened box
    front_spar_mesh = pyvista.Plane(center=[0, root_y_coordinate + span/2, 0], direction=[-1, 0, 0], i_size=height,
                                    j_size=span, i_resolution=no_elements[0], j_resolution=no_elements[1])
    # Discretize rear spar of the box
    rear_spar_mesh = pyvista.Plane(center=[width, root_y_coordinate + span/2, 0], direction=[1, 0, 0], i_size=height,
                                   j_size=span, i_resolution=no_elements[0], j_resolution=no_elements[1])
    # Initialize lists of the PolyData objects corresponding to the box segments and to the ribs
    top_skin_meshes = []
    top_stiffeners_meshes = []
    bottom_skin_meshes = []
    bottom_stiffeners_meshes = []
    no_stiffener_elements = np.ceil(stiffeners_height/edge_length/2).astype('int')*2
    x_0 = 0
    # Iterate through the x-coordinates of the stiffeners, except last one
    for count, x in enumerate(stiffeners_x_coordinates):
        # Find number of elements along the width
        no_width_elements = np.ceil((x - x_0)/edge_length/2).astype('int')*2
        # Discretize top skin segment
        top_skin_meshes.append(pyvista.Plane(center=[(x_0 + x)/2, root_y_coordinate + span/2, height/2], direction=[0, 0, 1], i_size=x - x_0,
         j_size=span, i_resolution=no_width_elements, j_resolution=no_elements[1]))
        # Discretize top stiffener
        top_stiffeners_meshes.append(pyvista.Plane(center=[x, root_y_coordinate + span/2, height/2 - stiffeners_height/2], direction=[1, 0, 0],
         i_size=stiffeners_height, j_size=span, i_resolution=no_stiffener_elements, j_resolution=no_elements[1]))
        # Discretize bottom skin segment
        bottom_skin_meshes.append(pyvista.Plane(center=[(x_0 + x)/2, root_y_coordinate + span/2, -height/2], direction=[0, 0, -1],
         i_size=x - x_0, j_size=span, i_resolution=no_width_elements, j_resolution=no_elements[1]))
        # Discretize bottom stiffener
        bottom_stiffeners_meshes.append(pyvista.Plane(center=[x, root_y_coordinate + span/2, -height/2 + stiffeners_height/2],
         direction=[1, 0, 0], i_size=stiffeners_height, j_size=span, i_resolution=no_stiffener_elements, j_resolution=no_elements[1]))
        # Update x_0
        x_0 = x
    # Discretize last rib-stiffener bay of the skins
    no_width_elements = np.ceil((width - x_0)/edge_length/2).astype('int')*2
    top_skin_meshes.append(pyvista.Plane(center=[(x_0 + width)/2, root_y_coordinate + span/2, height/2], direction=[0, 0, 1], i_size=width - x_0,
     j_size=span, i_resolution=no_width_elements, j_resolution=no_elements[1]))
    bottom_skin_meshes.append(pyvista.Plane(center=[(x_0 + width)/2, root_y_coordinate + span/2, -height/2], direction=[0, 0, -1], i_size=width - x_0,
     j_size=span, i_resolution=no_width_elements, j_resolution=no_elements[1]))
    # Merge all box segments and ribs together
    merged_mesh = front_spar_mesh.merge([rear_spar_mesh] + top_skin_meshes + top_stiffeners_meshes + bottom_skin_meshes +
     bottom_stiffeners_meshes)
    # Clean obtained mesh merging points closer than indicated tolerance
    cleaned_mesh = merged_mesh.clean(tolerance=edge_length/100)
    # Return cleaned mesh
    return cleaned_mesh


def mesh_stiffened_box_beam_with_pyvista(width: float, height: float, ribs_y_coordinates: ndarray, stiffeners_x_coordinates: ndarray,
    stiffeners_height: float, edge_length: float) -> PolyData:
    """
    Discretizes a box beam reinforced with ribs and stiffeners into quadrilateral shell elements using the pyvista package. Returns a
    PolyData object including the xyz coordinates of the nodes and their connectivity information.

    Parameters
    ----------
    width: float
        box beam width
    height: float
        box beam height
    ribs_y_coordinates: ndarray
        array of the y-coordinates of the ribs
    stiffeners_x_coordinates: ndarray
        array of the x-coordinates of the stiffeners
    stiffeners_height: float
        height of the stiffeners
    edge_length: float
        prescribed length of the edges of the shell elements used to discretize the geometry

    Returns
    -------
    cleaned_box_beam_mesh: PolyData
        pyvista object including the xyz coordinates of the nodes and their connectivity information
    """
    # Initialize lists of the PolyData objects corresponding to the box segments and to the ribs
    stiffened_box_meshes = []
    rib_meshes = []
    rib_segments_x_coordinates = np.concatenate(([0.], stiffeners_x_coordinates, [width]))  # create array of the x-coordiantes defining the rib segments
    rib_segments_widths = np.ediff1d(rib_segments_x_coordinates)  # calculate the width of each rib segment
    # Iterate through the y-coordinates of the rib, except last one
    for count, y in enumerate(ribs_y_coordinates[:-1]):
        # Discretize stiffened box segment between current and next rib and add PolyData object to the list
        stiffened_box_meshes.append(mesh_stiffened_box_with_pyvista(width, ribs_y_coordinates[count+1] - y, height,
         stiffeners_x_coordinates, stiffeners_height, edge_length, y))
        # Discretize current rib and add PolyData object to the list
        rib_meshes = rib_meshes + [mesh_rib_with_pyvista(y, rib_segments_widths[i], height, edge_length,
         rib_segments_x_coordinates[i]) for i in range(len(rib_segments_widths))]
    # Discretize last rib and add PolyData object to the list
    rib_meshes = rib_meshes + [mesh_rib_with_pyvista(ribs_y_coordinates[-1], rib_segments_widths[i], height, edge_length,
     rib_segments_x_coordinates[i]) for i in range(len(rib_segments_widths))]
    # Merge all stiffene box segments and ribs together
    merged_box_beam_mesh = stiffened_box_meshes[0].merge(stiffened_box_meshes[1:] + rib_meshes)
    # Clean obtained mesh merging points closer than indicated tolerance
    cleaned_box_beam_mesh = merged_box_beam_mesh.clean(tolerance=edge_length/100)
    # Return cleaned mesh
    return cleaned_box_beam_mesh


def create_base_bdf_input(young_modulus: float, poisson_ratio: float, density: float, shell_thickness: float,
                          nodes_xyz_array: ndarray, nodes_connectivity_matrix: ndarray) -> BDF:
    """
    Returns a BDF object with material properties, nodes, elements, boundary conditions and output files defaults for
    the box beam model.

    Parameters
    ----------
    young_modulus: float
        Young's modulus of the column
    poisson_ratio: float
        Poisson's ratio of the column
    density: float
        density of the column
    shell_thickness: float
        thickness of the shell elements composing the box beam
    nodes_xyz_array: ndarray
        nx3 array with the xyz coordinates of the nodes
    nodes_connectivity_matrix: ndarray
        nx4 array with the connectivity information of the nodes, where each row indicates the indices of the
         nodes_xyz_array corresponding to the nodes composing the element

    Returns
    -------
    bdf_input: BDF
        pyNastran object representing the bdf input of the box beam model
    """
    # Create an instance of the BDF class without debug or info messages
    bdf_input = BDF(debug=None)
    # Add MAT1 card (isotropic material)
    material_id = 1
    bdf_input.add_mat1(mid=material_id, E=young_modulus, G='', nu=poisson_ratio, rho=density)
    # Add PSHELL card (properties of thin shell elements)
    property_id = 1
    bdf_input.add_pshell(pid=property_id, mid1=material_id, t=shell_thickness, mid2=material_id, mid3=material_id)
    # Add GRID cards (nodes) based on input coordinates
    nodes_id_array = np.arange(1, np.size(nodes_xyz_array, 0) + 1)
    for count, node_xyz in enumerate(nodes_xyz_array):
        bdf_input.add_grid(nid=nodes_id_array[count], xyz=node_xyz)
    # Add CQUAD4 cards (shell elements) based on input connectivity matrix
    for count, nodes_indices in enumerate(nodes_connectivity_matrix):
        bdf_input.add_cquad4(eid=count+1, pid=property_id,
                             nids=[nodes_id_array[nodes_indices[0]], nodes_id_array[nodes_indices[1]],
                                   nodes_id_array[nodes_indices[2]], nodes_id_array[nodes_indices[3]]])
    # Add SPC1 card (single-point constraint) defining fixed boundary conditions at the root nodes
    root_nodes_ids = nodes_id_array[np.abs(nodes_xyz_array[:, 1]) < shell_thickness / 100]
    constraint_set_id = 1
    bdf_input.add_spc1(constraint_set_id, '123456', root_nodes_ids)
    # Add SCP1 card to case control deck
    bdf_input.create_subcases(0)
    bdf_input.case_control_deck.subcases[0].add_integer_type('SPC', constraint_set_id)
    # Set defaults for output files
    bdf_input.add_param('POST', [1])  # add PARAM card to store results in a op2 file
    bdf_input.case_control_deck.subcases[0].add('ECHO', 'NONE', [], 'STRING-type')  # request no Bulk Data to be printed
    bdf_input.case_control_deck.subcases[0].add_result_type('DISPLACEMENT', 'ALL', ['PLOT'])  # store displacement data of all nodes in the op2 file
    bdf_input.case_control_deck.subcases[0].add_result_type('OLOAD', 'ALL', ['PLOT'])  # store form and type of applied load vector
    # Cross-reference BDF object
    bdf_input._xref = True
    bdf_input.cross_reference()
    # Return BDF object
    return bdf_input
