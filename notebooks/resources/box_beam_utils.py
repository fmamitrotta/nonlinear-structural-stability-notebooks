"""
This file is part of the GitHub repository nonlinear-structural-stability-notebooks, created by Francesco M. A. Mitrotta.
Copyright (C) 2022 Francesco Mario Antonio Mitrotta

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from pyNastran.bdf.bdf import BDF
from numpy import ndarray
import numpy as np
import pyvista as pv
from pyvista import PolyData
from typing import Tuple


def mesh_box(height: float, length: float, width: float, element_length: float) -> tuple[ndarray, ndarray]:
    """
    Discretizes a box with the input dimensions into quadrilateral shell elements returning an array with the xyz
    coordinates of the nodes and an array with their connectivity information.

    Parameters
    ----------
    height: float
        box height
    length: float
        box length
    width: float
        box width
    element_length: float
        target length of the shell elements used to discretize the geometry

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
                             (0., length, height / 2),  # 4
                             (width, length, height / 2),  # 5
                             (width, length, -height / 2),  # 6
                             (0., length, -height / 2)])  # 7
    # Define faces by sequence of vertices
    faces = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    # Define number of nodes along the length of the box making sure that there is an odd number of nodes to better
    # approximate the buckling shape
    no_nodes_length = np.ceil(np.linalg.norm(vertices_xyz[4] - vertices_xyz[0]) / element_length / 2).astype('int') * 2 + 1
    # Initialize array with nodes' coordinates
    nodes_xyz_array = np.empty((no_nodes_length, 0, 3))
    # Initialize array with nodes' scalar indices
    nodes_index_array = np.empty((no_nodes_length, 0), dtype=int)
    # Initialize variable scalar index of last node
    last_node_index = 0
    # Define nodes of the mesh for each face of the box
    for face in faces:
        # Define number of nodes along the width or height of the box making sure that there is an odd number of nodes
        # to better approximate the buckling shape
        no_nodes = np.ceil(np.linalg.norm(vertices_xyz[face[1]] - vertices_xyz[face[0]]) / element_length / 2).astype(
            'int') * 2 + 1
        # Find coordinates of nodes along the root and the tip of the current face
        root_nodes_xyz = np.linspace(vertices_xyz[face[0]], vertices_xyz[face[1]], no_nodes)
        tip_nodes_xyz = np.linspace(vertices_xyz[face[3]], vertices_xyz[face[2]], no_nodes)
        # Find coordinates and scalar indices of nodes along the length for each couple of root-tip points of the current face
        for j in range(no_nodes - 1):
            # Concatenate array of nodes' coordinates with coordinates of new nodes found
            nodes_xyz_array = np.concatenate((nodes_xyz_array, np.reshape(np.linspace(
                root_nodes_xyz[j], tip_nodes_xyz[j], no_nodes_length), (no_nodes_length, 1, 3))), axis=1)
            # Concatenate array of nodes' scalar indices with scalar indices of new nodes found
            nodes_index_array = np.concatenate((nodes_index_array, np.reshape(np.arange(
                last_node_index+j*no_nodes_length, last_node_index+(j+1)*no_nodes_length), (no_nodes_length, 1))), axis=1)
        # Update last scalar index
        last_node_index = nodes_index_array[-1, -1]+1
    # Concatenate the arrays of the nodes' coordinates and the nodes's scalar indices with the coordinates and scalar indices of the first
    # set of nodes. This is needed for the generation of the connectivity matrix
    nodes_xyz_array = np.concatenate((nodes_xyz_array, np.reshape(nodes_xyz_array[:, 0, :], (no_nodes_length, 1, 3))), axis=1)
    nodes_index_array = np.concatenate((nodes_index_array, np.reshape(nodes_index_array[:, 0], (no_nodes_length, 1))), axis=1)
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
    # Transform array of node's coordinates into a 2D array neglecting last set of nodes added previously.
    # order='F is used to reshape the array stacking the sequence of nodes along the length one below the other
    nodes_xyz_array = np.reshape(nodes_xyz_array[:, :-1, :], (-1, 3), order='F')
    # Return array with nodes' coordinates and array with the connectivity matrix
    return nodes_xyz_array, nodes_connectivity_matrix


def mesh_box_with_pyvista(height: float, length: float, width: float, element_length: float, y_0: float = 0.) -> PolyData:
    """
    Discretizes a box with the input dimensions into quadrilateral shell elements using the pyvista package. Returns a
    PolyData object including the xyz coordinates of the nodes and their connectivity information.

    Parameters
    ----------
    height: float
        box height
    length: float
        box length
    width: float
        box width
    element_length: float
        target length of the shell elements used to discretize the geometry
    y_0: float
        y-coordinate of the root of the box

     Returns
     -------
     cleaned_mesh: PolyData
        pyvista object including the xyz coordinates of the nodes and their connectivity information
    """
    # Find number of elements along each side of the box based on the prescribed edge length. We make sure that there is
    # an even number of elements over each dimension to better approximate the buckling shape
    no_elements = [np.ceil(side/element_length/2).astype('int')*2 for side in [width, length, height]]
    # Discretize top skin of the box
    top_skin_mesh = pv.Plane(center=[width/2, y_0 + length/2, height/2], direction=[0, 0, 1],
                                  i_size=width, j_size=length, i_resolution=no_elements[0], j_resolution=no_elements[1])
    # Discretize bottom skin of the box
    bottom_skin_mesh = pv.Plane(center=[width/2, y_0 + length/2, -height/2], direction=[0, 0, -1],
                                     i_size=width, j_size=length, i_resolution=no_elements[0],
                                     j_resolution=no_elements[1])
    # Discretize front spar of the box
    front_spar_mesh = pv.Plane(center=[0, y_0 + length/2, 0], direction=[-1, 0, 0], i_size=height,
                                    j_size=length, i_resolution=no_elements[2], j_resolution=no_elements[1])
    # Discretize rear spar of the box
    rear_spar_mesh = pv.Plane(center=[width, y_0 + length/2, 0], direction=[1, 0, 0], i_size=height,
                                   j_size=length, i_resolution=no_elements[2], j_resolution=no_elements[1])
    # Merge different components together
    merged_mesh = top_skin_mesh.merge([bottom_skin_mesh, front_spar_mesh, rear_spar_mesh])
    # Clean obtained mesh merging points closer than indicated tolerance
    cleaned_mesh = merged_mesh.clean(tolerance=element_length/100)
    # Return cleaned mesh
    return cleaned_mesh


def mesh_rib_with_pyvista(height: float, width: float, y_coordinate: float, element_length: float, x_0: float = 0.) -> PolyData:
    """
    Discretizes a rib with the input dimensions into quadrilateral shell elements using the pyvista package. Returns a
    PolyData object including the xyz coordinates of the nodes and their connectivity information.

    Parameters
    ----------
    height: float
        rib height
    width: float
        rib width
    y_coordinate: float
        y-coordinate of the rib
    element_length: float
        target length of the shell elements used to discretize the geometry
    x_0: float
        x-coordinate of front rib edge

    Returns
    -------
    rib_mesh: PolyData
        pyvista object including the xyz coordinates of the nodes and their connectivity information
    """
    # Find number of elements along each side of the rib
    no_elements = [np.ceil(side/element_length/2).astype('int')*2 for side in [height, width]]
    # Discretize rib
    rib_mesh = pv.Plane(center=[x_0 + width/2, y_coordinate, 0], direction=[0, 1, 0], i_size=height,
                             j_size=width, i_resolution=no_elements[0], j_resolution=no_elements[1])
    # Return discretized geometry
    return rib_mesh


def mesh_box_beam_with_pyvista(height: float, ribs_y_coordinates: ndarray, width: float, element_length: float) -> PolyData:
    """
    Discretizes a box beam reinforced with ribs into quadrilateral shell elements using the pyvista package. Returns a
    PolyData object including the xyz coordinates of the nodes and their connectivity information.

    Parameters
    ----------
    height: float
        box beam height
    ribs_y_coordinates: ndarray
        y-coordinates of the ribs
    width: float
        box beam width
    element_length: float
        target length of the shell elements used to discretize the geometry

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
        box_meshes.append(mesh_box_with_pyvista(length=ribs_y_coordinates[count+1]-y, height=height, width=width,
                                                element_length=element_length, y_0=y))
        # Discretize current rib and add PolyData object to the list
        rib_meshes.append(mesh_rib_with_pyvista(y_coordinate=y, width=width, height=height, element_length=element_length))
    # Discretize last rib and add PolyData object to the list
    rib_meshes.append(mesh_rib_with_pyvista(y_coordinate=ribs_y_coordinates[-1], width=width, height=height,
                                            element_length=element_length))
    # Merge all box segments and ribs together
    merged_box_beam_mesh = box_meshes[0].merge(box_meshes[1:] + rib_meshes)
    # Clean obtained mesh merging points closer than indicated tolerance
    cleaned_box_beam_mesh = merged_box_beam_mesh.clean(tolerance=element_length/100)
    # Return cleaned mesh
    return cleaned_box_beam_mesh


def mesh_stiffened_box_with_pyvista(height: float, length: float, width: float, stiffeners_x_coordinates: ndarray,
                                    stiffeners_height: float, element_length: float, y_0: float = 0.) -> PolyData:
    """
    Discretizes a box reinforced with stiffeners with the input dimensions into quadrilateral shell elements using the pyvista package.
    Returns a PolyData object including the xyz coordinates of the nodes and their connectivity information.

    Parameters
    ----------
    height: float
        box height
    length: float
        box length
    width: float
        box width
    stiffeners_x_coordinates: ndarray
        array with the x-coordinates of the stiffeners
    stiffeners_height: float
        height of the stiffeners
    element_length: float
        target length of the shell elements used to discretize the geometry
    y_0: float
        position of the box root

     Returns
     -------
     cleaned_mesh: PolyData
        pyvista object including the xyz coordinates of the nodes and their connectivity information
    """
    # Find number of elements along each side of the box based on the prescribed edge length. We make sure that there is an even number of elements over each dimension to better approximate the buckling shape
    no_elements = [np.ceil(side/element_length/2).astype('int')*2 for side in [height, length]]
    # Discretize front spar of the stiffened box
    front_spar_mesh = pv.Plane(center=[0, y_0 + length/2, 0], direction=[-1, 0, 0], i_size=height,
                                    j_size=length, i_resolution=no_elements[0], j_resolution=no_elements[1])
    # Discretize rear spar of the box
    rear_spar_mesh = pv.Plane(center=[width, y_0 + length/2, 0], direction=[1, 0, 0], i_size=height,
                                   j_size=length, i_resolution=no_elements[0], j_resolution=no_elements[1])
    # Initialize lists of the PolyData objects corresponding to the box segments and to the ribs
    top_skin_meshes = []
    top_stiffeners_meshes = []
    bottom_skin_meshes = []
    bottom_stiffeners_meshes = []
    no_stiffener_elements = np.ceil(stiffeners_height/element_length/2).astype('int')*2
    x_0 = 0
    # Iterate through the x-coordinates of the stiffeners, except last one
    for x in stiffeners_x_coordinates:
        # Find number of elements along the width
        no_width_elements = np.ceil((x - x_0)/element_length/2).astype('int')*2
        # Discretize top skin segment
        top_skin_meshes.append(pv.Plane(center=[(x_0 + x)/2, y_0 + length/2, height/2], direction=[0, 0, 1],
                                             i_size=x - x_0, j_size=length, i_resolution=no_width_elements, j_resolution=no_elements[1]))
        # Discretize top stiffener
        top_stiffeners_meshes.append(pv.Plane(center=[x, y_0 + length/2, height/2 - stiffeners_height/2],
                                                   direction=[1, 0, 0], i_size=stiffeners_height, j_size=length,
                                                   i_resolution=no_stiffener_elements, j_resolution=no_elements[1]))
        # Discretize bottom skin segment
        bottom_skin_meshes.append(pv.Plane(center=[(x_0 + x)/2, y_0 + length/2, -height/2], direction=[0, 0, -1],
         i_size=x - x_0, j_size=length, i_resolution=no_width_elements, j_resolution=no_elements[1]))
        # Discretize bottom stiffener
        bottom_stiffeners_meshes.append(pv.Plane(center=[x, y_0 + length/2, -height/2 + stiffeners_height/2],
         direction=[1, 0, 0], i_size=stiffeners_height, j_size=length, i_resolution=no_stiffener_elements, j_resolution=no_elements[1]))
        # Update x_0
        x_0 = x
    # Discretize last rib-stiffener bay of the skins
    no_width_elements = np.ceil((width - x_0)/element_length/2).astype('int')*2
    top_skin_meshes.append(pv.Plane(center=[(x_0 + width)/2, y_0 + length/2, height/2], direction=[0, 0, 1], i_size=width - x_0,
                                         j_size=length, i_resolution=no_width_elements, j_resolution=no_elements[1]))
    bottom_skin_meshes.append(pv.Plane(center=[(x_0 + width)/2, y_0 + length/2, -height/2], direction=[0, 0, -1],
                                            i_size=width - x_0, j_size=length, i_resolution=no_width_elements, j_resolution=no_elements[1]))
    # Merge all box segments and ribs together
    merged_mesh = front_spar_mesh.merge([rear_spar_mesh] + top_skin_meshes + top_stiffeners_meshes + bottom_skin_meshes +
     bottom_stiffeners_meshes)
    # Clean obtained mesh merging points closer than indicated tolerance
    cleaned_mesh = merged_mesh.clean(tolerance=element_length/100)
    # Return cleaned mesh
    return cleaned_mesh


def mesh_stiffened_box_beam_with_pyvista(height: float, width: float, ribs_y_coordinates: ndarray, stiffeners_x_coordinates: ndarray,
    stiffeners_height: float, element_length: float) -> PolyData:
    """
    Discretizes a box beam reinforced with ribs and stiffeners into quadrilateral shell elements using the pyvista package. Returns a
    PolyData object including the xyz coordinates of the nodes and their connectivity information.

    Parameters
    ----------
    height: float
        box beam height
    width: float
        box beam width
    ribs_y_coordinates: ndarray
        array of the y-coordinates of the ribs
    stiffeners_x_coordinates: ndarray
        array of the x-coordinates of the stiffeners
    stiffeners_height: float
        height of the stiffeners
    element_length: float
        target length of the shell elements used to discretize the geometry

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
        stiffened_box_meshes.append(mesh_stiffened_box_with_pyvista(width=width, length=ribs_y_coordinates[count+1] - y, height=height,
                                                                    stiffeners_x_coordinates=stiffeners_x_coordinates,
                                                                    stiffeners_height=stiffeners_height, y_0=y,
                                                                    element_length=element_length))
        # Discretize current rib and add PolyData object to the list
        rib_meshes = rib_meshes + [mesh_rib_with_pyvista(y_coordinate=y, width=rib_segments_widths[i], height=height,
                                                         element_length=element_length, x_0=rib_segments_x_coordinates[i])
                                                         for i in range(len(rib_segments_widths))]
    # Discretize last rib and add PolyData object to the list
    rib_meshes = rib_meshes + [mesh_rib_with_pyvista(y_coordinate=ribs_y_coordinates[-1], width=rib_segments_widths[i], height=height,
                                                     element_length=element_length, x_0=rib_segments_x_coordinates[i])
                                                     for i in range(len(rib_segments_widths))]
    # Merge all stiffene box segments and ribs together
    merged_box_beam_mesh = stiffened_box_meshes[0].merge(stiffened_box_meshes[1:] + rib_meshes)
    # Clean obtained mesh merging points closer than indicated tolerance
    cleaned_box_beam_mesh = merged_box_beam_mesh.clean(tolerance=element_length/100)
    # Return cleaned mesh
    return cleaned_box_beam_mesh


def create_base_bdf_input(young_modulus: float, poisson_ratio: float, density: float, shell_thickness: float,
                          nodes_xyz_array: ndarray, nodes_connectivity_matrix: ndarray, parallel:bool = False,
                          no_cores:int = 8) -> BDF:
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
    parallel: bool
        flag to enable or disable the parallel execution of Nastran
    no_cores: int
        number of cores used for the parallel execution of Nastran

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
        bdf_input.add_cquad4(eid=count + 1, pid=property_id,
                             nids=[nodes_id_array[nodes_indices[0]], nodes_id_array[nodes_indices[1]],
                                   nodes_id_array[nodes_indices[2]], nodes_id_array[nodes_indices[3]]])
    # Add SPC1 card (single-point constraint) defining fixed boundary conditions at the root nodes
    tolerance = np.linalg.norm(bdf_input.nodes[2].xyz - bdf_input.nodes[1].xyz)/100
    root_nodes_ids = nodes_id_array[np.abs(nodes_xyz_array[:, 1]) < tolerance]
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
    # Set parallel execution of Nastran if requested
    if parallel:
        bdf_input.system_command_lines[0:0] = [f"NASTRAN PARALLEL={no_cores:d}"]
    # Return BDF object
    return bdf_input


def define_property_patches(bdf_input: BDF, ribs_y_locations: ndarray, stiffeners_x_locations: ndarray):
    """
    Defines the property cards of the design patches of the box beam model based on the input locations of the ribs.

    Parameters
    ----------
    bdf_input: BDF
        pyNastran object representing the bdf input of the box beam model
    ribs_y_locations: ndarray
        y-coordinates of the ribs
    stiffeners_x_locations: ndarray
        x-coordinates of the stiffeners
    """
    # Find element ids and centroid coordinates
    element_ids = np.array(list(bdf_input.element_ids))
    centroids_xyz = np.empty((len(bdf_input.elements), 3))
    for count, (eid, elem) in enumerate(bdf_input.elements.items()):
        centroids_xyz[count] = elem.Centroid()
    # Define tolerance to find elements inside each patch based on the distance between the first two nodes of the first element
    tolerance = np.linalg.norm(bdf_input.elements[1].nodes_ref[1].xyz - bdf_input.elements[1].nodes_ref[0].xyz)/100
    # Find default thickness
    rib_pid = 1  # initialize PSHELL id
    t = bdf_input.properties[rib_pid].t  # find default thickness
    # Add PSHELL card for each optimization patch and group element ids of external and internal structure
    material_id = 1  # default material id
    internal_elements_ids = element_ids[np.where(np.abs(centroids_xyz[:, 1] - ribs_y_locations[0]) < tolerance)[0]]  # initialize both internal and external elements ids with the first rib
    external_elements_ids = element_ids[np.where(np.abs(centroids_xyz[:, 1] - ribs_y_locations[0]) < tolerance)[0]]
    for i in range(1, len(ribs_y_locations)):  # iterate over the ribs except the first
        # Stiffened box path
        stiffened_box_pid = rib_pid + 1  # increment PSHELL id
        bdf_input.add_pshell(pid=stiffened_box_pid, mid1=material_id, t=t, mid2=material_id, mid3=material_id)  # add PSHELL card
        bdf_input.properties[stiffened_box_pid].cross_reference(bdf_input)  # cross-reference PSHELL card
        stiffened_box_element_indices = np.where((centroids_xyz[:, 1] > ribs_y_locations[i - 1]) &
                                                (centroids_xyz[:, 1] < ribs_y_locations[i]))[0]  # find indices of the elements belonging to current stiffened box patch
        stiffened_box_element_ids = element_ids[stiffened_box_element_indices]  # find corresponding element ids
        stiffened_box_element_centroids_xyz = centroids_xyz[stiffened_box_element_indices]  # find corresponding element centroids
        stiffeners_boolean = np.any(np.isclose(stiffened_box_element_centroids_xyz[:, 0], stiffeners_x_locations[:, None],
                                               atol=tolerance), axis=0)  # find boolean array of which elements belong to stiffeners
        internal_elements_ids = np.concatenate((internal_elements_ids, stiffened_box_element_ids[stiffeners_boolean]))  # add element ids of stiffeners to internal elements ids
        external_elements_ids = np.concatenate((external_elements_ids, stiffened_box_element_ids[~stiffeners_boolean]))  # add remaining element ids to external elements ids
        for eid in stiffened_box_element_ids:  # iterate over element ids of current stiffened box patch
            elem = bdf_input.elements[eid]  # get element object
            elem.uncross_reference()  # uncross-reference element object
            elem.pid = stiffened_box_pid  # update PSHELL id
            elem.cross_reference(bdf_input)  # recross-reference element object
        # Rib patch
        rib_pid = stiffened_box_pid + 1  # increment PSHELL id
        bdf_input.add_pshell(pid=rib_pid, mid1=material_id, t=t, mid2=material_id, mid3=material_id)  # add PSHELL card
        bdf_input.properties[rib_pid].cross_reference(bdf_input)  # cross-reference PSHELL card
        rib_element_ids = element_ids[np.where(np.abs(centroids_xyz[:, 1] - ribs_y_locations[i]) < tolerance)[0]]  # find element ids of current rib
        internal_elements_ids = np.concatenate((internal_elements_ids, rib_element_ids))  # add element ids of rib to internal elements ids
        for eid in rib_element_ids:  # iterate over element ids of current rib
            elem = bdf_input.elements[eid]  # get element object
            elem.uncross_reference()  # uncross-reference element object
            elem.pid = rib_pid  # update PSHELL id
            elem.cross_reference(bdf_input)  # recross-reference element object  # find which elements belong to stiffeners
    external_elements_ids = np.concatenate((external_elements_ids, rib_element_ids))  # add element ids of last rib to external elements ids


def discretize_length(length: float, target_element_length: float) -> int:
    """
    Calculates the required number of nodes to evenly discretize a specified length such that each element has a length
    equal to or less than a target length. The calculation ensures an even number of elements by rounding up if necessary.

    Parameters
    ----------
    length : float
        The total length to be discretized.
    target_element_length : float
        The maximum allowable length of each discretized element.
    
    Returns
    -------
    int
        The number of nodes required to discretize the length.
    """
    # Calculate the minimum number of elements needed and ensure it's even, then calculate nodes
    return int(np.ceil(length / target_element_length / 2)) * 2 + 1


def mesh_between_profiles(start_profile_xyz_array: ndarray, end_profile_xyz_array: ndarray, no_nodes: int, tag: str) -> PolyData:
    """
    Generates a mesh between two profiles using linear interpolation to create quadrilateral shell elements.
    Ensures compatibility with the PyVista package by arranging the mesh data into a format it can process.
    
    Parameters
    ----------
    start_profile_xyz_array : ndarray
        An Nx3 array representing XYZ coordinates of the start profile's nodes.
    end_profile_xyz_array : ndarray
        An Nx3 array representing XYZ coordinates of the end profile's nodes.
    no_nodes : int
        The total number of nodes to be placed between the start and end profiles, inclusive of the profiles themselves.
    tag : str
        A tag to identify the mesh.
    
    Returns
    -------
    mesh_polydata : PolyData
        A PyVista PolyData object containing the mesh's points and connectivity.
    
    Raises
    ------
    ValueError
        If the start and end profiles have differing numbers of nodes.
    """
    # Ensure the profiles have the same number of nodes
    if start_profile_xyz_array.shape != end_profile_xyz_array.shape:
        raise ValueError("Start and end profiles must have the same number of nodes.")
    # Generate interpolated points between the profiles
    t = np.linspace(0, 1, no_nodes)[np.newaxis, :, np.newaxis]
    interpolated_nodes = start_profile_xyz_array[:, np.newaxis, :] * (1 - t) + end_profile_xyz_array[:, np.newaxis, :] * t
    # Reshape to a single array of points for the mesh
    mesh_xyz_array = interpolated_nodes.reshape(-1, 3)
    # Create quadrilateral connectivity between points
    no_profile_nodes = start_profile_xyz_array.shape[0]
    indices = np.arange(no_profile_nodes * no_nodes).reshape(no_profile_nodes, no_nodes)
    quads = np.hstack([indices[:-1, :-1].reshape(-1, 1), indices[1:, :-1].reshape(-1, 1),
                       indices[1:, 1:].reshape(-1, 1), indices[:-1, 1:].reshape(-1, 1)])
    # Format for PyVista: prepend each quad with a 4 to indicate it's a quadrilateral
    faces = np.insert(quads, 0, 4, axis=1).flatten()
    # Create and return the PyVista PolyData object
    mesh_polydata = pv.PolyData()
    mesh_polydata.points = mesh_xyz_array
    mesh_polydata.faces = faces
    mesh_polydata.cell_data['tag'] = [tag]*mesh_polydata.n_cells  # add tag to cell data
    return mesh_polydata


def mesh_along_y_axis(x_coordinates: ndarray, y_coordinates_start: ndarray, y_coordinates_end: ndarray, z_coordinates: ndarray,
                      no_y_nodes: int, tag: str) -> PolyData:
    """
    Helper function to create a mesh along the Y-axis between two points.
    
    Parameters are arrays of X, starting Y, ending Y, and Z coordinates, along with the number of nodes to interpolate
    along the Y direction. Calls mesh_between_profiles to generate the mesh.
    
    Parameters
    ----------
    x_coordinates : ndarray
        X coordinates for meshing.
    y_coordinates_start : ndarray
        Starting Y coordinates for meshing.
    y_coordinates_end : ndarray
        Ending Y coordinates for meshing.
    z_coordinates : ndarray
        Z coordinates for meshing.
    no_y_nodes : int
        Number of nodes to use along the Y-axis.
    tag : str
        A tag to identify the mesh.
    
    Returns
    -------
    PolyData
        The mesh generated along the Y-axis as a PyVista PolyData object.
    """
    start_profile_xyz_array = np.column_stack((x_coordinates, y_coordinates_start, z_coordinates))
    end_profile_xyz_array = np.column_stack((x_coordinates, y_coordinates_end, z_coordinates))
    return mesh_between_profiles(start_profile_xyz_array, end_profile_xyz_array, no_y_nodes, tag)


def mesh_along_z_axis(x_coordinates: ndarray, y_coordinates: ndarray, z_coordinates_start: ndarray, z_coordinates_end: ndarray,
                      no_z_nodes: int, tag: str) -> PolyData:
    """
    Helper function to create a mesh along the Z-axis between two points.
    
    Parameters are arrays of X, Y, starting Z, and ending Z coordinates, along with the number of nodes
    to interpolate along the Z direction. Calls mesh_between_profiles to generate the mesh.

    Parameters
    --------- 
    x_coordinates : ndarray
        X coordinates for meshing.
    y_coordinates : ndarray
        Y coordinates for meshing.
    z_coordinates_start : ndarray
        Starting Z coordinates for meshing.
    z_coordinates_end : ndarray
        Ending Z coordinates for meshing.
    no_z_nodes : int
        Number of nodes to use along the Z-axis.
    tag : str
        A tag to identify the mesh.

    Returns
    -------
    PolyData
        The mesh generated along the Z-axis as a PyVista PolyData object.
    """
    start_profile_xyz_array = np.column_stack((x_coordinates, y_coordinates, z_coordinates_start))
    end_profile_xyz_array = np.column_stack((x_coordinates, y_coordinates, z_coordinates_end))
    return mesh_between_profiles(start_profile_xyz_array, end_profile_xyz_array, no_z_nodes, tag)


def find_coordinates_along_arc(x_c: float, z_c: float, r: float, p1: ndarray, p2: ndarray, y_start: float, y_end: float,
                               element_length: float) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Calculates the coordinates for nodes along an arc defined by its center, radius, and two points on the arc. This is
    used to discretize arcs into segments of a specified maximum length.

    Parameters
    ----------
    x_c, z_c : float
        The x and z coordinates of the arc's center.
    r : float
        The radius of the arc.
    p1, p2 : ndarray
        The starting and ending points of the arc segment.
    y_start, y_end : float
        The y coordinates for the start and end of the arc, assuming the arc is extruded linearly in y-direction.
    element_length : float
        The target maximum length of each segment along the arc.

    Returns
    -------
    Tuple[ndarray, ndarray, ndarray, ndarray]
        The x, starting y, ending y, and z coordinates of the nodes along the arc.
    """
    # Calculate the initial angle of the arc based on the starting point and the center
    angle_0 = np.arctan2(p1[1] - z_c, p1[0] - x_c)
    # Calculate the total angle spanned by the arc segment between points p1 and p2
    segment_arc_angle = np.arccos((2*r**2 - np.linalg.norm(p1 - p2)**2) / (2*r**2))
    # Determine the length of the arc segment using the radius and the segment's angle
    segment_arc_length = r * segment_arc_angle
    # Calculate the number of nodes required to discretize the arc segment based on the specified element length
    no_arc_segment_nodes = discretize_length(segment_arc_length, element_length)
    # Generate angles for each node along the arc segment, evenly spaced over the segment's total angle
    arc_angles = np.linspace(0, segment_arc_angle, no_arc_segment_nodes)
    # Calculate the x coordinates of the nodes along the arc using cosine
    x_coordinates = x_c + r * np.cos(angle_0 - arc_angles)
    # Create arrays for the starting and ending y coordinates, filled with the specified y_start and y_end values
    y_coordinates_start = y_start * np.ones(no_arc_segment_nodes)
    y_coordinates_end = y_end * np.ones(no_arc_segment_nodes)
    # Calculate the z coordinates of the nodes along the arc using sine
    z_coordinates = z_c + r * np.sin(angle_0 - arc_angles)
    # Return the calculated coordinates as four arrays: x, y_start, y_end, and z
    return x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates


def mesh_stiffened_box_beam_with_curved_skins(height: float, width: float, arc_height: float, ribs_y_coordinates: ndarray,
                                              stiffeners_x_coordinates: ndarray, stiffeners_height: float, element_length: float) -> PolyData:
    """
    Creates a mesh for a box beam with curved skins, reinforced with ribs and stiffeners. The beam is discretized into quadrilateral
    shell elements. It uses PyVista for mesh generation, handling curved surfaces with special consideration.

    Parameters
    ----------
    height : float
        The height of the box beam.
    width : float
        The width of the box beam.
    arc_height : float
        The height of the arc (curvature) on the top and bottom skins.
    ribs_y_coordinates : ndarray
        The y coordinates of the ribs within the beam.
    stiffeners_x_coordinates : ndarray
        The x coordinates of the stiffeners along the beam.
    stiffeners_height : float
        The height of the stiffeners.
    element_length : float
        The target length of the elements used in mesh discretization.

    Returns
    -------
    cleaned_box_beam_mesh : PolyData
        The final cleaned mesh of the box beam, including all discretized elements.
    """
    # Initialize the list to store individual mesh segments
    meshes = []
    # Calculate the arc's radius and center position based on beam geometry
    r = ((width / 2) ** 2 + arc_height ** 2) / (2 * arc_height)
    x_c = width / 2
    z_c_top = -r + arc_height + height / 2

    # Function to convert x-coordinate to z-coordinate on the arc, given the arc's geometry
    def x2z(x, r, x_c, z_c):
        return z_c + np.sqrt(r ** 2 - (x - x_c) ** 2)
    
    # Prepare coordinates for the width segments
    width_segment_x_coordinates = np.hstack(([0.], stiffeners_x_coordinates, [width]))
    width_segment_z_coordinates = x2z(width_segment_x_coordinates, r, x_c, z_c_top)
    # Pair up start and end points for segments to mesh
    width_segment_xz_coordinates_start = np.column_stack((width_segment_x_coordinates[:-2], width_segment_z_coordinates[:-2]))
    width_segment_xz_coordinates_end = np.column_stack((width_segment_x_coordinates[1:-1], width_segment_z_coordinates[1:-1]))
    # Determine the number of nodes along the spar and stiffeners based on their heights
    no_z_nodes_spar = discretize_length(height, element_length)
    no_z_nodes_stiffener = discretize_length(stiffeners_height, element_length)
    # Loop through each pair of ribs to create segments between them
    for i, y_start in enumerate(ribs_y_coordinates[:-1]):
        y_end = ribs_y_coordinates[i + 1]
        no_y_nodes = discretize_length(y_end - y_start, element_length)
        # Set up coordinates for meshing the front and rear spars
        x_coordinates = np.zeros(no_z_nodes_spar)
        y_coordinates_start = y_start * np.ones(no_z_nodes_spar)
        y_coordinates_end = y_end * np.ones(no_z_nodes_spar)
        z_coordinates = np.linspace(-height / 2, height / 2, no_z_nodes_spar)
        # Add front and rear spar segments to the mesh list
        meshes.append(mesh_along_y_axis(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes, "front spar"))
        meshes.append(mesh_along_y_axis(x_coordinates + width, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes, "rear spar"))
        # Mesh each section of the top and bottom skins and the stiffeners
        for j, _ in enumerate(width_segment_xz_coordinates_start):
            # Determine coordinates along the arc for top skin segments and their mirrored bottom segments
            x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates = find_coordinates_along_arc(
                x_c, z_c_top, r, width_segment_xz_coordinates_start[j], width_segment_xz_coordinates_end[j], y_start,
                y_end, element_length)
            # Mesh top and bottom skin segments
            meshes.append(mesh_along_y_axis(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes, "top skin"))
            meshes.append(mesh_along_y_axis(x_coordinates, y_coordinates_start, y_coordinates_end, -z_coordinates, no_y_nodes, "bottom skin"))
            # Mesh rib segment
            meshes.append(mesh_along_z_axis(x_coordinates, y_coordinates_start, z_coordinates, -z_coordinates, no_z_nodes_spar, "rib"))
            # Determine the coordinates for the stiffeners
            x_coordinates = stiffeners_x_coordinates[j]*np.ones(no_z_nodes_stiffener)
            y_coordinates_start = y_start * np.ones(no_z_nodes_stiffener)
            y_coordinates_end = y_end * np.ones(no_z_nodes_stiffener)
            z_coordinates = np.linspace(z_coordinates[-1] - stiffeners_height, z_coordinates[-1], no_z_nodes_stiffener)
            # Mesh top skin stiffener
            meshes.append(mesh_along_y_axis(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes, "stiffener"))
            # Mesh bottom skin stiffener, mirroring the top
            meshes.append(mesh_along_y_axis(x_coordinates, y_coordinates_start, y_coordinates_end, -z_coordinates, no_y_nodes, "stiffener"))
        # Determine the coordinates for the final top and bottom skin segments
        x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates = find_coordinates_along_arc(
            x_c, z_c_top, r, width_segment_xz_coordinates_end[-1], np.array([width, height/2]), y_start,
            y_end, element_length)
        # Mesh the final top and bottom skin segments
        meshes.append(mesh_along_y_axis(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes, "top skin"))
        meshes.append(mesh_along_y_axis(x_coordinates, y_coordinates_start, y_coordinates_end, -z_coordinates, no_y_nodes, "bottom skin"))
        # Mesh the final rib segment
        meshes.append(mesh_along_z_axis(x_coordinates, y_coordinates_start, z_coordinates, -z_coordinates, no_z_nodes_spar, "rib"))
    # Prepare coordinates for the last rib
    width_segment_xz_coordinates_start = np.column_stack((width_segment_x_coordinates[:-1], width_segment_z_coordinates[:-1]))
    width_segment_xz_coordinates_end = np.column_stack((width_segment_x_coordinates[1:], width_segment_z_coordinates[1:]))
    # Mesh the last rib segment by segment
    for j, _ in enumerate(width_segment_xz_coordinates_start):
        # Determine coordinates to mesh the rib segment
        x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates = find_coordinates_along_arc(
            x_c, z_c_top, r, width_segment_xz_coordinates_start[j], width_segment_xz_coordinates_end[j], ribs_y_coordinates[-1],
            ribs_y_coordinates[-1], element_length)
        # Mesh rib segment
        meshes.append(mesh_along_z_axis(x_coordinates, y_coordinates_start, z_coordinates, -z_coordinates, no_z_nodes_spar, "rib"))
    # Merge all the mesh segments together
    merged_box_beam_mesh = meshes[0].merge(meshes[1:])
    # Clean the obtained mesh by merging points closer than a specified tolerance
    cleaned_box_beam_mesh = merged_box_beam_mesh.clean(tolerance=element_length / 100)

    # Tag the points based on the faces they belong to
    # Step 1: gather the indices of points that form each face
    cells = cleaned_box_beam_mesh.faces.reshape(-1, 5)[:, 1:]  # assume quad cells
    point_indices = cells.flatten()  # flatten the cells array to get a list of point indices, repeated per cell occurrence
    cell_tags_repeated = np.repeat(cleaned_box_beam_mesh.cell_data['tag'], 4)  # array of the same shape as point_indices, where each cell tag is repeated for each of its points
    # Step 2: Map cell tags to point tags using an indirect sorting approach
    sort_order = np.argsort(point_indices)  # get the sort order to rearrange the point indices
    sorted_point_indices = point_indices[sort_order]  # sort the point indices
    sorted_tags = cell_tags_repeated[sort_order]  # sort the cell tags in the same order
    _, boundaries_indices = np.unique(sorted_point_indices, return_index=True)  # find the boundaries between different points in the sorted point indices
    # Step 3: Split the sorted tags array at these boundaries to get lists of tags for each point
    tags_split = np.split(sorted_tags, boundaries_indices[1:])  # split the sorted tags array at the boundaries to get lists of tags for each point
    point_tags_list = np.array([', '.join(tags) for tags in tags_split])  # convert each list of tags into a comma-separated string
    cleaned_box_beam_mesh.point_data['tags'] = point_tags_list  # apply the tags to the point_data of the mesh
    
    # Return the cleaned, final mesh
    return cleaned_box_beam_mesh
