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


def tag_points(mesh: PolyData):
    """
    Assigns tags to the points in a mesh based on the faces they belong to. This is a post-processing step after mesh generation
    to facilitate further analysis or visualization based on specific features or sections of the mesh.

    Parameters
    ----------
    mesh : PolyData
        A PyVista PolyData object containing the mesh data.

    """
    # Tag the points based on the faces they belong to
    # Step 1: gather the indices of points that form each face
    cells = mesh.faces.reshape(-1, 5)[:, 1:]  # assume quad cells
    point_indices = cells.flatten()  # flatten the cells array to get a list of point indices, repeated per cell occurrence
    cell_tags_repeated = np.repeat(mesh.cell_data['tag'], 4)  # array of the same shape as point_indices, where each cell tag is repeated for each of its points
    # Step 2: Map cell tags to point tags using an indirect sorting approach
    sort_order = np.argsort(point_indices)  # get the sort order to rearrange the point indices
    sorted_point_indices = point_indices[sort_order]  # sort the point indices
    sorted_tags = cell_tags_repeated[sort_order]  # sort the cell tags in the same order
    _, boundaries_indices = np.unique(sorted_point_indices, return_index=True)  # find the boundaries between different points in the sorted point indices
    # Step 3: Split the sorted tags array at these boundaries to get lists of tags for each point
    tags_split = np.split(sorted_tags, boundaries_indices[1:])  # split the sorted tags array at the boundaries to get lists of tags for each point
    point_tags_list = np.array([', '.join(tags) for tags in tags_split])  # convert each list of tags into a comma-separated string
    mesh.point_data['tags'] = point_tags_list  # apply the tags to the point_data of the mesh
    
def mesh_spars_segment(no_z_nodes, no_y_nodes, y_start, y_end, height, width):
    """
    Generates mesh for the spars of a box beam structure, using coordinates along the Y-axis for the main spar and the rear spar.
    This function generates two separate meshes for the front and rear spars of the box beam.

    Parameters
    ----------
    no_z_nodes : int
        Number of nodes to use along the Z-axis (height) of the spar.
    no_y_nodes : int
        Number of nodes to use along the Y-axis (length) between the front and rear spars.
    y_start : float
        The starting Y coordinate of the spars.
    y_end : float
        The ending Y coordinate of the spars.
    height : float
        The height of the spars.
    width : float
        The width between the front and rear spars, used to set the X coordinate of the rear spar.

    Returns
    -------
    list
        A list containing two PyVista PolyData objects, one for the front spar and one for the rear spar.
    """
    # Set up coordinates for meshing the front and rear spar segment
    x_coordinates = np.zeros(no_z_nodes)
    y_coordinates_start = y_start * np.ones(no_z_nodes)
    y_coordinates_end = y_end * np.ones(no_z_nodes)
    z_coordinates = np.linspace(-height / 2, height / 2, no_z_nodes)

    # Mesh front and rear spar segment as PolyData objects
    front_spar_mesh = mesh_along_y_axis(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes, "front spar")
    rear_spar_mesh = mesh_along_y_axis(x_coordinates + width, y_coordinates_end, y_coordinates_start, z_coordinates, no_y_nodes, "rear spar")  # swap start and end coordinates to ensure that all normal vectors to the elements point outward
    
    # Return list with PolyData objects
    return [front_spar_mesh, rear_spar_mesh]


def mesh_skins_segment(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes):
    """
    Generates mesh for the top and bottom skins of a box beam structure. This function handles the mesh generation by projecting
    the provided coordinates along the Y-axis for the skins.

    Parameters
    ----------
    x_coordinates : ndarray
        X coordinates for the skin meshing.
    y_coordinates_start : ndarray
        Starting Y coordinates for the skin meshing.
    y_coordinates_end : ndarray
        Ending Y coordinates for the skin meshing.
    z_coordinates : ndarray
        Z coordinates for the skin meshing, which are mirrored for the bottom skin to create a symmetrical structure.
    no_y_nodes : int
        Number of nodes to use along the Y-axis for interpolating between start and end coordinates.

    Returns
    -------
    list
        A list containing two PyVista PolyData objects, one for the top skin and one for the bottom skin.
    """
    # Mesh top and bottom skin segments as PolyData objects
    top_skin_mesh = mesh_along_y_axis(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes, "top skin")
    bottom_skin_mesh = mesh_along_y_axis(x_coordinates, y_coordinates_end, y_coordinates_start, -z_coordinates, no_y_nodes, "bottom skin")  # swap start and end coordinates to ensure that all normal vectors to the elements point outward
    
    # Return list with PolyData objects
    return [top_skin_mesh, bottom_skin_mesh]


def mesh_rib_skins_segment(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes, no_z_nodes):
    """
    Generates meshes for the rib and skin segments of a box beam structure. This function handles the mesh generation for the rib
    using the Z-axis and additional top and bottom skins using the Y-axis.

    Parameters
    ----------
    x_coordinates : ndarray
        X coordinates for the rib and skin meshing.
    y_coordinates_start : ndarray
        Starting Y coordinates for the skin meshing.
    y_coordinates_end : ndarray
        Ending Y coordinates for the skin meshing.
    z_coordinates : ndarray
        Z coordinates for the rib meshing.
    no_y_nodes : int
        Number of nodes to use along the Y-axis for interpolating between start and end coordinates for the skins.
    no_z_nodes : int
        Number of nodes to use along the Z-axis for interpolating along the rib segment.

    Returns
    -------
    list
        A list containing three PyVista PolyData objects, one for the rib and two for the top and bottom skins.
    """
    # Mesh rib segment as a PolyData object
    rib_mesh = mesh_along_z_axis(x_coordinates, y_coordinates_start, z_coordinates, -z_coordinates, no_z_nodes, "rib")
    
    # Mesh top and bottom skin segments as PolyData objects
    skin_meshes = mesh_skins_segment(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes)
    
    # Return list with PolyData objects
    return [rib_mesh] + skin_meshes


def mesh_stiffeners_segment(x, y_start, y_end, z, no_y_nodes, no_z_nodes, height):
    """
    Generates meshes for stiffeners along a box beam structure, handling the positioning and orientation along the X and Z axes,
    and interpolating along the Y-axis. 

    Parameters
    ----------
    x : float
        X coordinate for the position of the stiffeners.
    y_start : float
        Starting Y coordinate for the stiffeners.
    y_end : float
        Ending Y coordinate for the stiffeners.
    z : float
        Z coordinate for the base of the stiffeners.
    no_y_nodes : int
        Number of nodes to use along the Y-axis for interpolating between the start and end coordinates.
    no_z_nodes : int
        Number of nodes to use along the Z-axis to create the height of the stiffeners.
    height : float
        The height of the stiffeners from the base Z coordinate.

    Returns
    -------
    list
        A list containing two PyVista PolyData objects, one for the top skin stiffener and one for the bottom skin stiffener.
    """
    # Determine the coordinates for the stiffeners
    x_coordinates = x * np.ones(no_z_nodes)
    y_coordinates_start = y_start * np.ones(no_z_nodes)
    y_coordinates_end = y_end * np.ones(no_z_nodes)
    z_coordinates = np.linspace(z - height, z, no_z_nodes)

    # Mesh top and bottom skin stiffeners as PolyData objects
    top_skin_stiffener = mesh_along_y_axis(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes, "stiffener")
    bottom_skin_stiffener = mesh_along_y_axis(x_coordinates, y_coordinates_start, y_coordinates_end, -z_coordinates, no_y_nodes, "stiffener")  # mirror along z-axis
    
    # Return list with PolyData objects
    return [top_skin_stiffener, bottom_skin_stiffener]


def merge_meshes(mesh_list, element_length):
    """
    Merges multiple mesh segments into a single unified mesh. This function also cleans the mesh by merging close points and
    tagging points based on the faces they belong to, facilitating smoother and more coherent mesh structures.

    Parameters
    ----------
    mesh_list : list
        List of PolyData objects representing different mesh segments to be merged.
    element_length : float
        The target length of elements used in mesh discretization, used to set the tolerance for point merging.

    Returns
    -------
    PolyData
        The final cleaned and unified mesh as a PyVista PolyData object.
    """
    # Merge all the mesh segments together
    merged_mesh = mesh_list[0].merge(mesh_list[1:])
    
    # Clean the obtained mesh by merging points closer than a specified tolerance
    cleaned_mesh = merged_mesh.clean(tolerance=element_length / 100)

    # Tag the points based on the faces they belong to
    tag_points(cleaned_mesh)
    
    # Return the cleaned, final mesh
    return cleaned_mesh


def mesh_box_beam(height: float, width: float, length: ndarray, element_length: float) -> PolyData:
    """
    Generates a mesh for a basic box beam structure using given dimensions and element length.
    This function handles the entire process of mesh generation including calculating node distribution, creating meshes for spars,
    skins, and merging them into a final structure.

    Parameters
    ----------
    height : float
        The height of the box beam.
    width : float
        The width of the box beam.
    length : float
        The length along the Y-axis of the box beam.
    element_length : float
        The target length of the elements used in mesh discretization.

    Returns
    -------
    PolyData
        A PyVista PolyData object containing the fully meshed box beam.
    """
    # Initialize the list of PolyData objects storing the individual mesh segments
    meshes = []

    # Determine the number of nodes along the spars based on their heights
    no_z_nodes_spar = discretize_length(height, element_length)
    
    # Determine the number of nodes and the x-z coordinates along the width based on the element length
    no_x_nodes = discretize_length(width, element_length)
    x_coordinates = np.linspace(0, width, no_x_nodes)
    z_coordinates =  height / 2 * np.ones(no_x_nodes)
    
    # Determine the number of nodes along the y-axis for the segment
    y_start = 0.
    y_end = length
    no_y_nodes = discretize_length(y_end - y_start, element_length)

    # Mesh the spars segment and add PolyData objects to the list
    meshes += mesh_spars_segment(no_z_nodes_spar, no_y_nodes, y_start, y_end, height, width)
        
    # Determine coordinates along the segment for top skin
    y_coordinates_start = y_start * np.ones(no_x_nodes)
    y_coordinates_end = y_end * np.ones(no_x_nodes)
        
    # Mesh segment with rib and top and bottom skins
    meshes += mesh_skins_segment(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes)
    
    # Merge all the mesh segments together
    cleaned_box_beam_mesh = merge_meshes(meshes, element_length)
    return cleaned_box_beam_mesh


def mesh_box_beam_reinforced_with_ribs(height: float, width: float, ribs_y_coordinates: ndarray, element_length: float) -> PolyData:
    """
    Generates a mesh for a box beam structure reinforced with ribs. The function handles the mesh generation
    for spars and skin segments between each rib and then merges them into a final structure.

    Parameters
    ----------
    height : float
        The height of the box beam.
    width : float
        The width of the box beam.
    ribs_y_coordinates : ndarray
        The y coordinates where ribs are located within the beam.
    element_length : float
        The target length of the elements used in mesh discretization.

    Returns
    -------
    PolyData
        A PyVista PolyData object containing the meshed box beam with ribs.
    """
    # Initialize the list of PolyData objects storing the individual mesh segments
    meshes = []

    # Determine the number of nodes along the spars based on their heights
    no_z_nodes_spar = discretize_length(height, element_length)
    
    # Determine the number of nodes and the x-z coordinates along the width based on the element length
    no_x_nodes = discretize_length(width, element_length)
    x_coordinates = np.linspace(0, width, no_x_nodes)
    z_coordinates =  height / 2 * np.ones(no_x_nodes)

    # Loop through each pair of ribs to create segments between them
    for i, y_start in enumerate(ribs_y_coordinates[:-1]):
        # Determine the number of nodes along the y-axis for the segment
        y_end = ribs_y_coordinates[i + 1]
        no_y_nodes = discretize_length(y_end - y_start, element_length)

        # Mesh the spars segment and add PolyData objects to the list
        meshes += mesh_spars_segment(no_z_nodes_spar, no_y_nodes, y_start, y_end, height, width)
        
        # Determine coordinates along the segment for top skin
        y_coordinates_start = y_start * np.ones(no_x_nodes)
        y_coordinates_end = y_end * np.ones(no_x_nodes)
        
        # Mesh segment with rib and top and bottom skins
        meshes += mesh_rib_skins_segment(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes, no_z_nodes_spar)

    # Determine coordinates for last rib
    y_coordinates = y_end * np.ones(no_x_nodes)

    # Mesh the last rib
    meshes.append(mesh_along_z_axis(x_coordinates, y_coordinates, z_coordinates, -z_coordinates, no_z_nodes_spar, "rib"))
    
    # Merge all the mesh segments together
    cleaned_box_beam_mesh = merge_meshes(meshes, element_length)
    return cleaned_box_beam_mesh


def mesh_box_beam_reinforced_with_ribs_and_stiffeners(height: float, width: float, ribs_y_coordinates: ndarray,
                                                      stiffeners_x_coordinates: ndarray, stiffeners_height: float, element_length: float) -> PolyData:
    """
    Creates a mesh for a box beam reinforced with ribs and stiffeners. The function handles the mesh generation
    for spars and skin segments between each rib and for skin and rib segments between stiffeners. Finally, the
    function merges the segments into a final mesh.

    Parameters
    ----------
    height : float
        The height of the box beam.
    width : float
        The width of the box beam.
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
    # Initialize the list of PolyData objects storing the individual mesh segments
    meshes = []
    
    # Prepare coordinates for the width segments
    width_segment_x_coordinates = np.hstack(([0.], stiffeners_x_coordinates, [width]))

    # Pair up start and end points for segments to mesh
    width_segment_x_coordinates_start = width_segment_x_coordinates[:-2]
    width_segment_x_coordinates_end = width_segment_x_coordinates[1:-1]

    # Determine the number of nodes along the spar and stiffeners based on their heights
    no_z_nodes_spar = discretize_length(height, element_length)
    no_z_nodes_stiffener = discretize_length(stiffeners_height, element_length)

    # Loop through each pair of ribs to create segments between them
    for i, y_start in enumerate(ribs_y_coordinates[:-1]):
        # Determine the number of nodes along the y-axis for the segment
        y_end = ribs_y_coordinates[i + 1]
        no_y_nodes = discretize_length(y_end - y_start, element_length)

        # Mesh the spars segment and add PolyData objects to the list
        meshes += mesh_spars_segment(no_z_nodes_spar, no_y_nodes, y_start, y_end, height, width)

        # Iterate through the starting coordinates of the skin segments between stiffeners
        for j, _ in enumerate(width_segment_x_coordinates_start):
            # Determine coordinates along the segment for top skin
            no_x_nodes = discretize_length(width_segment_x_coordinates_end[j] - width_segment_x_coordinates_start[j], element_length)
            x_coordinates = np.linspace(width_segment_x_coordinates_start[j], width_segment_x_coordinates_end[j], no_x_nodes)
            y_coordinates_start = y_start * np.ones(no_x_nodes)
            y_coordinates_end = y_end * np.ones(no_x_nodes)
            z_coordinates =  height / 2 * np.ones(no_x_nodes)
            
            # Mesh segment with rib and top and bottom skins
            meshes += mesh_rib_skins_segment(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes, no_z_nodes_spar)
            
            # Mesh the stiffeners segment
            meshes += mesh_stiffeners_segment(stiffeners_x_coordinates[j], y_start, y_end, z_coordinates[-1], no_y_nodes, no_z_nodes_stiffener, stiffeners_height)

        # Determine the coordinates for the final top skin segment
        no_x_nodes = discretize_length(width - width_segment_x_coordinates_end[-1], element_length)
        x_coordinates = np.linspace(width_segment_x_coordinates_end[-1], width, no_x_nodes)
        y_coordinates_start = y_start * np.ones(no_x_nodes)
        y_coordinates_end = y_end * np.ones(no_x_nodes)
        z_coordinates =  height / 2 * np.ones(no_x_nodes)
        
        # Mesh final segment with rib and top and bottom skins
        meshes += mesh_rib_skins_segment(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes, no_z_nodes_spar)

    # Prepare coordinates for the last rib
    width_segment_x_coordinates_start = width_segment_x_coordinates[:-1]
    width_segment_x_coordinates_end = width_segment_x_coordinates[1:]

    # Mesh the last rib segment by segment
    for j, _ in enumerate(width_segment_x_coordinates_start):
        # Determine coordinates to mesh the rib segment
        no_x_nodes = discretize_length(width_segment_x_coordinates_end[j] - width_segment_x_coordinates_start[j], element_length)
        x_coordinates = np.linspace(width_segment_x_coordinates_start[j], width_segment_x_coordinates_end[j], no_x_nodes)
        y_coordinates_start = ribs_y_coordinates[-1] * np.ones(no_x_nodes)
        z_coordinates =  height / 2 * np.ones(no_x_nodes)
        
        # Mesh rib segment
        meshes.append(mesh_along_z_axis(x_coordinates, y_coordinates_start, z_coordinates, -z_coordinates, no_z_nodes_spar, "rib"))

    # Merge all the mesh segments together
    cleaned_box_beam_mesh = merge_meshes(meshes, element_length)
    return cleaned_box_beam_mesh


def mesh_reinforced_box_beam_with_curved_skins(height: float, width: float, arc_height: float, ribs_y_coordinates: ndarray,
                                               stiffeners_x_coordinates: ndarray, stiffeners_height: float, element_length: float) -> PolyData:
    """
    Creates a mesh for a box beam reinforced with ribs and stiffeners and employing curved skins. The beam is discretized into quadrilateral
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
        # Determine the number of nodes along the y-axis for the segment
        y_end = ribs_y_coordinates[i + 1]
        no_y_nodes = discretize_length(y_end - y_start, element_length)
        
        # Mesh the spars segment and add PolyData objects to the list
        meshes += mesh_spars_segment(no_z_nodes_spar, no_y_nodes, y_start, y_end, height, width)
        
        # Mesh each section of the top and bottom skins and the stiffeners
        for j, _ in enumerate(width_segment_xz_coordinates_start):
            # Determine coordinates along the arc for top skin segments and their mirrored bottom segments
            x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates = find_coordinates_along_arc(
                x_c, z_c_top, r, width_segment_xz_coordinates_start[j], width_segment_xz_coordinates_end[j], y_start,
                y_end, element_length)
            
            # Mesh segment with rib and top and bottom skins
            meshes += mesh_rib_skins_segment(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes, no_z_nodes_spar)
            
            # Mesh the stiffeners segment
            meshes += mesh_stiffeners_segment(stiffeners_x_coordinates[j], y_start, y_end, z_coordinates[-1], no_y_nodes, no_z_nodes_stiffener, stiffeners_height)

        # Determine the coordinates for the final top and bottom skin segments
        x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates = find_coordinates_along_arc(
            x_c, z_c_top, r, width_segment_xz_coordinates_end[-1], np.array([width, height/2]), y_start,
            y_end, element_length)
        
        # Mesh segment with rib and top and bottom skins
        meshes += mesh_rib_skins_segment(x_coordinates, y_coordinates_start, y_coordinates_end, z_coordinates, no_y_nodes, no_z_nodes_spar)
    
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
    cleaned_box_beam_mesh = merge_meshes(meshes, element_length)
    return cleaned_box_beam_mesh


def create_base_bdf_input(young_modulus: float, poisson_ratio: float, density: float, shell_thickness: float,
                          nodes_xyz_array: ndarray, nodes_connectivity_matrix: ndarray, parallel:bool = False,
                          no_cores:int = 4) -> BDF:
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
