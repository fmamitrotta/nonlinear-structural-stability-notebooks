"""
This file is part of the GitHub repository nonlinear-structural-stability-notebooks, created by Francesco M. A. Mitrotta.
Copyright (C) 2024 Francesco Mario Antonio Mitrotta

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import matplotlib.pyplot as plt  # import matplotlib for plotting
import tol_colors as tc  # package for colorblind-friendly colors
from matplotlib.axes import Axes  # import Axes class for type hinting
from mpl_toolkits.mplot3d.axes3d import Axes3D  # import Axes3D class for 3D plotting
from numpy import ndarray  # import ndarray for type hinting
import numpy as np  # import numpy for numerical operations
from pyNastran.op2.op2 import OP2  # import OP2 class for type hinting
from matplotlib.figure import Figure  # import Figure class for type hinting
from matplotlib.cm import ScalarMappable  # import ScalarMappable class for type hinting
from typing import Tuple, Union  # import Tuple and Union for type hinting
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # import Poly3DCollection class for 3D plotting
from matplotlib.colorbar import Colorbar  # import Colorbar class for type hinting


# Set default color cycle to TOL bright and register Tol's color-blind friendly colormaps
plt.rc('axes', prop_cycle=plt.cycler('color', list(tc.tol_cset('bright'))))
plt.cm.register_cmap('sunset', tc.tol_cmap('sunset'))
plt.cm.register_cmap('rainbow_PuRd', tc.tol_cmap('rainbow_PuRd'))

# Define list of colors and red color for unstable segments
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']  # list with default color cycle
UNSTABLE_COLOR = COLORS[1]  # set red color for unstable segments


def plot_2d_load_displacements_stability(axes: Axes, displacements: ndarray, loads: ndarray, eigenvalues: ndarray, marker: str, color: str) -> int:
    """
    Plot a load-displacement diagram highlighting the stability of the equilibrium points.

    Parameters
    ----------
    axes: Axes
        object of the axes where the load-displacement diagram will be plotted
    displacements: ndarray
        numpy array of the displacements that will be plotted on the x-axis
    loads: ndarray
        numpy array of the loads that will be plotted on the y-axis
    eigenvalues : ndarray
        numpy array with the eigenvalues of the tangent stiffness matrix for each equilibrium point
    marker: str
        string with the marker style for the plot
    color: str
        string with the color that will be used for the stable segments of the load-displacement curve
    
    Returns
    -------
    actual_neg_eigenvalues: int
        integer with the number of actual negative eigenvalues at the last equilibrium point
    """
    
    # Group the equilibrium points into segments with a constant number of negative eigenvalues
    change_no_negative_eigenvalues_indices = np.where(np.diff((eigenvalues < 0).sum(axis=0)))[0] + 1  # find the indices where the number of negative eigenvalues changes
    segments_indices_list = [list(range(i, j)) for i, j in zip(
        [0] + list(change_no_negative_eigenvalues_indices), list(change_no_negative_eigenvalues_indices) + [eigenvalues.shape[1]])]  # create a list of arrays including the indices of each segment
    
    # Count the number of negative eigenvalues for each segment
    no_neg_eigvals = [np.sum(eigenvalues[:, segment[0]] < 0, axis=0) for segment in segments_indices_list]
    
    # Create a list of arrays with the plotting indices for each segment
    plot_indices_list = [indices + [indices[-1] + 1] for indices in segments_indices_list[:-1]]  # the first point of the next segment is added to each segment to make the path appear continuous
    plot_indices_list.append(segments_indices_list[-1])  # add the last segment to the plotting list
    
    # Plot the first segment
    actual_neg_eigenvalues = no_neg_eigvals[0]  # get the number of negative eigenvalues at the first equilibrium point
    
    # Plot with a solid line if segment is stable
    if actual_neg_eigenvalues == 0:
        axes.plot(
            displacements[plot_indices_list[0]], loads[plot_indices_list[0]], marker + '-', color=color)
    
    # Plot with a dashed red line if segment is unstable
    else:
        axes.plot(
            displacements[plot_indices_list[0]], loads[plot_indices_list[0]], marker + '--', color=UNSTABLE_COLOR)
    
    # Plot the remaining segments
    lowest_eigenvalue_row = eigenvalues[0]  # get array of lowest eigenvalues across all equilibrium points
    for count, current_segment in enumerate(segments_indices_list[1:]):  # loop through the remaining segments
        last_segment = segments_indices_list[count]  # store array of indices of the last segment
        
        # Predict the the lowest eigenvalue at the first equilibrium point of the current segment based on the last two points of the last segment
        lowest_new_eigenvalue = lowest_eigenvalue_row[current_segment[0]]  # get the lowest eigenvalue at the first equilibrium point of the segment
        last_segment_end_index = last_segment[-1]  # get the index of the last equilibrium point of the last segment
        predicted_eigenvalue = lowest_eigenvalue_row[last_segment_end_index] +\
            np.diff(lowest_eigenvalue_row[last_segment_end_index - 1:last_segment_end_index + 1])  # predict the lowest eigenvalue at the first equilibrium point of the current segment by linearly extrapolating the last two points
        tolerance = np.std(np.concatenate(
            (np.abs(np.diff(lowest_eigenvalue_row[last_segment])), np.abs(np.diff(lowest_eigenvalue_row[current_segment])))))*3  # set the tolerance as three times the standard deviation of the absolute value of the change in the lowest eigenvalue during last and current segment
        
        # If the new lowest eigenvalue is close to the predicted value update the actual number of negative eigenvalues
        if np.abs(lowest_new_eigenvalue - predicted_eigenvalue) <= tolerance:
            delta_num_neg_eigvals = no_neg_eigvals[count + 1] - no_neg_eigvals[count]  # get the change in the number of negative eigenvalues
            actual_neg_eigenvalues += delta_num_neg_eigvals
            
        # Plot with a solid line if segment is stable
        if actual_neg_eigenvalues == 0:
            axes.plot(
                displacements[plot_indices_list[count + 1]], loads[plot_indices_list[count + 1]], marker + '-', color=color)
            
        # Plot with a dashed red line if segment is unstable
        else:
            axes.plot(
                displacements[plot_indices_list[count + 1]], loads[plot_indices_list[count + 1]], marker + '--', color=UNSTABLE_COLOR)
        
    # Return current number of actual negative eigenvalues
    return actual_neg_eigenvalues


def plot_3d_load_displacements_stability(axes: Axes3D, displacements1: ndarray, displacements2: ndarray, loads: ndarray, eigenvalues: ndarray,
                                         marker: str, color: str):
    """
    Plot a 3D load-displacement diagram highlighting the stability of the equilibrium points.

    Parameters
    ----------
    axes: Axes3D
        object of the axes where the load-displacement diagram will be plotted
    displacements1: ndarray
        numpy array of the displacements that will be plotted on the x-axis
    displacements2: ndarray
        numpy array of the displacements that will be plotted on the y-axis
    loads: ndarray
        numpy array of the loads that will be plotted on the z-axis
    eigenvalues : ndarray
        numpy array with the eigenvalues of the tangent stiffness matrix for each equilibrium point
    marker: str
        string with the marker style for the plot
    color: str
        string with the color that will be used for the stable segments of the load-displacement curve
    """
    
    # Group the equilibrium points into segments with a constant number of negative eigenvalues
    change_no_negative_eigenvalues_indices = np.where(np.diff((eigenvalues < 0).sum(axis=0)))[0] + 1  # find the indices where the number of negative eigenvalues changes
    segments_indices_list = [list(range(i, j)) for i, j in zip(
        [0] + list(change_no_negative_eigenvalues_indices), list(change_no_negative_eigenvalues_indices) + [eigenvalues.shape[1]])]  # create a list of arrays including the indices of each segment
    
    # Count the number of negative eigenvalues for each segment
    no_neg_eigvals = [np.sum(eigenvalues[:, segment[0]] < 0, axis=0) for segment in segments_indices_list]
    
    # Create a list of arrays with the plotting indices for each segment
    plot_indices_list = [indices + [indices[-1] + 1] for indices in segments_indices_list[:-1]]  # the first point of the next segment is added to each segment to make the path appear continuous
    plot_indices_list.append(segments_indices_list[-1])  # add the last segment to the plotting list
    
    # Plot the first segment
    actual_neg_eigenvalues = no_neg_eigvals[0]  # get the number of negative eigenvalues at the first equilibrium point
    
    # Plot with a solid line if segment is stable
    if actual_neg_eigenvalues == 0:
        axes.plot3D(displacements1[plot_indices_list[0]],
                    displacements2[plot_indices_list[0]],
                    loads[plot_indices_list[0]], marker + '-', color=color)
    
    # Plot with a dashed red line if segment is unstable
    else:
        axes.plot3D(displacements1[plot_indices_list[0]],
                    displacements2[plot_indices_list[0]],
                    loads[plot_indices_list[0]], marker + '--', color=UNSTABLE_COLOR)
    
    # Plot the remaining segments
    lowest_eigenvalue_row = eigenvalues[0]  # get array of lowest eigenvalues across all equilibrium points
    for count, current_segment in enumerate(segments_indices_list[1:]):  # loop through the remaining segments
        last_segment = segments_indices_list[count]  # store array of indices of the last segment
        
        # Predict the the lowest eigenvalue at the first equilibrium point of the current segment based on the last two points of the last segment
        lowest_new_eigenvalue = lowest_eigenvalue_row[current_segment[0]]  # get the lowest eigenvalue at the first equilibrium point of the segment
        last_segment_end_index = last_segment[-1]  # get the index of the last equilibrium point of the last segment
        predicted_eigenvalue = lowest_eigenvalue_row[last_segment_end_index] +\
            np.diff(lowest_eigenvalue_row[last_segment_end_index - 1:last_segment_end_index + 1])  # predict the lowest eigenvalue at the first equilibrium point of the current segment by linearly extrapolating the last two points
        tolerance = np.std(np.concatenate(
            (np.abs(np.diff(lowest_eigenvalue_row[last_segment])), np.abs(np.diff(lowest_eigenvalue_row[current_segment])))))*3  # set the tolerance as three times the standard deviation of the absolute value of the change in the lowest eigenvalue during last and current segment
        
        # If the new lowest eigenvalue is close to the predicted value update the actual number of negative eigenvalues
        if np.abs(lowest_new_eigenvalue - predicted_eigenvalue) <= tolerance:
            delta_num_neg_eigvals = no_neg_eigvals[count + 1] - no_neg_eigvals[count]  # get the change in the number of negative eigenvalues
            actual_neg_eigenvalues += delta_num_neg_eigvals
            
        # Plot with a solid line if segment is stable
        if actual_neg_eigenvalues == 0:
            axes.plot3D(displacements1[plot_indices_list[count + 1]],
                        displacements2[plot_indices_list[count + 1]],
                        loads[plot_indices_list[count + 1]], marker + '-', color=color)
            
        # Plot with a dashed red line if segment is unstable
        else:
            axes.plot3D(displacements1[plot_indices_list[count + 1]],
                        displacements2[plot_indices_list[count + 1]],
                        loads[plot_indices_list[count + 1]], marker + '--', color=UNSTABLE_COLOR)
            

def plot_eigenvalue_diagram(axes: Axes, loads: ndarray, eigenvalues: ndarray):
    """
    Plot the eigenvalues of the tangent stiffness matrix as a function of the load.

    Parameters
    ----------
    axes: Axes
        object of the axes where the eigenvalue diagram will be plotted
    loads: ndarray
        numpy array of the loads that will be plotted on the x-axis
    eigenvalues: ndarray
        numpy array with the eigenvalues of the tangent stiffness matrix for each equilibrium point
    """
    
    last_plot_index = eigenvalues.shape[1]  # analyses that reach the maximum number of iterations do not print the eigenvalues of the last converged iteration
    axes.plot(loads[:last_plot_index], eigenvalues.T, 'o')  # convert eigenvalues from N/mm to N/m
    axes.grid(True)  # add grid to the plot
    

def plot_displacements(op2: OP2, displacement_data: ndarray, axes: Axes3D = None, displacement_component: str = 'magnitude',
                       displacement_unit_scale_factor: float = 1., coordinate_unit_scale_factor:float = 1.,
                       displacement_amplification_factor: float = 1., colormap: str = 'rainbow_PuRd', clim: Union[list, ndarray] = None,
                       length_unit: str = 'mm', angle_unit: str = 'rad') -> Tuple[Figure, Axes3D, ScalarMappable]:
    """
    Plot the deformed shape coloured by displacement magnitude based on input OP2 object and displacement data.

    Parameters
    ----------
    op2: OP2
        pyNastran object created reading an op2 file with the load_geometry option set to True
    displacement_data: ndarray
        array with displacement data to plot
    axes: Axes3D
        object of the axes where the deformed shape will be plotted
    displacement_component: str
        name of the displacement component used for the colorbar
    displacement_unit_scale_factor: float
        unit scale factor for displacements
    coordinate_unit_scale_factor: float
        unit scale factor for nodes coordinates
    displacement_amplification_factor: float
        amplification factor for displacements
    colormap: str
        name of the colormap used for the displacement colorbar
    clim: Union[list, ndarray]:
        colorbar values limits
    length_unit: str
        measurement unit of coordinates and displacements, used in the label of the axes
    angle_unit: str
        measurement unit of angles, used in the label of the colorbar when the displacement component is a rotation

    Returns
    -------
    fig: Figure
        object of the plot's figure
    ax: Axes3D
        object of the plot's axes
    scalar_to_rgba_map: ScalarMappable
        object of the color mapping for the scalar corresponding to the displacement component
    """
    
    # Find mapping from node ids to node indexes and extract coordinates of undeformed nodes
    node_id_to_index = {}
    undeformed_coordinates_array = np.empty((len(op2.nodes), 3))
    for node_index, (node_id, node) in enumerate(op2.nodes.items()):
        node_id_to_index[node_id] = node_index
        undeformed_coordinates_array[node_index] = node.xyz
        
    # Apply scalings and find deformed coordinates
    undeformed_coordinates_array *= coordinate_unit_scale_factor  # apply unit conversion to undeformed coordinates
    displacement_data[:, :3] = displacement_data[:, :3] * displacement_unit_scale_factor  # apply unit conversion to displacements
    deformed_coordinates_array = undeformed_coordinates_array + displacement_data[:, :3] * displacement_amplification_factor  # apply amplification factor to displacements and calculate deformed coordinates
    
    # Find nodes' indices for each shell element (only CQUAD4 and CTRIA3 elements are considered)
    shell_elements_nodes_indices = np.array([
        [node_id_to_index[node_id] for node_id in element.node_ids] for element in op2.elements.values() if
        element.type in ['CTRIA3', 'CQUAD4', 'CTRIAR', 'CQUADR']])
    
    # Find vertices' coordinates for Poly3DCollection
    vertices = deformed_coordinates_array[shell_elements_nodes_indices]
    
    # Create Poly3DCollection
    pc = Poly3DCollection(vertices, linewidths=.05, edgecolor='k')
    
    # Handle displacement component and extract the corresponding array of nodal displacements
    if displacement_component == 'magnitude':
        nodal_displacement_array = np.linalg.norm(displacement_data[:, :3], axis=1)  # calculate displacement magnitude
    else:
        component_dict = {'tx': 0, 'ty': 1, 'tz': 2, 'rx': 3, 'ry': 4, 'rz': 5}
        nodal_displacement_array = displacement_data[:, component_dict[displacement_component]]  # select displacement component
        
        # Convert rotation displacements to degrees if necessary
        if displacement_component in ['rx', 'ry', 'rz'] and angle_unit == 'deg':
            nodal_displacement_array = np.rad2deg(nodal_displacement_array)
        
    # Calculate average displacement component for each element
    fringe_data = np.mean(nodal_displacement_array[shell_elements_nodes_indices], axis=1)
    
    # Create colormap for the scalar values of the displacement component
    scalar_to_rgba_map = ScalarMappable(cmap=colormap)
    scalar_to_rgba_map.set_array(fringe_data)
    
    # If no color limits are provided, set them to the minimum and maximum values of the nodal displacement component
    if clim is None:
        vmin, vmax = np.amin(nodal_displacement_array), np.amax(nodal_displacement_array)
    # If color limits are provided, set them to the provided values
    else:
        vmin, vmax = clim
    scalar_to_rgba_map.set_clim(vmin=vmin, vmax=vmax)
        
    # Map scalar values to RGBA values and set face color of Poly3DCollection
    rgba_array = scalar_to_rgba_map.to_rgba(fringe_data)
    pc.set_facecolor([(rgb[0], rgb[1], rgb[2]) for rgb in rgba_array])
    
    # If axes object is not provided, create a new figure and axes
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
    # If axes object is provided, get figure object from provided axes object
    else:
        fig = axes.get_figure()
    
    # Add polygons to the plot
    axes.add_collection3d(pc)
    
    # Set axes label
    axes.set_xlabel(f"$x,\,\mathrm{{{length_unit}}}$")
    axes.set_ylabel(f"$y,\,\mathrm{{{length_unit}}}$")
    axes.set_zlabel(f"$z,\,\mathrm{{{length_unit}}}$")
    
    # Set axes limits
    x_coords = vertices[..., 0].ravel()
    y_coords = vertices[..., 1].ravel()
    z_coords = vertices[..., 2].ravel()
    axes.set_xlim(x_coords.min(), x_coords.max())
    axes.set_ylim(y_coords.min(), y_coords.max())
    axes.set_zlim(z_coords.min(), z_coords.max())
    
    # Set aspect ratio of the axes
    axes.set_aspect('equal', 'box')#set_box_aspect([ub - lb for lb, ub in (getattr(axes, f"get_{a}lim")() for a in 'xyz')])
    
    # Return axes object
    return fig, axes, scalar_to_rgba_map


def plot_eigenvector(op2: OP2, subcase_id: Union[int, tuple], axes: Axes3D = None, eigenvector_number: int = 1,
                     displacement_component: str = 'magnitude', unit_scale_factor: float = 1.,
                     displacement_amplification_factor: float = 1., colormap: str = 'rainbow_PuRd',
                     clim: Union[list, ndarray] = None, length_unit: str = 'm') -> Tuple[Figure, Axes3D, Colorbar]:
    """
    Plot one of the eigenvectors included in the input OP2 object.

    Parameters
    ----------
    op2: OP2
        pyNastran object created reading an op2 file with the load_geometry option set to True
    subcase_id: int, tuple
        key of the eigenvectors' dictionary in the OP2 object corresponding to the selected subcase
    axes: Axes3D
        object of the axes where the eigenvector will be plotted
    eigenvector_number: int
        number of the eigenvector to be plotted
    displacement_component: str
        name of the displacement component used for the colorbar
    unit_scale_factor: float
        scale factor for unit conversion
    displacement_amplification_factor: float
        scale factor applied to the displacements
    colormap: str
        name of the colormap used for the displacement colorbar
    clim: Union[list, ndarray]
        colorbar values limits
    length_unit: str
        measurement unit of coordinates and displacements, used in the label of axes and colormap

    Returns
    -------
    fig: Figure
        figure object
    ax: Axes3D
        axes object
    cbar: Colorbar
        colorbar object
    """
    
    # Choose eigenvectors as displacement data
    displacement_data = op2.eigenvectors[subcase_id].data[eigenvector_number - 1, :, :]  # convert to 0-based index
    
    # Call plotting function
    fig, ax, m = plot_displacements(op2=op2, displacement_data=displacement_data.copy(), axes=axes,
                                    displacement_component=displacement_component, displacement_unit_scale_factor=1.,
                                    coordinate_unit_scale_factor=unit_scale_factor,
                                    displacement_amplification_factor=displacement_amplification_factor, colormap=colormap,
                                    clim=clim, length_unit=length_unit)
    
    # Define dictionary for colorbar label
    cbar_label_dict = {
        'tx': "Nondimensional $u_x$",
        'ty': "Nondimensional $u_y$",
        'tz': "Nondimensional $u_z$",
        'rx': "Nondimensional $\\theta_x$",
        'ry': "Nondimensional $\\theta_y$",
        'rz': "Nondimensional $\\theta_z$",
        'magnitude': "Nondimensional $\|u\|$"}
    
    # Add colorbar
    cbar = fig.colorbar(mappable=m, ax=ax, label=cbar_label_dict[displacement_component])
    
    # If axes object is not provided, set whitespace to 0
    if axes is None:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # Return axes object
    return fig, ax, cbar


def plot_deformation(op2: OP2, subcase_id: Union[int, tuple] = 1, axes: Axes3D = None, load_step: int = 0,
                     displacement_component: str = 'magnitude', unit_scale_factor: float = 1.,
                     displacement_amplification_factor:float = 1., colormap: str = 'rainbow_PuRd',
                     clim: Union[list, ndarray] = None, length_unit: str = 'm', angle_unit: str = 'rad') -> Tuple[Figure, Axes3D, Colorbar]:
    """
    Plot the static deformation of the input OP2 object.

    Parameters
    ----------
    op2: OP2
        pyNastran object created reading an op2 file with the load_geometry option set to True
    subcase_id: int, tuple
        key of the displacements' dictionary in the OP2 object corresponding to the selected subcase
    axes: Axes3D
        object of the axes where the static deformation will be plotted
    load_step: int
        integer representing the load step of a nonlinear or a time-dependent analysis, default is 0 so that last step
        is chosen
    displacement_component: str
        name of the displacement component used for the colorbar
    unit_scale_factor: float
        scale factor for unit conversion
    displacement_amplification_factor: float
        scale factor applied to the displacements
    colormap: str
        name of the colormap used for the displacement colorbar
    clim: Union[list, ndarray]:
        colorbar values limits
    length_unit: str
        measurement unit of coordinates and displacements, used in the label of axes and colormap
    angle_unit: str
        measurement unit of angles, used in the label of the colorbar when the displacement component is a rotation
    
    Returns
    -------
    fig: Figure
        figure object
    ax: Axes3D
        axes object
    cbar: Colorbar
        colorbar object
    """
    
    # Choose static displacements as displacement data
    displacement_data = op2.displacements[subcase_id].data[load_step - 1, :, :]  # convert to 0-based index
    
    # Call plotting function
    fig, ax, m = plot_displacements(op2=op2, displacement_data=displacement_data.copy(), axes=axes,
                                    displacement_component=displacement_component, displacement_unit_scale_factor=unit_scale_factor,
                                    coordinate_unit_scale_factor=unit_scale_factor,
                                    displacement_amplification_factor=displacement_amplification_factor, colormap=colormap,
                                    clim=clim, length_unit=length_unit, angle_unit=angle_unit)
    
    # Define dictionary for colorbar label
    cbar_label_dict = {
        'tx': f"$u_x$, {length_unit}",
        'ty': f"$u_y$, {length_unit}",
        'tz': f"$u_z$, {length_unit}",
        'rx': f"$\\theta_x,\,\mathrm{{{angle_unit}}}$",
        'ry': f"$\\theta_y,\,\mathrm{{{angle_unit}}}$",
        'rz': f"$\\theta_z,\,\mathrm{{{angle_unit}}}$",
        'magnitude': f"\|u\|, {length_unit}"}
    
    # Add colorbar
    cbar = fig.colorbar(mappable=m, ax=ax, label=cbar_label_dict[displacement_component])
    
    # If axes object is not provided, set whitespace to 0
    if axes is None:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # Return axes object
    return fig, ax, cbar


def plot_max_displacement_node(axes: Axes3D, op2: OP2, subcase_id: int = 2, unit_scale_factor: float = 1., displacement_amplification_factor: float = 1.):
    """
    Plot the node where the maximum displacement occurs in a critical buckling mode.
    
    Parameters
    ----------
    axes: Axes3D
        object of the axes where the node will be plotted
    op2: OP2
        object of the op2 file containing the results of a SOL 105 analysis
    subcase_id: int
        integer with the subcase id of the eigenvector to be plotted
    unit_scale_factor: float
        float with the scale factor for the units
    displacement_amplification_factor: float
        float with the amplification factor for the displacements
    
    Returns
    -------
    max_displacement_node_id: int
        integer with the node id where the maximum displacement occurs
    """

    # Find node where max displacement occurs
    max_displacement_index = np.argmax(
        np.linalg.norm(op2.eigenvectors[subcase_id].data[0, :, 0:3], axis=1))  # find index of max displacement magnitude
    max_displacement_node_id = op2.eigenvectors[subcase_id].node_gridtype[
        max_displacement_index, 0]

    # Plot node where maximum displacement occurs
    max_displacement_node_xyz = op2.nodes[max_displacement_node_id].xyz*unit_scale_factor + op2.eigenvectors[
        subcase_id].data[0, max_displacement_index, 0:3]*displacement_amplification_factor  # add displacement to node position and convert to m
    axes.plot(max_displacement_node_xyz[0], max_displacement_node_xyz[1], max_displacement_node_xyz[2], 'x',
              label=f"Node {max_displacement_node_id:d} (max displacement)", zorder=4)
    
    # Return node id
    return max_displacement_node_id
