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


def plot_2d_load_displacements_stability(axes: Axes, displacements: ndarray, loads: ndarray, eigenvalues: ndarray, marker: str, color: str):
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
    """
    
    # Group the equilibrium points into segments with a constant number of negative eigenvalues
    change_negative_eigenvalues_indices = np.where(np.diff((eigenvalues < 0).sum(axis=0)))[0] + 1  # find the indices where the number of negative eigenvalues changes
    segments_indices_list = [list(range(i, j)) for i, j in zip(
        [0] + list(change_negative_eigenvalues_indices), list(change_negative_eigenvalues_indices) + [eigenvalues.shape[1]])]  # create a list of arrays including the indices of each segment
    
    # Count the number of negative eigenvalues for each segment
    no_neg_eigvals = [np.sum(eigenvalues[:, segment[0]] < 0, axis=0) for segment in segments_indices_list]
    
    # Create a list of arrays with the plotting indices for each segment
    plot_indices_list = [indices + [indices[-1] + 1] for indices in segments_indices_list[:-1]]  # the first point of the next segment is added to each segment to make the path appear continuous
    plot_indices_list.append(segments_indices_list[-1])  # add the last segment to the plotting list
    
    # Plot the first segment
    actual_neg_eigenvalues = no_neg_eigvals[0]  # get the number of negative eigenvalues at the first equilibrium point
    
    # Plot with a solid line if segment is stable
    if actual_neg_eigenvalues == 0:
        axes.plot(displacements[plot_indices_list[0]],
                  loads[plot_indices_list[0]], marker + '-', color=color)
    
    # Plot with a dashed red line if segment is unstable
    else:
        axes.plot(displacements[plot_indices_list[0]],
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
            axes.plot(displacements[plot_indices_list[count + 1]],
                      loads[plot_indices_list[count + 1]], marker + '-', color=color)
            
        # Plot with a dashed red line if segment is unstable
        else:
            axes.plot(displacements[plot_indices_list[count + 1]],
                      loads[plot_indices_list[count + 1]], marker + '--', color=UNSTABLE_COLOR)


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
    
    # Divide the equilibrium points into segments with a different number of negative eigenvalues
    num_neg_eigval_changes = np.where(np.diff((eigenvalues < 0).sum(axis=0)))[0] + 1  # find the indices where the number of negative eigenvalues changes
    index_segments = [list(range(i, j)) for i, j in zip([0] + list(num_neg_eigval_changes), list(num_neg_eigval_changes) + [eigenvalues.shape[1]])]  # create a list of indices for each segment with a different number of negative eigenvalues
    plot_index_segments = [indices + [indices[-1] + 1] for indices in index_segments[:-1]]  # add the first point of the next segment to the last point of each segment for plotting purposes, so that the overall path appears continuous
    plot_index_segments.append(index_segments[-1])  # add the last segment
    num_neg_eigvals = [np.sum(eigenvalues[:, segment[0]] < 0, axis=0) for segment in index_segments]  # count the number of negative eigenvalues for each segment
    
    # Plot the first segment
    actual_neg_eigenvalues = num_neg_eigvals[0]  # get the number of negative eigenvalues at the first equilibrium point
    if actual_neg_eigenvalues == 0:  # plot with a solid line if segment is stable
        axes.plot3D(displacements1[plot_index_segments[0]],
                    displacements2[plot_index_segments[0]],
                    loads[plot_index_segments[0]], marker + '-', color=color)
    else:  # plot with a dashed red line if segment is unstable
        axes.plot3D(displacements1[plot_index_segments[0]],
                    displacements2[plot_index_segments[0]],
                    loads[plot_index_segments[0]], marker + '--', color=UNSTABLE_COLOR)
    
    # Plot the remaining segments
    lowest_eigenvalue_row = eigenvalues[0]  # get the lowest eigenvalue at each equilibrium point
    for count, current_segment_indices in enumerate(index_segments[1:]):  # loop through the remaining segments
        last_segment_indices = index_segments[count]
        delta_num_neg_eigvals = num_neg_eigvals[count + 1] - num_neg_eigvals[count]  # get the change in the number of negative eigenvalues
        lowest_new_eigenvalue = lowest_eigenvalue_row[current_segment_indices[0]]  # get the lowest eigenvalue at the first equilibrium point of the segment
        predicted_eigenvalue = lowest_eigenvalue_row[last_segment_indices[-1]] + np.diff(lowest_eigenvalue_row[last_segment_indices[-1] - 1:last_segment_indices[-1] + 1])  # predict the lowest eigenvalue at the first equilibrium point of the next segment using the last two points of the previous segment
        eigenvalue_abs_diffs = np.abs(np.diff(lowest_eigenvalue_row[last_segment_indices + current_segment_indices]))  # calculate the absolute differences in the lowest eigenvalue during the previous and the current segment
        tolerance = np.mean(eigenvalue_abs_diffs) + np.std(eigenvalue_abs_diffs)*7  # set the tolerance as average change plus seven times the standard deviation
        if np.abs(lowest_new_eigenvalue - predicted_eigenvalue) <= tolerance:  # if the lowest eigenvalue at the first equilibrium point of the segment is close to the predicted value update the actual number of negative eigenvalues
            actual_neg_eigenvalues += delta_num_neg_eigvals
        if actual_neg_eigenvalues == 0:  # plot with a solid line if segment is stable
            axes.plot3D(displacements1[plot_index_segments[count + 1]],
                        displacements2[plot_index_segments[count + 1]],
                        loads[plot_index_segments[count + 1]], marker + '-', color=color)
        else:  # plot with a dashed red line if segment is unstable
            axes.plot3D(displacements1[plot_index_segments[count + 1]],
                        displacements2[plot_index_segments[count + 1]],
                        loads[plot_index_segments[count + 1]], marker + '--', color=UNSTABLE_COLOR)
            

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
    

def plot_displacements(op2: OP2, displacement_data: ndarray, displacement_component: str = "magnitude",
                       displacement_unit_scale_factor: float = 1., coordinate_unit_scale_factor:float =1.,
                       displacement_amplification_factor: float = 1., colormap: str = "rainbow_PuRd", clim: Union[list, ndarray] = None,
                       length_unit: str = "mm") -> Tuple[Figure, Axes3D, ScalarMappable]:
    """
    Plot the deformed shape coloured by displacement magnitude based on input OP2 object and displacement data.

    Parameters
    ----------
    op2: OP2
        pyNastran object created reading an op2 file with the load_geometry option set to True
    displacement_data: ndarray
        array with displacement data to plot
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

    Returns
    -------
    fig: Figure
        object of the plotted figure
    ax: Axes3D
        object of the plot's axes
    m: ScalarMappable
        mappable object for the colorbar
    """
    # Extract mapping from node ids to indexes and undeformed coordinates
    node_id_to_index = {}
    undeformed_coordinates_array = np.empty((len(op2.nodes), 3))
    for node_index, (node_id, node) in enumerate(op2.nodes.items()):
        node_id_to_index[node_id] = node_index
        undeformed_coordinates_array[node_index] = node.xyz
    # Apply scalings and find deformed coordinates
    undeformed_coordinates_array *= coordinate_unit_scale_factor  # apply unit conversion to undeformed coordinates
    displacement_data[:, :3] = displacement_data[:, :3] * displacement_unit_scale_factor  # apply unit conversion to displacements
    deformed_coordinates_array = undeformed_coordinates_array + displacement_data[:, :3] * displacement_amplification_factor  # apply amplification factor to displacements and calculate deformed coordinates
    # Find node indexes for each element
    element_node_indexes = np.array([[node_id_to_index[node_id] for node_id in element.node_ids] for element in op2.elements.values()])
    # Find vertices coordinates for Poly3DCollection
    vertices = deformed_coordinates_array[element_node_indexes]
    # Create Poly3DCollection
    pc = Poly3DCollection(vertices, linewidths=.05, edgecolor='k')
    # Handle displacement component and calculate nodal displacement array of interest
    if displacement_component == 'magnitude':
        nodal_displacement_array = np.linalg.norm(displacement_data[:, :3], axis=1)  # calculate displacement magnitude
    else:
        component_dict = {'tx': 0, 'ty': 1, 'tz': 2, 'rx': 3, 'ry': 4, 'rz': 5}
        nodal_displacement_array = displacement_data[:, component_dict[displacement_component]]  # select displacement component
    # Calculate average displacement for each element
    fringe_data = np.mean(nodal_displacement_array[element_node_indexes], axis=1)
    # Create colormap for the displacement magnitude
    m = ScalarMappable(cmap=colormap)
    m.set_array(fringe_data)
    # Set colormap min and max values and displacement values to colors
    if clim is None:
        m.set_clim(vmin=np.amin(nodal_displacement_array), vmax=np.amax(nodal_displacement_array))
    else:
        m.set_clim(vmin=clim[0], vmax=clim[1])
    rgba_array = m.to_rgba(fringe_data)
    # Color the elements' face by the average displacement of interest
    pc.set_facecolor([(rgb[0], rgb[1], rgb[2]) for rgb in rgba_array])
    # Initialize figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Add polygons to the plot
    ax.add_collection3d(pc)
    # Set axes label
    ax.set_xlabel(f"$x,\,\mathrm{{{length_unit}}}$")
    ax.set_ylabel(f"$y,\,\mathrm{{{length_unit}}}$")
    ax.set_zlabel(f"$z,\,\mathrm{{{length_unit}}}$")
    # Set axes limits
    x_coords = vertices[..., 0].ravel()
    y_coords = vertices[..., 1].ravel()
    z_coords = vertices[..., 2].ravel()
    ax.set_xlim(x_coords.min(), x_coords.max())
    ax.set_ylim(y_coords.min(), y_coords.max())
    ax.set_zlim(z_coords.min(), z_coords.max())
    # Set aspect ratio of the axes
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in 'xyz')])
    # Return axes object
    return fig, ax, m


def plot_buckling_mode(op2: OP2, subcase_id: Union[int, tuple], mode_number: int = 1, displacement_component: str = 'magnitude',
                       unit_scale_factor: float = 1., displacement_amplification_factor: float = 200., colormap: str = 'rainbow_PuRd',
                       clim: Union[list, ndarray] = None, length_unit: str = 'mm') -> Tuple[Figure, Axes3D, Colorbar]:
    """
    Plot the buckling shape using the eigenvectors of the input OP2 object.

    Parameters
    ----------
    op2: OP2
        pyNastran object created reading an op2 file with the load_geometry option set to True
    subcase_id: int, tuple
        key of the eigenvectors' dictionary in the OP2 object corresponding to the selected subcase
    mode_number: int
        number of the buckling mode to be plotted
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
    displacement_data = op2.eigenvectors[subcase_id].data[mode_number - 1, :, :]
    # Call plotting function
    fig, ax, m = plot_displacements(op2=op2, displacement_data=displacement_data.copy(),
                                    displacement_component=displacement_component, displacement_unit_scale_factor=1.,
                                    coordinate_unit_scale_factor=unit_scale_factor,
                                    displacement_amplification_factor=displacement_amplification_factor, colormap=colormap,
                                    clim=clim, length_unit=length_unit)
    # Add colorbar
    label_dict = {
        'tx': "Nondimensional $u_x$",
        'ty': "Nondimensional $u_y$",
        'tz': "Nondimensional $u_z$",
        'rx': "Nondimensional $\\theta_x$",
        'ry': "Nondimensional $\\theta_y$",
        'rz': "Nondimensional $\\theta_z$",
        'magnitude': "Nondimensional $\|u\|$"}
    cbar = fig.colorbar(mappable=m, ax=ax, label=label_dict[displacement_component])
    # Set whitespace to 0
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # Return axes object
    return fig, ax, cbar


def plot_static_deformation(op2: OP2, subcase_id: Union[int, tuple] = 1, load_step: int = 0,
                            displacement_component: str = "magnitude", unit_scale_factor: float = 1.,
                            displacement_amplification_factor:float = 1., colormap: str = "rainbow_PuRd",
                            clim: Union[list, ndarray] = None, length_unit: str = "mm") -> Tuple[Figure, Axes3D, Colorbar]:
    """
    Plot the buckling shape using the eigenvectors of the input OP2 object.

    Parameters
    ----------
    op2: OP2
        pyNastran object created reading an op2 file with the load_geometry option set to True
    subcase_id: int, tuple
        key of the displacements' dictionary in the OP2 object corresponding to the selected subcase
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
    displacement_data = op2.displacements[subcase_id].data[load_step - 1, :, :]
    # Call plotting function
    fig, ax, m = plot_displacements(op2=op2, displacement_data=displacement_data.copy(),
                                    displacement_component=displacement_component, displacement_unit_scale_factor=unit_scale_factor,
                                    coordinate_unit_scale_factor=unit_scale_factor,
                                    displacement_amplification_factor=displacement_amplification_factor, colormap=colormap,
                                    clim=clim, length_unit=length_unit)
    # Add colorbar
    label_dict = {
        'tx': f"$u_x$, {length_unit}",
        'ty': f"$u_y$, {length_unit}",
        'tz': f"$u_z$, {length_unit}",
        'rx': "$\\theta_x,\,\mathrm{rad}$",
        'ry': "$\\theta_y,\,\mathrm{rad}$",
        'rz': "$\\theta_z,\,\mathrm{rad}$",
        'magnitude': f"\|u\|, {length_unit}"}
    cbar = fig.colorbar(mappable=m, ax=ax, label=label_dict[displacement_component])
    # Set whitespace to 0
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
