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
import numpy as np  # import numpy for numerical operations

plt.rc('axes', prop_cycle=plt.cycler('color', list(tc.tol_cset('bright'))))  # set default color cycle to TOL bright
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']  # list with default color cycle
UNSTABLE_COLOR = COLORS[1]  # set red color for unstable segments


def plot_2d_load_displacements_stability(eigenvalues, axes, displacements, loads, marker, color):
    """
    Plot a load-displacement diagram highlighting the stability of the equilibrium points.

    Parameters
    ----------
    eigenvalues : ndarray
        numpy array with the eigenvalues of the tangent stiffness matrix for each equilibrium point
    axes: Axes
        object of the axes where the load-displacement diagram will be plotted
    displacements: ndarray
        numpy array of the displacements that will be plotted on the x-axis
    loads: ndarray
        numpy array of the loads that will be plotted on the y-axis
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
        axes.plot(displacements[plot_index_segments[0]],
                  loads[plot_index_segments[0]], marker + '-', color=color)
    else:  # plot with a dashed red line if segment is unstable
        axes.plot(displacements[plot_index_segments[0]],
                  loads[plot_index_segments[0]], marker + '--', color=UNSTABLE_COLOR)
    # Plot the remaining segments
    lowest_eigenvalue_row = eigenvalues[0]  # get the lowest eigenvalue at each equilibrium point
    for count, segment in enumerate(index_segments[1:]):  # loop through the remaining segments
        delta_num_neg_eigvals = num_neg_eigvals[count + 1] - num_neg_eigvals[count]  # get the change in the number of negative eigenvalues
        lowest_new_eigenvalue = lowest_eigenvalue_row[segment[0]]  # get the lowest eigenvalue at the first equilibrium point of the segment
        predicted_eigenvalue = lowest_eigenvalue_row[index_segments[count][-1]] + np.diff(lowest_eigenvalue_row[index_segments[count][-2:]])  # predict the lowest eigenvalue at the first equilibrium point of the next segment using the last two points of the previous segment
        tolerance = np.std(np.abs(np.diff(lowest_eigenvalue_row[index_segments[count] + index_segments[count + 1]])))*3  # set the tolerance as three times the standard deviation of the changes in the lowest eigenvalue during the previous and the current segment
        if np.abs(lowest_new_eigenvalue - predicted_eigenvalue) <= tolerance:  # if the lowest eigenvalue at the first equilibrium point of the segment is close to the predicted value update the actual number of negative eigenvalues
            actual_neg_eigenvalues += delta_num_neg_eigvals
        if actual_neg_eigenvalues == 0:  # plot with a solid line if segment is stable
            axes.plot(displacements[plot_index_segments[count + 1]],
                      loads[plot_index_segments[count + 1]], marker + '-', color=color)
        else:  # plot with a dashed red line if segment is unstable
            axes.plot(displacements[plot_index_segments[count + 1]],
                      loads[plot_index_segments[count + 1]], marker + '--', color=UNSTABLE_COLOR)


def plot_3d_load_displacements_stability(eigenvalues, axes, displacements1, displacements2, loads, marker, color):
    """
    Plot a 3D load-displacement diagram highlighting the stability of the equilibrium points.

    Parameters
    ----------
    eigenvalues : ndarray
        numpy array with the eigenvalues of the tangent stiffness matrix for each equilibrium point
    axes: Axes3D
        object of the axes where the load-displacement diagram will be plotted
    displacements1: ndarray
        numpy array of the displacements that will be plotted on the x-axis
    displacements2: ndarray
        numpy array of the displacements that will be plotted on the y-axis
    loads: ndarray
        numpy array of the loads that will be plotted on the z-axis
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
    for count, segment in enumerate(index_segments[1:]):  # loop through the remaining segments
        delta_num_neg_eigvals = num_neg_eigvals[count + 1] - num_neg_eigvals[count]  # get the change in the number of negative eigenvalues
        lowest_new_eigenvalue = lowest_eigenvalue_row[segment[0]]  # get the lowest eigenvalue at the first equilibrium point of the segment
        predicted_eigenvalue = lowest_eigenvalue_row[index_segments[count][-1]] + np.diff(lowest_eigenvalue_row[index_segments[count][-2:]])  # predict the lowest eigenvalue at the first equilibrium point of the next segment using the last two points of the previous segment
        tolerance = np.std(np.abs(np.diff(lowest_eigenvalue_row[index_segments[count] + index_segments[count + 1]])))*3  # set the tolerance as three times the standard deviation of the changes in the lowest eigenvalue during the previous and the current segment
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
