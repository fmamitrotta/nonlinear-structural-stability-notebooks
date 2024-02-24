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
    negative_eigenvalues_mask = (eigenvalues < 0).any(axis=0)  # mask to identify unstable segments
    unstable_segments = []  # list to store indices of unstable segments
    stable_segments = []  # list to store indices of stable segments
    # Loop through the negative_eigenvalues_mask to identify and plot segments
    for i, is_negative in enumerate(negative_eigenvalues_mask):
        if is_negative:
            if stable_segments:
                # Plot the stable segment if there was one before
                stable_segments.append(i)  # make the stable segment finish at the first point of the unstable segment
                axes.plot(displacements[stable_segments], loads[stable_segments], marker + '-', color=color)
                stable_segments = []  # reset the stable segment indices
            unstable_segments.append(i)  # add the current index to the unstable segment, this will overwrite the blue point with a red one
        else:
            if unstable_segments:
                # Plot the unstable segment if there was one before
                unstable_segments.append(i)  # make the unstable segment finish at the first point of the stable segment
                axes.plot(displacements[unstable_segments], loads[unstable_segments],  marker + '--',
                            color=UNSTABLE_COLOR)
                unstable_segments = []  # reset the unstable segment indices
            stable_segments.append(i)  # add the current index to the stable segment, this will overwrite the red point with a blue one
    # Plot the remaining segments if any
    if stable_segments:
        axes.plot(displacements[stable_segments], loads[stable_segments], marker + '-', color=color)
    if unstable_segments:
        axes.plot(displacements[unstable_segments], loads[unstable_segments], marker + "--", color=UNSTABLE_COLOR)


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
    negative_eigenvalues_mask = (eigenvalues < 0).any(axis=0)  # mask to identify unstable segments
    unstable_segments = []  # list to store indices of unstable segments
    stable_segments = []  # list to store indices of stable segments
    # Loop through the negative_eigenvalues_mask to identify and plot segments
    for i, is_negative in enumerate(negative_eigenvalues_mask):
        if is_negative:
            if stable_segments:
                # Plot the stable segment if there was one before
                stable_segments.append(i)  # make the stable segment finish at the first point of the unstable segment
                axes.plot3D(displacements1[stable_segments], displacements2[stable_segments], loads[stable_segments], marker + '-', color=color)
                stable_segments = []  # reset the stable segment indices
            unstable_segments.append(i)  # add the current index to the unstable segment, this will overwrite the blue point with a red one
        else:
            if unstable_segments:
                # Plot the unstable segment if there was one before
                unstable_segments.append(i)  # make the unstable segment finish at the first point of the stable segment
                axes.plot3D(displacements1[unstable_segments], displacements2[unstable_segments], loads[unstable_segments],  marker + '--',
                            color=UNSTABLE_COLOR)
                unstable_segments = []  # reset the unstable segment indices
            stable_segments.append(i)  # add the current index to the stable segment, this will overwrite the red point with a blue one
    # Plot the remaining segments if any
    if stable_segments:
        axes.plot3D(displacements1[stable_segments], displacements2[stable_segments], loads[stable_segments], marker + '-', color=color)
    if unstable_segments:
        axes.plot3D(displacements1[unstable_segments], displacements2[unstable_segments], loads[unstable_segments], marker + "--", color=UNSTABLE_COLOR)
