"""
This file is part of the GitHub repository
nonlinear-structural-stability-notebooks, created by Francesco M. A.
Mitrotta.
Copyright (C) 2025 Francesco Mario Antonio Mitrotta

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from itertools import compress
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mphys import Multipoint
import openmdao.api as om
from resources.NastranBuilder import NastranBuilder
from mphys.scenario_structural import ScenarioStructural
from resources.STW import wingGeometry


def find_element_ids(layout, part_names):
    """
    Find the element ids of specified parts in a STW layout.

    Parameters
    ----------
    layout : layout object
        The layout object created with pyLayout.
    part_names : list of str
        The names of the parts to find element ids for.

    Returns
    -------
    dict
        A dictionary mapping part names to their element ids.
    """
    # Find number of structural segments
    num_segments = len(layout.elemTopo.lIndex)

    # Initialize list of element ids for each segment and element counter
    elem_ids = [None] * num_segments
    elem = 1

    # Loop over all segments and find the element ids
    for i in range(num_segments):
        local = layout.elemTopo.lIndex[i]  # array with local node indices
        num_rows = local.shape[0] - 1  # number of element rows
        num_cols = local.shape[1] - 1  # number of element columns
        elem_ids[i] = np.arange(elem, elem + num_rows * num_cols)
        elem += num_rows * num_cols  # increment element counter

    # Find the element ids of the specified parts
    part_element_ids = {name: None for name in part_names}
    descriptions = np.array(layout.faceDescript)
    for i, name in enumerate(part_names):
        mask = np.char.find(descriptions, name) >= 0
        part_element_ids[name] = np.concatenate(list(compress(elem_ids, mask)))

    # Return the element ids of the specified parts
    return part_element_ids


def plot_elements_normals(bdf, elem_dict):
    """
    Plot the elements and their normal vectors for the specified parts in a STW layout.
    
    Parameters
    ----------
    bdf : BDF object
        The BDF object containing the finite element model.
    elem_dict : dict
        A dictionary mapping part names to their element ids, as returned by the find_element_ids function.
    """
    # Iterate through the structural parts
    for part in elem_dict.keys():
        # Print the name of the part being plotted
        print(part)
        # Initialize arrays with coordinates and vector components
        nodes = np.empty((len(elem_dict[part]), 4, 3))
        centroids = np.empty((len(elem_dict[part]), 3))
        normals = np.empty((len(elem_dict[part]), 3))
        # Iterate through the elements of the bdf input
        for count, eid in enumerate(elem_dict[part]):
            nodes[count] = bdf.elements[
                eid
            ].get_node_positions()  # get the array with the coordinates of the nodes belonging to the element
            _, centroid, normal = bdf.elements[
                eid
            ].AreaCentroidNormal()  # find the coordinates of the centroid and the components of the normal vector of the element
            centroids[count] = (
                centroid  # add coordinates of centroid to appropriate array
            )
            normals[count] = (
                normal  # add components of normal vector to appropriate array
            )
        # Define limit for the axes range
        bounds = np.array(
            [
                [np.amin(nodes[:, :, 0]), np.amax(nodes[:, :, 0])],
                [np.amin(nodes[:, :, 1]), np.amax(nodes[:, :, 1])],
                [np.amin(nodes[:, :, 2]), np.amax(nodes[:, :, 2])],
            ]
        )
        # Define list to set the aspect ratio of the plot
        aspect_ratio = [x[1] - x[0] for x in list(bounds)]
        # Create figure and axes
        fig = plt.figure(tight_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        # Plot elements
        pc = Poly3DCollection(nodes, linewidths=0.5, alpha=0.5)
        pc.set_edgecolor("k")
        ax.add_collection3d(pc)
        # Plot normal vectors
        ax.quiver(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
            length=0.1,
            color="blue",
            arrow_length_ratio=0.5,
        )
        # Set axes label
        ax.set_xlabel("$x$, m")
        ax.set_ylabel("$y$, m")
        ax.set_zlabel("$z$, m")
        # Set aspect ratio
        ax.set_box_aspect(aspect_ratio)
        # Set axes limits
        ax.set_xlim(bounds[0, :])
        ax.set_ylim(bounds[1, :])
        ax.set_zlim(bounds[2, :])
        # Adjust number of ticks of x and z axes
        ax.locator_params(axis="x", nbins=3)
        ax.locator_params(axis="z", nbins=2)
        # Adjust ticks label of y- and z-axis
        ax.tick_params(axis="y", which="major", pad=25)
        ax.tick_params(axis="z", which="major", pad=6)
        # Adjust axis label y and z axes
        ax.yaxis.labelpad = 70
        ax.zaxis.labelpad = 10
        # Show plot
        plt.show()


def find_intersection_nodes(layout, part_names):
    """
    Find the intersection nodes' ids of specified parts in a STW layout.

    Parameters
    ----------
    layout : layout object
        The layout object created with pyLayout.
    part_names : list of str
        The names of the parts to find element ids for.

    Returns
    -------
    dict
        A dictionary mapping part names to their element ids.
    """
    # Store face descriptions in a numpy array
    descriptions = np.array(layout.faceDescript)

    # Find the nodes' ids of each part
    part_nodes_ids = [np.array([])] * len(part_names)
    for i, name in enumerate(part_names):
        mask = np.char.find(descriptions, name) >= 0
        nodes_lindex = list(compress(layout.elemTopo.lIndex, mask))
        part_nodes_ids[i] = np.unique(
            np.concatenate([arr.flatten() for arr in nodes_lindex]) + 1
        )

    # Find the intersection nodes' ids
    intersection_nodes_ids = part_nodes_ids[0]
    for i in range(1, len(part_names)):
        intersection_nodes_ids = np.intersect1d(
            intersection_nodes_ids, part_nodes_ids[i], assume_unique=True
        )

    # Return the intersection nodes' ids
    return intersection_nodes_ids


def find_tip_nodes(layout, bdf):
    """
    Find the ids and coordinates of the nodes at the tip rib of a STW layout.
    
    Parameters
    ----------
    layout : layout object
        The layout object created with pyLayout.
    bdf : BDF object
        The BDF object containing the finite element model.
    
    Returns
    -------
    tip_nodes_ids : array-like
        The IDs of the tip nodes.
    tip_nodes_xyz_array : (N, 3) ndarray
        The coordinates of the tip nodes.
    """
    # Find the ids of the intersection nodes between the upper skin, rib and spar
    intersection_nodes_ids = find_intersection_nodes(layout, ["U_SKIN", "RIB", "SPAR"])

    # Create a (N,3) array of coordinates for the intersection nodes
    intersection_nodes_xyz_array = np.vstack(
        [bdf.nodes[node_id].xyz for node_id in intersection_nodes_ids]
    )

    # Mask to find the nodes on the tip rib
    tip_nodes_mask = np.isclose(intersection_nodes_xyz_array[:, 1], layout.X[-1, 0, 1])

    # Return the ids and coordinates of the tip nodes
    return (
        intersection_nodes_ids[tip_nodes_mask],
        intersection_nodes_xyz_array[tip_nodes_mask],
    )
    

def calculate_tip_deflection(node_1_disp, node_2_disp):
    """
    Calculate the tip deflection as the average z-displacement of two nodes.
    
    Parameters
    ----------
    node_1_disp : (N, 3) ndarray
        The displacement of the first node.
    node_2_disp : (N, 3) ndarray
        The displacement of the second node.

    Returns
    -------
    delta_z_tip : (N,) ndarray
        The tip deflection.
    """
    node_1_disp = np.atleast_2d(node_1_disp)
    node_2_disp = np.atleast_2d(node_2_disp)
    delta_z_tip = (node_1_disp[:, 2] + node_2_disp[:, 2]) / 2
    return delta_z_tip


def calculate_tip_twist(node_1_coords, node_2_coords, node_1_disp, node_2_disp):
    """
    Calculate the tip twist angle as the difference between the angles of the line connecting two nodes before and after deformation.
    
    Parameters
    ----------
    node_1_coords : (3,) array-like
        The coordinates of the first node.
    node_2_coords : (3,) array-like
        The coordinates of the second node.
    node_1_disp : (N, 3) ndarray
        The displacement of the first node.
    node_2_disp : (N, 3) ndarray
        The displacement of the second node.
    
    Returns
    -------
    delta_theta_tip : (N,) ndarray
        The tip twist angle in radians.
    """
    # Ensure that the displacements are 2D arrays with shape (N, 3) for consistent indexing
    node_1_disp = np.atleast_2d(node_1_disp)
    node_2_disp = np.atleast_2d(node_2_disp)
    
    # Calculate the angle of the line connecting the two nodes before and after deformation
    # Δ𝜃 = tan−1( (𝑧2 + Δ𝑧2) − (𝑧1 + Δ𝑧1) / (𝑥1 + Δ𝑥1) − (𝑥2 + Δ𝑥2) ) − tan−1( (𝑧2 − 𝑧1) / (𝑥1 − 𝑥2) )
    delta_theta_tip = np.arctan(
        (
            (node_2_coords[2] + node_2_disp[:, 2])
            - (node_1_coords[2] + node_1_disp[:, 2])
        )
        / (
            (node_1_coords[0] + node_1_disp[:, 0])
            - (node_2_coords[0] + node_2_disp[:, 0])
        )
    ) - np.arctan(
        (node_2_coords[2] - node_1_coords[2]) / (node_1_coords[0] - node_2_coords[0])
    )
    
    # Return the tip twist angle in radians
    return delta_theta_tip


def apply_linearly_distributed_force(
    node_xyz_array, ratio, total_force, node_ids, bdf, set_id, direction_vector
):
    """
    Apply a linearly distributed force along the x-direction to a set of nodes.

    Parameters
    ----------
    node_xyz_array : (N, 3) ndarray
        Array of node coordinates.
    ratio : float
        The ratio F(x_first) / F(x_last). Use 1.0 for a uniform distribution.
    total_force : float
        The total force to be applied.
    node_ids : list of int
        The IDs of the nodes to apply the force to.
    bdf : BDF object
        The BDF object to apply the forces to.
    set_id : int
        The set ID of the Nastran FORCE cards to be created.
    direction_vector : (3,) ndarray
        The direction vector for the force application.
    """
    # Parameterise as F(t) = F_last * (ratio + (1 - ratio) * t),
    # where t = (x - x_first) / (x_last - x_first) ∈ [0, 1].
    #
    # By construction:
    #   F(t=0) = F_last * ratio  →  F_first / F_last = ratio  ✓
    #   F(t=1) = F_last          →  reference value at x_last
    #
    # Applying the total-force constraint Σ F(t_i) = total_force:
    #   F_last * [N * ratio + (1 - ratio) * Σ t_i] = total_force
    #   F_last = total_force / [N * ratio + (1 - ratio) * Σ t_i]
    #
    # When ratio = 1 the (1 - ratio) terms vanish and the denominator
    # reduces to N, giving the correct uniform value total_force / N —
    # no special-casing required.

    # Extract x coordinates
    # Note: this code assumes that the x-coordinates are unique and ordered,
    # which is indeed the case for how the mesh of the STW wingbox is generated
    # with pyLayout. If this is not the case, the code should be modified
    # to find the correct x_first and x_last values.
    x_coords = node_xyz_array[:, 0]
    x_first, x_last = x_coords[0], x_coords[-1]

    # Normalised positions in [0, 1]
    t = (x_coords - x_first) / (x_last - x_first)

    # Solve for the force at x_last, then reconstruct the distribution
    f_last = total_force / (len(t) * ratio + (1.0 - ratio) * np.sum(t))
    forces = f_last * (ratio + (1.0 - ratio) * t)

    # Apply forces to nodes in the BDF
    for node_id, nodal_force in zip(node_ids, forces):
        bdf.add_force(sid=set_id, node=node_id, mag=nodal_force, xyz=direction_vector)


def apply_distributed_tip_load(layout, bdf, ratio, total_force, set_id):
    """
    Apply a linearly distributed tip load to the upper and lower skin of a STW layout.
    
    Parameters
    ----------
    layout : layout object
        The layout object created with pyLayout.
    bdf : BDF object
        The BDF object to apply the forces to.
    ratio : float
        The ratio of the forces at the two ends.
    total_force : float
        The total force to be applied.
    set_id : int
        The set ID of the Nastran FORCE cards to be created.
    """
    # Iterate over upper and lower skin
    for skin_part in ["U_SKIN", "L_SKIN"]:
        # Find nodes at the intersection between ribs and skin
        intersection_nodes_ids = find_intersection_nodes(layout, ["RIB", skin_part])

        # Create a (N,3) array of coordinates for the intersection nodes
        intersection_nodes_xyz_array = np.vstack(
            [bdf.nodes[node_id].xyz for node_id in intersection_nodes_ids]
        )

        # Select only the nodes at the tip (y = max y)
        tip_nodes_mask = np.isclose(
            intersection_nodes_xyz_array[:, 1], layout.X[-1, 0, 1]
        )  # remember X has shape (numRibs, numSpars + num_stringers, 3)
        tip_node_ids = intersection_nodes_ids[tip_nodes_mask]
        tip_node_xyz_array = intersection_nodes_xyz_array[tip_nodes_mask]

        # Apply linearly distributed force along the x-axis to tip nodes
        apply_linearly_distributed_force(
            node_xyz_array=tip_node_xyz_array,
            ratio=ratio,
            total_force=total_force/2,  # split total force between upper and lower skin
            node_ids=tip_node_ids,
            bdf=bdf,
            set_id=set_id,
            direction_vector=np.array([0.0, 0.0, 1.0]),  # force in positive z-direction (upwards)
        )


class NastranDVComp(om.ExplicitComponent):
    """
    Component for grouping thickness design variables of the different
    structural parts into a global design vector.
    """

    def initialize(self):
        self.options.declare(
            "layout",
            default=None,
            desc="Layout object containing the structural layout",
            recordable=False,
        )

    def setup(self):
        self.layout = self.options["layout"]
        self.add_input("t_ribs", shape_by_conn=True, desc="Ribs thickness", units="m")
        self.add_input(
            "t_front_spar", shape_by_conn=True, desc="Front spar thickness", units="m"
        )
        self.add_input(
            "t_rear_spar", shape_by_conn=True, desc="Rear spar thickness", units="m"
        )
        self.add_input(
            "t_top_skin", shape_by_conn=True, desc="Top skin thickness", units="m"
        )
        self.add_input(
            "t_bottom_skin", shape_by_conn=True, desc="Bottom skin thickness", units="m"
        )
        self.add_input(
            "t_top_stiffeners",
            shape_by_conn=True,
            desc="Top stiffeners thickness",
            units="m",
        )
        self.add_input(
            "t_bottom_stiffeners",
            shape_by_conn=True,
            desc="Bottom stiffeners thickness",
            units="m",
        )
        self.add_output(
            "dv_struct",
            shape=len(self.layout.faceDescript),
            desc="Structural design vector",
        )

    def compute(self, inputs, outputs):
        # Initialize design vector
        dv_struct = np.empty(len(self.layout.faceDescript))
        descriptions = np.array(self.layout.faceDescript)

        # Ribs
        rib_indices = np.where(np.char.find(descriptions, "RIB") >= 0)[0]
        dv_struct[rib_indices] = np.repeat(inputs["t_ribs"], self.layout.nspars - 1)

        # Spars
        spar_indices = np.where(np.char.find(descriptions, "SPAR") >= 0)[0]
        dv_struct[spar_indices] = np.hstack(
            (inputs["t_front_spar"], inputs["t_rear_spar"])
        )

        # Top skin
        top_skin_indices = np.where(np.char.find(descriptions, "U_SKIN") >= 0)[0]
        dv_struct[top_skin_indices] = np.repeat(
            inputs["t_top_skin"], self.layout.nspars - 1
        )

        # Bottom skin
        bottom_skin_indices = np.where(np.char.find(descriptions, "L_SKIN") >= 0)[0]
        dv_struct[bottom_skin_indices] = np.repeat(
            inputs["t_bottom_skin"], self.layout.nspars - 1
        )

        # Top skin stiffeners
        top_stiffeners_indices = np.where(np.char.find(descriptions, "U_STRING") >= 0)[
            0
        ]
        dv_struct[top_stiffeners_indices] = np.repeat(
            inputs["t_top_stiffeners"], self.layout.nspars - 2
        )

        # Bottom skin stiffeners
        bottom_stiffeners_indices = np.where(
            np.char.find(descriptions, "L_STRING") >= 0
        )[0]
        dv_struct[bottom_stiffeners_indices] = np.repeat(
            inputs["t_bottom_stiffeners"], self.layout.nspars - 2
        )

        # Store design vector
        outputs["dv_struct"] = dv_struct


class NastranAnalysis(Multipoint):
    def initialize(self):
        self.options.declare("layout", default=None)
        self.options.declare("input_name_suffix", default="")
        self.options.declare("analysis_directory_path", default=".")
        self.options.declare("bdf", default=None)
        self.options.declare("run_flag", default=True)

    def setup(self):
        # ---------------#
        # Initialization #
        # ---------------#
        self.layout = self.options["layout"]
        self.input_name_suffix = self.options["input_name_suffix"]
        self.analysis_directory_path = self.options["analysis_directory_path"]
        self.bdf = self.options["bdf"]
        self.run_flag = self.options["run_flag"]

        # -----#
        # IVC #
        # -----#
        ivc = self.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
        ivc.add_output("yield_strength", val=wingGeometry.yieldStrength, units="Pa")

        # --------------------------#
        # Interpolation components #
        # --------------------------#
        # Function to add interpolation component
        def add_interp_comp(y_coords, part_name, y_cp_val):
            interp_comp = om.SplineComp(
                method="slinear",
                x_cp_val=wingGeometry.LESparCoords[:, 1],
                x_interp_val=y_coords,
            )
            interp_comp.add_spline(
                y_cp_name="t_cp",
                y_interp_name="t_interp",
                y_cp_val=y_cp_val,
                y_units="m",
            )
            self.add_subsystem(f"{part_name}_interp", interp_comp)
            self.connect(f"{part_name}_interp.t_interp", f"dv_comp.t_{part_name}")
        
        # Define ribs and panels y-coordinates for interpolation
        ribs_y_coords = self.layout.X[:, 0, 1]
        panels_y_coords = ribs_y_coords[:-1] + np.diff(ribs_y_coords) / 2
        
        # Define initial thickness vectors
        initial_panel_thickness_vector = wingGeometry.panelThickness * np.ones(3)
        initial_stiffener_thickness_vector = wingGeometry.stiffenerThickness * np.ones(3)

        # Ribs
        add_interp_comp(ribs_y_coords, "ribs", initial_panel_thickness_vector)

        # Spars and skin
        for part in ["front_spar", "rear_spar", "top_skin", "bottom_skin"]:
            add_interp_comp(panels_y_coords, part, initial_panel_thickness_vector)

        # Stiffeners
        for part in ["top_stiffeners", "bottom_stiffeners"]:
            add_interp_comp(panels_y_coords, part, initial_stiffener_thickness_vector)

        # -------------------------#
        # Design vector assembler #
        # -------------------------#
        self.add_subsystem("dv_comp", NastranDVComp(layout=self.layout))

        # ----------------------------#
        # Add builders and scenarios #
        # ----------------------------#
        # Nastran builder
        input_name = f"sol_{self.bdf.sol}_{self.input_name_suffix}"
        builder = NastranBuilder(
            analysis_directory_path=self.analysis_directory_path,
            input_name=input_name,
            bdf=self.bdf,
            run_flag=self.run_flag,
        )
        builder.initialize()

        # Structures only scenario
        scenario_name = f"sol_{self.bdf.sol}"
        self.mphys_add_scenario(
            scenario_name, ScenarioStructural(struct_builder=builder)
        )

        # Structural mesh
        self.add_subsystem(
            scenario_name + "_mesh", builder.get_mesh_coordinate_subsystem()
        )

        # Connections
        self.connect("yield_strength", scenario_name + ".yield_strength")
        self.connect(
            scenario_name + "_mesh.x_struct0", scenario_name + ".x_struct0"
        )
        self.connect("dv_comp.dv_struct", scenario_name + ".dv_struct")


def define_load_reference_axis(layout, bdf, le_xzy_array, te_xyz_array):
    """
    Define the load reference axis using RBE2 and RBE3 elements.
    
    Parameters
    ----------
    layout : layout object
        The layout object created with pyLayout.
    bdf : BDF object
        The BDF object containing the finite element model.
    le_xzy_array : array-like
        The leading edge coordinates.
    te_xyz_array : array-like
        The trailing edge coordinates.

    Returns
    -------
    spline_nodes_ids : array-like
        The IDs of the spline nodes created.
    """
    # Find nodes at the intersection between ribs and skin
    intersection_nodes_ids = find_intersection_nodes(layout, ["RIB", "SKIN"])
    # Find last node id and intialize array of spline nodes ids
    last_node_id = len(bdf.nodes)
    spline_nodes_ids = np.arange(
        last_node_id + 1, last_node_id + 1 + 3 * len(le_xzy_array)
    )

    # Create a (N,3) array of coordinates for the intersection nodes
    intersection_nodes_xyz_array = np.vstack(
        [bdf.nodes[node_id].xyz for node_id in intersection_nodes_ids]
    )

    # Loop over the y-coordinates of the ribs
    te_node_id = last_node_id  # initialize trailing edge node id
    for le_xyz, te_xyz in zip(le_xzy_array, te_xyz_array):
        # Add node at quarter chord
        c_4_node_id = te_node_id + 1
        c_4_xyz = le_xyz + (te_xyz - le_xyz) / 4
        bdf.add_grid(c_4_node_id, c_4_xyz)

        # Find id of the skin nodes intersecting current rib
        connection_nodes_ids = intersection_nodes_ids[
            np.isclose(intersection_nodes_xyz_array[:, 1],c_4_xyz[1])]

        # Add RBE3 element to connect quarter chord node with skin nodes
        rbe3_eid = len(bdf.elements) + len(bdf.rigid_elements) + 1
        bdf.add_rbe3(
            eid=rbe3_eid,
            refgrid=c_4_node_id,
            refc="123456",
            weights=[1.0],
            comps=["123456"],
            Gijs=[connection_nodes_ids.tolist()],
        )

        # Add nodes at wing leading and trailing edge
        le_node_id = c_4_node_id + 1
        bdf.add_grid(le_node_id, le_xyz)
        te_node_id = le_node_id + 1
        bdf.add_grid(te_node_id, te_xyz)

        # Add RBE2 element to connect leading and trailing edge node with
        # quarter chord node
        rbe2_eid = len(bdf.elements) + len(bdf.rigid_elements) + 1
        bdf.add_rbe2(rbe2_eid, c_4_node_id, "123456", [le_node_id, te_node_id])

    # Return spline nodes ids
    return spline_nodes_ids

