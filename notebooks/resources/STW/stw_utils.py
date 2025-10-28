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
        The ratio of the forces at the two ends.
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
    # Solve for coefficients a and b in F(x) = a + b*x
    # Constraint 1: F(x_first) = ratio * F(x_last)
    # Constraint 2: sum(F(x_i)) = total_force

    # Apply constraint 1 and solve for a
    # a + b·x_first = ratio × (a + b·x_last)
    # a + b·x_first = ratio·a + ratio·b·x_last
    # a - ratio·a = ratio·b·x_last - b·x_first
    # a(1 - ratio) = b(ratio·x_last - x_first)
    # a = b · (ratio·x_last - x_first) / (1 - ratio)

    # Apply constraint 2 and solve for b
    # Σ (a + b·x_i) = total_force
    # n·a + b·Σx_i = total_force
    # n·a + b·sum_x = total_force
    # n · [b · (ratio·x_last - x_first) / (1 - ratio)] + b·sum_x = total_force
    # b · [n · (ratio·x_last - x_first) / (1 - ratio) + sum_x] = total_force
    # b = total_force / [n · (ratio·x_last - x_first) / (1 - ratio) + sum_x]

    # Extract x coordinates and compute necessary sums
    # Note: this code assumes that the x-coordinates are unique and ordered,
    # which is indeed the case for how the mesh of the STW wingbox is generated
    # with pyLayout. If this is not the case, the code should be modified
    # to find the correct x_first and x_last values.
    x_coords = node_xyz_array[:, 0]
    x_first = x_coords[0]
    x_last = x_coords[-1]
    num_nodes = len(x_coords)
    sum_x = np.sum(x_coords)

    # Calculate coefficients a and b
    denominator = num_nodes * (ratio * x_last - x_first) / (1 - ratio) + sum_x
    b = total_force / denominator
    a = b * (ratio * x_last - x_first) / (1 - ratio)

    # Calculate forces
    forces = a + b * x_coords

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
    part_element_ids : dict
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
