"""
========================================================================
Baseline geometry definition
========================================================================
@File    :   wingGeometry.py
@Date    :   2023/03/28
@Author  :   Alasdair Christison Gray
@Modified :   2024/12/12
@ModifiedBy : Francesco Mario Antonio Mitrotta
@Description : This file contains all data required to define the geometry of the OML and wingbox for the MACH tutorial wing. Any code that works with the geometry of the wing should import this file and use the data contained within.
@Modifications : Added utility functions to calculate the mean aerodynamic chord of a trapezoidal wing, to generate a layout object for the TSW wingbox model, and to create a base bdf input file. Also added cross-sectional and material properties for the wingbox model.
"""

# ======================================================================
# Standard Python modules
import os
# ======================================================================

# ======================================================================
# External Python modules
# ======================================================================
import numpy as np
from pygeo import pyGeo, geo_utils
from pylayout import pyLayout
from typing import Tuple
from pylayout.pyLayout import Layout
from pyNastran.bdf.bdf import BDF

# ======================================================================
# Helper functions
# ======================================================================
def trapezoidal_wing_mean_aerodynamic_chord(
    root_chord: float, taper_ratio: float) -> float:
    """
    Calculate the mean aerodynamic chord of a trapezoidal wing.

    Parameters
    ----------
    root_chord: float
        Root chord of the wing
    taper_ratio: float
        Taper ratio of the wing

    Returns
    -------
    mean_aerodynamic_chord: float
        Mean aerodynamic chord of the wing
    """
    return 2/3 * root_chord * (1 + taper_ratio + taper_ratio**2) /\
        (1 + taper_ratio)

# ======================================================================
# Direction definition
# ======================================================================
spanIndex = 1  # index of the spanwise direction (0=x, 1=y, 2=z)
chordIndex = 0  # index of the chordwise direction (0=x, 1=y, 2=z)
verticalIndex = 2  # index of the vertical direction (0=x, 1=y, 2=z)

# ======================================================================
# OML Definition
# ======================================================================
semiSpan = 14.0  # semi-span of the wing in metres

sectionEta = np.array([0.0, 1.0])  # Normalised spanwise coordinates of the wing sections
sectionChord = np.array([5.0, 1.5])  # Chord length of the wing sections in metres
sectionChordwiseOffset = np.array([0.0, 7.5])  # Offset of each section in the chordwise direction in metres
sectionVerticalOffset = np.array([0.0, 0.0])  # Offset of each section in the vertical direction in metres
sectionTwist = np.array([0.0, 0.0])  # Twist of each section (around the spanwise axis) in degrees
sectionProfiles = ["rae2822.dat"] * 2  # Airfoil profile files for each section

teHeight = 0.25 * 0.0254  # thickness of the trailing edge (1/4 inch) in metres

LECoords = np.zeros((2, 3))  # Leading edge coordinates of each section
LECoords[:, spanIndex] = semiSpan * sectionEta
LECoords[:, chordIndex] = sectionChordwiseOffset
LECoords[:, verticalIndex] = sectionVerticalOffset

TECoords = np.zeros((2, 3))  # Trailing edge coordinates of each section
TECoords = np.zeros((2, 3))  # Trailing edge coordinates of each section
TECoords[:, spanIndex] = semiSpan * sectionEta
TECoords[:, chordIndex] = sectionChordwiseOffset + sectionChord *\
    np.cos(np.deg2rad(sectionTwist))
TECoords[:, verticalIndex] = sectionVerticalOffset - sectionChord *\
    np.sin(np.deg2rad(sectionTwist))


rootChord = sectionChord[0]
tipChord = sectionChord[1]
planformArea = semiSpan * (rootChord + tipChord) * 0.5  # planform area of the half wing

aspectRatio = 2 * (semiSpan**2) / planformArea
taperRatio = tipChord / rootChord

meanAerodynamicChord = trapezoidal_wing_mean_aerodynamic_chord(
    root_chord=rootChord, taper_ratio=taperRatio)

# --- No do the same for the tails ---
hTailRootChord = 3.25
hTailTipChord = 1.22
hTailSemiSpan = 6.5
hTailPlanformArea = hTailSemiSpan * (hTailRootChord + hTailTipChord) * 0.5
hTailaspectRatio = 2 * (hTailSemiSpan**2) / hTailPlanformArea
hTailSweep = 30.0
hTailTaperRatio = hTailTipChord / hTailRootChord
hTailMeanAerodynamicChord = trapezoidal_wing_mean_aerodynamic_chord(
    root_chord=hTailRootChord, taper_ratio=hTailTaperRatio)

vTailRootChord = 15.3 * 0.3048
vTailTipChord = 12.12 * 0.3048
vTailSemiSpan = 15.72 * 0.3048
vTailPlanformArea = vTailSemiSpan * (vTailRootChord + vTailTipChord) * 0.5
vTailaspectRatio = 2 * (vTailSemiSpan**2) / vTailPlanformArea
vTailSweep = 37.0
vTailTaperRatio = vTailTipChord / vTailRootChord
vTailMeanAerodynamicChord = trapezoidal_wing_mean_aerodynamic_chord(
    root_chord=vTailRootChord, taper_ratio=vTailTaperRatio)

# --- Nacelle ---
nacelleLength = 5.865
nacelleDiameter = 1.8
nacelleArea = np.pi * nacelleDiameter * nacelleLength

# --- Fuselage ---
fuselageLength = 112 * 0.3048  # 112 ft in metres
fuselageWidth = 3.4  # metres
fuselageArea = fuselageLength * np.pi * fuselageWidth  # very approximate

# ======================================================================
# Wingbox Definition
# ======================================================================
SOB = 1.5  # Spanwise coordinate of the side-of-body junction in metres
LESparFrac = 0.15  # Normalised chordwise location of the leading-edge spar
TESparFrac = 0.65  # Normalised chordwise location of the trailing-edge spar
numRibsCentrebody = 4  # Number of ribs in the centre wingbox
numRibsOuter = 19  # Number of ribs outboard of the SOB
numRibs = numRibsCentrebody + numRibsOuter  # Total number of ribs
numSpars = 2  # Number of spars (front and rear only)

stiffenerPitch = .15   # [m]
stiffenerHeight = .05  # [m]
rootLESparHeight = .5863  # [m]
tipTESparHeight = .1249   # [m]

LESparCoords = np.zeros((3, 3))  # Leading edge spar coordinates of each section
TESparCoords = np.zeros((3, 3))  # Trailing edge spar coordinates of each section

# Tip is easy because we know the chord length there
LESparCoords[-1] = LECoords[-1] + LESparFrac * (TECoords[-1] - LECoords[-1])
TESparCoords[-1] = LECoords[-1] + TESparFrac * (TECoords[-1] - LECoords[-1])

# We need to shift the tip of the wingbox slightly off from the tip of the OML
# so that pylayout's projections work
LESparCoords[-1, spanIndex] -= 1e-3
TESparCoords[-1, spanIndex] -= 1e-3

# For the side of body, we need to interpolate the leading and trailing edge
# coordinates
sobLE = LECoords[0] + SOB / semiSpan * (LECoords[1] - LECoords[0])
sobTE = TECoords[0] + SOB / semiSpan * (TECoords[1] - TECoords[0])
LESparCoords[1] = sobLE + LESparFrac * (sobTE - sobLE)
TESparCoords[1] = sobLE + TESparFrac * (sobTE - sobLE)

# From the side of body to the root, there is no sweep, so we just shift the
# SOB coordinates in the spanwise direction, then correct the vertical position
# so that the spar points lie on the root chord line
# We again need to shift the root of the wingbox slightly off from the root of
# the OML so that pylayout's projections work
LESparCoords[0] = LESparCoords[1]
LESparCoords[0, spanIndex] = 1e-3

TESparCoords[0] = TESparCoords[1]
TESparCoords[0, spanIndex] = 1e-3

rootLESparFrac = (LESparCoords[0, chordIndex] - LECoords[0, chordIndex]) / (
    TECoords[0, chordIndex] - LECoords[0, chordIndex]
)
rootTESparFrac = (TESparCoords[0, chordIndex] - LECoords[0, chordIndex]) / (
    TECoords[0, chordIndex] - LECoords[0, chordIndex]
)

LESparCoords[0, verticalIndex] = LECoords[0, verticalIndex] + rootLESparFrac * (
    TECoords[0, verticalIndex] - LECoords[0, verticalIndex]
)
TESparCoords[0, verticalIndex] = LECoords[0, verticalIndex] + rootTESparFrac * (
    TECoords[0, verticalIndex] - LECoords[0, verticalIndex]
)

# ======================================================================
# Cross-sectional properties definition
# ======================================================================
panelThickness = .0065  # [m]
stiffenerThickness = .006  # [m]

# ======================================================================
# Material properties definition
# ======================================================================
density = 2780.  # [kg/m^3]
youngModulus = 73.1e9  # [Pa]
poissonRatio = .3
yieldStrength = 420e6  # [Pa]

# ======================================================================
# BDF input generation functions
# ======================================================================
def create_layout(target_length: float, element_order: int = 2) -> Layout:
    """
    Generate a layout object for the TSW wingbox model.
    
    Parameters
    ----------
    target_length: float
        Target element length for the mesh
    element_order: int
        Order of the elements used in the mesh
    
    Returns
    -------
    layout: Layout
        Layout object containing the generated wingbox layout
    """
    # ==================================================================
    #       Specify number of stringers for each skin
    # ==================================================================
    root_width = TESparCoords[0][0] - LESparCoords[0][0]
    tip_width = TESparCoords[2][0] - LESparCoords[2][0]
    num_stringers = int((root_width + tip_width) / 2 / stiffenerPitch) - 1

    # ==================================================================
    #       Specify wingbox properties
    # ==================================================================
    ncols = numRibs  # number of columns (aligned with ribs)
    nrows = numSpars + num_stringers  # number of rows (aligned with spars)

    # ==================================================================
    #       Specify number of quad elements between each component
    # ==================================================================
    # Function to discretize a length into an even number of elements
    num_elements = lambda l: int(np.ceil(l / target_length / 2)) * 2

    num_element_chord = num_elements(stiffenerPitch)  # elements between each spar/stringer pair

    num_element_span_centrebody = num_elements(SOB/(numRibsCentrebody - 1))  # elements between each rib pair in the centrebody
    num_element_span_outer = num_elements((semiSpan - SOB)/numRibsOuter)  # elements between each rib pair in the outer wing

    num_element_vertical = num_elements((rootLESparHeight + tipTESparHeight)/2)  # elements between skins

    num_element_stringers = num_elements(stiffenerHeight)  # elmements within each stringer

    colSpace = np.ones(ncols - 1, "intc")  # elements between columns
    colSpace[:numRibsCentrebody - 1] = num_element_span_centrebody
    colSpace[numRibsCentrebody - 1:] = num_element_span_outer
    rowSpace = num_element_chord * np.ones(nrows + 1, "intc")  # elements between rows

    # ==================================================================
    #       Set up blanking arrays
    # ==================================================================
    # Blanking for ribs (None)
    ribBlank = np.ones((ncols, nrows - 1), "intc")

    # Blanking for spars
    sparBlank = np.zeros((nrows, ncols - 1), "intc")
    sparBlank[0, :] = 1  # Keep First
    sparBlank[-1, :] = 1  # Keep Last

    # Blanking for top and bottom skins stringers
    topStringerBlank = np.ones((nrows, ncols - 1), "intc")
    topStringerBlank[0, :] = 0  # no stringers at front spar
    topStringerBlank[-1, :] = 0  # no stringers at rear spar
    botStringerBlank = np.ones((nrows, ncols - 1), "intc")
    botStringerBlank[0, :] = 0  # no stringers at front spar
    botStringerBlank[-1, :] = 0  # no stringers at rear spar

    # Blanking for rib stiffeners:
    ribStiffenerBlank = np.zeros((ncols, nrows), "intc")  # No rib stiffeners
    teEdgeList = []

    # ==================================================================
    #       Set up array of grid coordinates for ribs, spars
    # ==================================================================
    # Initialize grid coordinate matrix
    X = np.zeros((ncols, nrows, 3))

    # Fill in LE and TE coordinates from root to side-of-body
    X[0:numRibsCentrebody, 0] = geo_utils.linearEdge(
        LESparCoords[0], LESparCoords[1], numRibsCentrebody)
    X[0:numRibsCentrebody, -1] = geo_utils.linearEdge(
        TESparCoords[0], TESparCoords[1], numRibsCentrebody)

    # Fill in LE and TE coordinates from side-of-body to tip
    X[numRibsCentrebody - 1 : ncols, 0] = geo_utils.linearEdge(
        LESparCoords[1], LESparCoords[2], ncols - numRibsCentrebody + 1)
    X[numRibsCentrebody - 1 : ncols, -1] = geo_utils.linearEdge(
        TESparCoords[1], TESparCoords[2], ncols - numRibsCentrebody + 1)

    # Finally fill in chord-wise with linear edges
    for i in range(ncols):
        X[i, :] = geo_utils.linearEdge(X[i, 0], X[i, -1], nrows)

    # Boundary conditions
    symBCDOF = [spanIndex, chordIndex + 3, verticalIndex + 3]
    symBCDOF = "".join([str(dof + 1) for dof in symBCDOF])

    sobBCDOF = [chordIndex, verticalIndex]
    sobBCDOF = "".join([str(dof + 1) for dof in sobBCDOF])

    ribBC = {
        0: {"all": symBCDOF},
        numRibsCentrebody - 1: {"edge": sobBCDOF},
    }

    # ==============================================================================
    #       Generate wingbox
    # ==============================================================================
    # Get surface definition to use for projections
    surface_filename = "wing.igs"  # debug
    surface_filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), surface_filename)
    geo = pyGeo("iges", fileName=surface_filepath)

    # Initialize pyLayout
    layout = pyLayout.Layout(
        geo,
        teList=[],
        nribs=ncols,
        nspars=nrows,
        elementOrder=element_order,
        X=X,
        ribBlank=ribBlank,
        sparBlank=sparBlank,
        ribStiffenerBlank=ribStiffenerBlank,
        spanSpace=colSpace,
        ribSpace=rowSpace,
        vSpace=num_element_vertical,
        ribBC=ribBC,
        rightWing=True,
        topStringerBlank=topStringerBlank,
        botStringerBlank=botStringerBlank,
        stringerSpace=num_element_stringers,
        minStringerHeight=stiffenerHeight,
        maxStringerHeight=stiffenerHeight,
    )
    
    # Return the layout object
    return layout

def create_base_bdf(
    target_length: float, element_order: int = 2,
    parallel:bool = False, no_cores:int = 4) -> Tuple[BDF, Layout]:
    """
    Generate a base bdf input file for the TSW wingbox model.

    Parameters
    ----------
    target_length: float
        Target element length for the mesh
    element_order: int
        Order of the elements used in the mesh
    parallel: bool
        flag to enable parallel execution of Nastran
    no_cores: int
        Number of cores to use for parallel execution

    Returns
    -------
    bdf: BDF
        BDF object containing the generated input file
    layout: Layout
        Layout object containing the generated wingbox layout
    """
    # Create pyLayout object
    layout = create_layout(target_length, element_order)
    
    # Write bdf file
    directory_path = os.path.dirname(os.path.abspath(__file__))
    bdf_filepath = os.path.join(directory_path, "stw_wingbox.bdf")
    layout.finalize(bdf_filepath)
        
    # Create an instance of the BDF class without debug or info messages
    bdf = BDF(debug=None)
    
    # Add MAT1 card (isotropic material)
    material_id = 1
    bdf.add_mat1(
        mid=material_id, E=youngModulus, G='', nu=poissonRatio, rho=density)
    
    # Create a list of thickness values based on the face descriptions
    thicknesses = [
        stiffenerThickness if 'STRING' in desc else panelThickness for desc
        in layout.faceDescript]

    # Add PSHELL card with appropriate thickness value for panels and stringers
    for i, (desc, t) in enumerate(zip(layout.faceDescript, thicknesses)):
        bdf.add_pshell(
            pid=i + 1, mid1=material_id, t=t, mid2=material_id,
            mid3=material_id, comment=desc)
    
    # Read bdf file into BDF object
    bdf.read_bdf(bdf_filepath)
    
    # Add SCP1 card to case control deck
    bdf.create_subcases(0)
    bdf.case_control_deck.subcases[0].add_integer_type('SPC', 1)
    
    # Set defaults for output files
    bdf.add_param('POST', [1])  # add PARAM card to store results in a op2 file
    bdf.case_control_deck.subcases[0].add('ECHO', 'NONE', [], 'STRING-type')  # request no Bulk Data to be printed
    bdf.case_control_deck.subcases[0].add_result_type(
        'DISPLACEMENT', 'ALL', ['PLOT'])  # store displacement data of all nodes in the op2 file
    bdf.case_control_deck.subcases[0].add_result_type('OLOAD', 'ALL', ['PLOT'])  # store form and type of applied load vector
    
    # Set parallel execution of Nastran if requested
    if parallel:
        bdf.system_command_lines[0:0] = [f"NASTRAN PARALLEL={no_cores:d}"]
        
    # Return the BDF object and Layout object
    return bdf, layout
