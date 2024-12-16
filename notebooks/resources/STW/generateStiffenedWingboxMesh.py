"""
==============================================================================
Wingbox mesh generation
==============================================================================
@File    :   generate_wingbox.py
@Date    :   2023/03/29
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import argparse
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from pygeo import pyGeo, geo_utils
from pylayout import pyLayout

# ==============================================================================
# Extension modules
# ==============================================================================
from wingGeometry import wingGeometry  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--order", type=int, default=2, help="Order of elements", choices=[2, 3, 4])
parser.add_argument(
    "--nChord",
    type=int,
    default=25,
    help="Number of elements in the chordwise direction",
)
parser.add_argument(
    "--nSpan",
    type=int,
    default=10,
    help="Number of elements in the spanwise direction (between each pair of ribs)",
)
parser.add_argument(
    "--nVertical",
    type=int,
    default=10,
    help="Number of elements in the vertical direction",
)
parser.add_argument("--name", type=str, default="stiffened_wingbox", help="Name of output file")
args = parser.parse_args()

# ==============================================================================
#       Specify number of stringers for each skin
# ==============================================================================
stiffener_pitch = .15   # m
stiffener_height = .05  # m
le_spar_coords = wingGeometry["wingbox"]["LESparCoords"]
te_spar_coords = wingGeometry["wingbox"]["TESparCoords"]
root_width = te_spar_coords[0][0] - le_spar_coords[0][0]
tip_width = te_spar_coords[2][0] - le_spar_coords[2][0]
num_stringers = int((root_width + tip_width) / 2 / stiffener_pitch) - 1

# ==============================================================================
#       Specify wingbox properties
# ==============================================================================
chords = wingGeometry["wing"]["sectionChord"]  # root and tip chords
sob = wingGeometry["wingbox"]["SOB"]  # span location of side-of-body
ncols = wingGeometry["wingbox"]["numRibs"]  # number of columns (aligned with ribs)
nrows = wingGeometry["wingbox"]["numSpars"] + num_stringers  # number of rows (aligned with spars)
nbreak = wingGeometry["wingbox"]["numRibsCentrebody"]  # column index of side-of-body kink

# ==============================================================================
#       Specify number of quad elements between each component
# ==============================================================================
target_length = .1126  # m
num_elements = lambda l: int(np.ceil(l / target_length / 2)) * 2  # rounding up is used because target element length acts as the maximum allowable length

num_element_chord = num_elements(stiffener_pitch)  # elements between each spar/stringer pair

side_of_body = wingGeometry["wingbox"]["SOB"]
semispan = wingGeometry["wing"]["semiSpan"]
num_element_span_centrebody = num_elements(side_of_body/(nbreak - 1))  # elements between each rib pair in the centrebody
num_element_span_outer = num_elements((semispan - side_of_body)/(ncols - nbreak - 1))  # elements between each rib pair in the outer wing

root_le_height = .5863  # m
tip_te_height = .1249   # m
num_element_vertical = num_elements((root_le_height + tip_te_height)/2)  # elements between skins

num_element_stringers = num_elements(.05*(root_le_height + tip_te_height)/2)

colSpace = np.ones(ncols - 1, "intc")  # elements between columns
colSpace[:nbreak - 1] = num_element_span_centrebody
colSpace[nbreak - 1:] = num_element_span_outer
rowSpace = num_element_chord * np.ones(nrows + 1, "intc")  # elements between rows

# ==============================================================================
#       Set up blanking arrays
# ==============================================================================

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

# ==============================================================================
#       Set up array of grid coordinates for ribs, spars
# ==============================================================================

# Initialize grid coordinate matrix
X = np.zeros((ncols, nrows, 3))

# Fill in LE and TE coordinates from root to side-of-body
X[0:nbreak, 0] = geo_utils.linearEdge(le_spar_coords[0], le_spar_coords[1], nbreak)
X[0:nbreak, -1] = geo_utils.linearEdge(te_spar_coords[0], te_spar_coords[1], nbreak)

# Fill in LE and TE coordinates from side-of-body to tip
X[nbreak - 1 : ncols, 0] = geo_utils.linearEdge(le_spar_coords[1], le_spar_coords[2], ncols - nbreak + 1)
X[nbreak - 1 : ncols, -1] = geo_utils.linearEdge(te_spar_coords[1], te_spar_coords[2], ncols - nbreak + 1)

# Finally fill in chord-wise with linear edges
for i in range(ncols):
    X[i, :] = geo_utils.linearEdge(X[i, 0], X[i, -1], nrows)

# Boundary conditions
spanIndex = wingGeometry["spanIndex"]
chordIndex = wingGeometry["chordIndex"]
verticalIndex = wingGeometry["verticalIndex"]
symBCDOF = [spanIndex, chordIndex + 3, verticalIndex + 3]
symBCDOF = "".join([str(dof + 1) for dof in symBCDOF])

sobBCDOF = [chordIndex, verticalIndex]
sobBCDOF = "".join([str(dof + 1) for dof in sobBCDOF])

ribBC = {
    0: {"all": symBCDOF},
    nbreak - 1: {"edge": sobBCDOF},
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
    elementOrder=args.order,
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
    minStringerHeight=stiffener_height,
    maxStringerHeight=stiffener_height,
)
# Write bdf file
layout.finalize(f"{args.name}.bdf")

# Write a tecplot file
# layout.writeTecplot(f"{args.name}.dat")
