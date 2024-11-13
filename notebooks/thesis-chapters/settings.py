import matplotlib.pyplot as plt  # module for plotting
import tol_colors as tc  # package for colorblind-friendly colors
from matplotlib.lines import Line2D  # class defining the characters for the marker styles


# Update matplotlib settings to match the style of the thesis
DEFAULT_FONT_SIZE = 10
DEFAULT_MARKER_SIZE = 3
plt.rcParams.update({
    # Font settings to match LaTeX document
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # You might need to install or specify the correct path if it's not available
    'font.size': DEFAULT_FONT_SIZE,  # Font size matches the LaTeX document class option
    
    # Figure layout
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    
    # Enabling LaTeX for text rendering
    'text.usetex': True,
    
    # Size of markers in plots
    'lines.markersize': DEFAULT_MARKER_SIZE
})

# Text width of the LaTeX document in inches
TEXTWIDTH_INCHES = 6.19802

# Set default colors for plotting
plt.rc('axes', prop_cycle=plt.cycler('color', list(tc.tol_cset('bright'))))  # set default color cycle to TOL bright
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']  # retrieve list with succession of standard matplotlib colors
UNSTABLE_COLOR = COLORS[1]  # red
GLASS_CEILING_COLOR = COLORS[2]  # green
del COLORS[1:3]  # delete green and red from list of colors
MARKERS = list(Line2D.markers.keys())[2:]  # list of marker characters

# Subcases id
FIRST_SUBCASE_ID = 1
SECOND_SUBCASE_ID = 2

# Component index
Z_COMPONENT_INDEX = 2
