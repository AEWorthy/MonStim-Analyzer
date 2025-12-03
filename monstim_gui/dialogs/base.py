from matplotlib import colors as mcolors

# Small set of pleasant colors for latency windows
COLOR_OPTIONS = list(mcolors.TABLEAU_COLORS.keys())
TAB_COLOR_NAMES = [c.replace("tab:", "") for c in COLOR_OPTIONS]
