import os
import sys

# Define the paths to the dependencies directories
current_dir = os.path.dirname(__file__)
bop_toolkit_root = os.path.join(current_dir, "dependencies", "@bop_toolkit")
dex_ycb_toolkit_root = os.path.join(current_dir, "dependencies", "@dex-ycb-toolkit")
freihand_root = os.path.join(current_dir, "dependencies", "@freihand")
manopth_root = os.path.join(current_dir, "dependencies", "@manopth")


# Add these paths to sys.path
sys.path.append(bop_toolkit_root)
sys.path.append(dex_ycb_toolkit_root)
sys.path.append(freihand_root)
sys.path.append(manopth_root)

