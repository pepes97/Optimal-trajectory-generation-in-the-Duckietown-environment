"""utils.py
"""

from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec

def generate_4_2_layout():
    """ Returns a 4x2 standard layout
    """
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 2)
    return fig, gs

def generate_1_1_layout():
    """ Returns a 1x1 standard layout
    """
    fig, ax = plt.subplots(1, 1)
    return fig, ax
