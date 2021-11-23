"""
Created on Fri Apr 23 16:14:08 2021

@author: valentina
"""

import matplotlib.pyplot as plt

def prepare_plot(title, subtitle, xlabel, ylabel):
    """set up a figure and axis with some basic properties"""
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    fig.suptitle(title, y=1.01, fontsize=14) # make title larger than default
    ax.set_title(subtitle)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax

def plot_error_ref(ax, dt):
    """create an error log-log plot in the current axis"""
    o1 = dt
    o2 = dt*dt
    o3 = dt**3
    o4 = dt**4
    ax.loglog(dt, o1, 'k-', label='$\mathcal{O}(\Delta t)$', lw=1)
    ax.loglog(dt, o2, 'k--', label='$\mathcal{O}(\Delta t^2)$', lw=1)
    ax.loglog(dt, o3, 'k:', label='$\mathcal{O}(\Delta t^3)$', lw=1)
    ax.loglog(dt, o4, 'k-.', label='$\mathcal{O}(\Delta t^4)$', lw=1)
    return ax