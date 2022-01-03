"""
Created on Fri Apr 23 16:14:08 2021

@author: valentina
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def prepare_plot(title, subtitle, xlabel, ylabel):
    """set up a figure and axis with some basic properties"""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    if subtitle != "":
        fig.suptitle(title, y=1.01, fontsize=14) # make title larger than default
        ax.set_title(subtitle)
    else:
        fig.suptitle(title, fontsize=14) # make title larger than default

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

def plot_displacements(t, sol, path="."):
    _, ax = prepare_plot("Displacements", "", "t [s]", "u [m]")
    ax.plot(t, sol[0,:], label="$u_1$")
    ax.plot(t, sol[1,:], label="$u_2$")
    ax.legend()
    plt.savefig(f"{path}/displacements.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_velocities(t, sol, path="."):
    _, ax = prepare_plot("Velocities", "", "t [s]", "v [m/s]")
    ax.plot(t, sol[2,:], label="$v_1$")
    ax.plot(t, sol[3,:], label="$v_2$")
    ax.legend()
    plt.savefig(f"{path}/velocities.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_energy(t, energy, path="."):
    _, ax = prepare_plot("Energy", "", "t [s]", "Energy")
    ax.plot(t, energy, label='energy')
    ax.legend()
    plt.savefig(f"{path}/energy.png", dpi=300, bbox_inches="tight")
    plt.close()

def create_solution_plots(t, sol, dir_name: str ="plots"):
    plotdir_path = f"./{dir_name}"
    try:
        os.mkdir(plotdir_path)
    except FileExistsError:
        pass
    energy = compute_energy(sol[0,:], sol[1,:], sol[2,:], sol[3,:])
    plot_displacements(t, sol, plotdir_path)
    plot_velocities(t, sol, plotdir_path)
    plot_energy(t, energy, plotdir_path)

def compute_energy(u1, u2, v1, v2):
    return v1**2 + v2**2 + u1**2 + u2**2 + (u2-u1)**2

def l1_norm(vec):
    return np.sum(np.abs(vec))

def l2_norm(vec):
    return np.sum(vec**2)

def max_norm(vec):
    return np.max(np.abs(vec))

def analytical_solution(t_end: float, N: int):
    t = np.linspace(0, t_end, N+1)
    result = np.array([
        0.5 * (np.cos(t) + np.cos(np.sqrt(3)*t)),
        0.5 * (np.cos(t) - np.cos(np.sqrt(3)*t)),
        0.5 * (- np.sin(t) - np.sqrt(3) * np.sin(np.sqrt(3)*t)),
        0.5 * (- np.sin(t) + np.sqrt(3) * np.sin(np.sqrt(3)*t)),
    ])
    return result

def interpolate_linear(left_value, right_value, percentage):
    if percentage == 0:
        return left_value
    elif percentage == 1:
        print(f"What's happening? {left_value}, {right_value}, {percentage}")
        return right_value
    else:
        return percentage * right_value + (1 - percentage) * left_value