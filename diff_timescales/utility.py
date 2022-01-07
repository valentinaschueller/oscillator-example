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
    u_data = np.array([u1, u2])
    v_data = np.array([v1, v2])
    m1 = 1
    m2 = 1
    k1 = 20
    k2 = 0.1
    k12 = 1
    M = np.array(
        [[m1, 0],
        [0, m2]], dtype=float)
    K = np.array(
        [[(k1 + k12), -k12],
        [-k12, (k2 + k12)]], dtype=float)
    kinetic_energy = 0.5* np.array([np.dot(v.T,np.dot(M,v)) for v in v_data.T])
    spring_energy = 0.5 * np.array([np.dot(u.T,np.dot(K,u)) for u in u_data.T])
    return kinetic_energy + spring_energy

def l1_norm(vec):
    return np.sum(np.abs(vec))

def l2_norm(vec):
    return np.sqrt(np.sum(vec**2))

def max_norm(vec):
    return np.max(np.abs(vec))

def analytical_solution(t_end: float, N: int):
    t = np.linspace(0, t_end, N+1)
    w1 = 1.02463408140723
    w2 = 4.58804152108705
    result = np.array([
            0.0262527968225597*np.cos(w1 * t) + 0.473747203177441*np.cos(w2 * t),
            0.523746578189158*np.cos(w1 * t) - 0.0237465781891588*np.cos(w2 * t),
            -0.0268995103566541*np.sin(w1 * t) - 2.17357183867696*np.sin(w2 * t),
            -0.536648594033029*np.sin(w1 * t) + 0.108950286715601*np.sin(w2 * t),
        ])
    return result

def interpolate_linear(left_value, right_value, percentage):
    if percentage == 0:
        return left_value
    elif percentage == 1:
        return right_value
    else:
        return percentage * right_value + (1 - percentage) * left_value
