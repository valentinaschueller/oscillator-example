import numpy as np
import matplotlib.pyplot as plt

from utility import prepare_plot

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

def create_diff_timescale_plot():
    k1 = 20
    k2 = 0.1
    k12 = 1
    m1 = 1
    m2 = 1
    title = "Analytical Solution - Different Time Scales"
    subtitle = rf"$k_1$ = {k1}, $k_2$ = {k2}, $k_{{{12}}}$ = {k12}"
    xlabel = "time [s]"
    ylabel = "u(t)"
    t = np.linspace(0, 20, 1000)
    w1 = 1.02463408140723
    w2 = 4.58804152108705
    u1 = 0.0262527968225597*np.cos(w1 * t) + 0.473747203177441*np.cos(w2 * t)
    u2 = 0.523746578189158*np.cos(w1 * t) - 0.0237465781891588*np.cos(w2 * t)

    _, ax = prepare_plot(title, subtitle, xlabel, ylabel)
    ax.plot(t, u1, label=r'$u_1$', color='darkcyan', linestyle='--')
    ax.plot(t, u2, label=r'$u_2$', color='olive')
    # colors of axes
    ax.spines['bottom'].set_color('k')
    ax.xaxis.label.set_color('k')
    ax.tick_params(axis='x', colors='k')
    ax.spines['left'].set_color('k')
    ax.yaxis.label.set_color('k')
    ax.tick_params(axis='y', colors='k')
    # remove top and right spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # move x axis up
    ax.spines['bottom'].set_position(('data', 0.0))
    plt.locator_params(axis="x", nbins=4)
    plt.locator_params(axis="y", nbins=6)
    ax.xaxis.set_label_coords(1,0.55)
    ax.legend()
    plt.savefig("ana_sol_diff_scale.pdf", dpi=300, bbox_inches='tight')

def create_diff_timescale_plot():
    k1 = 20
    k2 = 0.1
    k12 = 1
    m1 = 1
    m2 = 1
    title = "Oscillating Masses -- Different Time Scales"
    subtitle = rf"$k_1$ = {k1}, $k_2$ = {k2}, $k_{{{12}}}$ = {k12}"
    xlabel = "time [s]"
    ylabel = "u(t)"
    t = np.linspace(0, 20, 1000)
    w1 = 1.02463408140723
    w2 = 4.58804152108705
    u1 = 0.0262527968225597*np.cos(w1 * t) + 0.473747203177441*np.cos(w2 * t)
    u2 = 0.523746578189158*np.cos(w1 * t) - 0.0237465781891588*np.cos(w2 * t)

    _, ax = prepare_plot(title, subtitle, xlabel, ylabel)
    ax.plot(t, u1, label=r'$u_1$', color='darkcyan', linestyle='--')
    ax.plot(t, u2, label=r'$u_2$', color='olive')
    # colors of axes
    ax.spines['bottom'].set_color('k')
    ax.xaxis.label.set_color('k')
    ax.tick_params(axis='x', colors='k')
    ax.spines['left'].set_color('k')
    ax.yaxis.label.set_color('k')
    ax.tick_params(axis='y', colors='k')
    # remove top and right spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # move x axis up
    ax.spines['bottom'].set_position(('data', 0.0))
    plt.locator_params(axis="x", nbins=4)
    plt.locator_params(axis="y", nbins=6)
    ax.xaxis.set_label_coords(1,0.55)
    ax.legend()
    plt.savefig("ana_sol_diff_scale_pres.pdf", dpi=300, bbox_inches='tight')

def create_same_timescale_plot_pres():
    title = "Analytical Solution"
    subtitle = rf"$k_1$ = $k_2$ = $k_{{{12}}}$ = 1"
    xlabel = "time [s]"
    ylabel = "u(t)"
    t = np.linspace(0, 20, 1000)

    u1 = 0.5 * (np.cos(t) + np.cos(np.sqrt(3)*t))
    u2 = 0.5 * (np.cos(t) - np.cos(np.sqrt(3)*t))

    _, ax = prepare_plot(title, subtitle, xlabel, ylabel)
    ax.plot(t, u1, label=r'$u_1$', color='darkcyan', linestyle='--')
    ax.plot(t, u2, label=r'$u_2$', color='olive')
    # colors of axes
    ax.spines['bottom'].set_color('k')
    ax.xaxis.label.set_color('k')
    ax.tick_params(axis='x', colors='k')
    ax.spines['left'].set_color('k')
    ax.yaxis.label.set_color('k')
    ax.tick_params(axis='y', colors='k')
    # remove top and right spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # move x axis up
    ax.spines['bottom'].set_position(('data', 0.0))
    plt.locator_params(axis="x", nbins=4)
    plt.locator_params(axis="y", nbins=6)
    ax.xaxis.set_label_coords(1,0.55)
    ax.legend()
    plt.savefig("ana_sol_same_scale_pres.pdf", dpi=300, bbox_inches='tight')

def create_same_timescale_plot():
    title = "Analytical Solution"
    subtitle = rf"$k_1$ = $k_2$ = $k_{{{12}}}$ = 1"
    xlabel = "time [s]"
    ylabel = "u(t)"
    t = np.linspace(0, 20, 1000)

    u1 = 0.5 * (np.cos(t) + np.cos(np.sqrt(3)*t))
    u2 = 0.5 * (np.cos(t) - np.cos(np.sqrt(3)*t))

    _, ax = prepare_plot(title, subtitle, xlabel, ylabel)
    ax.plot(t, u1, label=r'$u_1$', color='darkcyan', linestyle='--')
    ax.plot(t, u2, label=r'$u_2$', color='olive')
    # colors of axes
    ax.spines['bottom'].set_color('k')
    ax.xaxis.label.set_color('k')
    ax.tick_params(axis='x', colors='k')
    ax.spines['left'].set_color('k')
    ax.yaxis.label.set_color('k')
    ax.tick_params(axis='y', colors='k')
    # remove top and right spine
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # move x axis up
    ax.spines['bottom'].set_position(('data', 0.0))
    plt.locator_params(axis="x", nbins=4)
    plt.locator_params(axis="y", nbins=6)
    ax.xaxis.set_label_coords(1,0.55)
    ax.legend()
    plt.savefig("ana_sol_same_scale.pdf", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    create_diff_timescale_plot()
    create_same_timescale_plot_pres()
    create_same_timescale_plot()
    create_same_timescale_plot_pres()