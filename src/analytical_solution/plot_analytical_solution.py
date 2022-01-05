import numpy as np
import matplotlib.pyplot as plt

from ..utility import prepare_plot

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

def create_diff_timescale_plot():
    k1 = 20
    k2 = 0.1
    k12 = 0.5
    m1 = 1
    m2 = 1
    title = "Analytical Solution - Different Time Scales"
    subtitle = rf"$k_1$ = {k1}, $k_2$ = {k2}, $k_{{{12}}}$ = {k12}"
    xlabel = "time [s]"
    ylabel = "u(t)"
    t = np.linspace(0, 20, 1000)
    u1 = 0.0126747231152769*np.cos(0.542045916608024*t) + 0.987325276884725*np.cos(4.52837567172696*t)
    u2 = 0.51221563121716*np.cos(0.542045916608024*t) - 0.0122156312171607*np.cos(4.52837567172696*t)

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

def create_same_timescale_plot():
    title = "Analytical Solution - Same Time Scales"
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
    create_same_timescale_plot()