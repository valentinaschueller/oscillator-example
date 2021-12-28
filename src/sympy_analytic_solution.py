from sympy import *
from sympy.plotting import plot as symplot
from sympy import lambdify
import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from utility import prepare_plot

k1 = 20
k2 = 0.1
k12 = 0.5
m1 = 1
m2 = 1

def solve_two_mass_spring_system():
    # two mass-spring-system
    init_printing(use_unicode=True)
    t = symbols("t")
    u1, u2, v1, v2 = symbols("u1 u2 v1 v2", cls=Function)

    initial_conditions = {
        u1(0): 1,
        u2(0): 0.5,
        v1(0): 0,
        v2(0): 0,
    }

    eq_u1 = Eq(u1(t).diff(t), v1(t))
    eq_u2 = Eq(u2(t).diff(t), v2(t))
    eq_v1 = Eq(v1(t).diff(t), -(k1 + k12)/m1 * u1(t) + k12/m1 * u2(t))
    eq_v2 = Eq(v2(t).diff(t), -(k2 + k12)/m2 * u2(t) + k12/m2 * u1(t))

    result = dsolve(
        [eq_u1, eq_u2, eq_v1, eq_v2],
        [u1(t), u2(t), v1(t), v2(t)],
        ics=initial_conditions,
        )
    print(result)
    p1 = symplot(result[0].rhs, (t,0,20), line_color="tab:blue", show=False)
    p2 = symplot(result[1].rhs, (t,0,20), line_color="tab:orange", show=False)
    p1.extend(p2)
    p1.show()


if __name__ == '__main__':
    solve_two_mass_spring_system()