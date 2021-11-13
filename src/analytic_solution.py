from sympy import *

def solve_two_mass_spring_system():
    # two mass-spring-system
    init_printing(use_unicode=True)
    t = symbols("t")
    u1, u2, v1, v2 = symbols("u1 u2 v1 v2", cls=Function)

    initial_conditions = {
        u1(0): 1,
        u2(0): 0,
        v1(0): 0,
        v2(0): 0,
    }

    eq_u1 = Eq(u1(t).diff(t), v1(t))
    eq_u2 = Eq(u2(t).diff(t), v2(t))
    eq_v1 = Eq(v1(t).diff(t), -2 * u1(t) + u2(t))
    eq_v2 = Eq(v2(t).diff(t), -2 * u2(t) + u1(t))

    result = dsolve(
        [eq_u1, eq_u2, eq_v1, eq_v2],
        [u1(t), u2(t), v1(t), v2(t)],
        ics=initial_conditions,
        )
    print(result)

if __name__ == '__main__':
    solve_two_mass_spring_system()