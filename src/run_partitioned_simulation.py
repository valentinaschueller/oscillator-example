from system_partition import SameTimescales
from coupling_schemes import *
from timestepping import *

def partitioned_newmark_beta(t_end: float, N: int, coupling_scheme_str: str = ""):
    if coupling_scheme_str == "css":
        run_simulation = run_css_simulation
        left_system = SameTimescales(left_system=True, result_shape=(3, N+1))
        right_system = SameTimescales(left_system=False, result_shape=(3, N+1))
    elif coupling_scheme_str == "cps":
        run_simulation = run_cps_simulation
        left_system = SameTimescales(left_system=True, result_shape=(3, N+1))
        right_system = SameTimescales(left_system=False, result_shape=(3, N+1))
    elif coupling_scheme_str == "strang":
        run_simulation = run_strang_simulation
        left_system = SameTimescales(left_system=True, result_shape=(3, 2*N+1))
        right_system = SameTimescales(left_system=False, result_shape=(3, N+1))
    elif coupling_scheme_str == "implicit-cps":
        run_simulation = run_implicit_cps_simulation
        left_system = SameTimescales(left_system=True, result_shape=(3, N+1))
        right_system = SameTimescales(left_system=False, result_shape=(3, N+1))
    elif coupling_scheme_str == "implicit-css":
        run_simulation = run_implicit_css_simulation
        left_system = SameTimescales(left_system=True, result_shape=(3, N+1))
        right_system = SameTimescales(left_system=False, result_shape=(3, N+1))
    else:
        raise NotImplementedError(f"Coupling scheme {coupling_scheme_str} not implemented!")
    gamma = 0.5
    beta = 0.25
    # create solvers
    solver_left = NewmarkBeta(
        left_system.A_second_order, left_system.M, left_system.K,
        beta, gamma, left_system.second_order_force
    )
    solver_right = NewmarkBeta(
        right_system.A_second_order, right_system.M, right_system.K,
        beta, gamma, right_system.second_order_force
    )
    # run simulation
    left_result, right_result = run_simulation(left_system, solver_left, right_system, solver_right, t_end, N)
    full_result = np.array(
        [left_result[0],
        right_result[0],
        left_result[1],
        right_result[1]]
    )
    return full_result

def partitioned_generalized_alpha(t_end: float, N: int, coupling_scheme_str: str = ""):
    if coupling_scheme_str == "css":
        run_simulation = run_css_simulation
        left_system = SameTimescales(left_system=True, result_shape=(3, N+1))
        right_system = SameTimescales(left_system=False, result_shape=(3, N+1))
    elif coupling_scheme_str == "cps":
        run_simulation = run_cps_simulation
        left_system = SameTimescales(left_system=True, result_shape=(3, N+1))
        right_system = SameTimescales(left_system=False, result_shape=(3, N+1))
    elif coupling_scheme_str == "strang":
        run_simulation = run_strang_simulation
        left_system = SameTimescales(left_system=True, result_shape=(3, 2*N+1))
        right_system = SameTimescales(left_system=False, result_shape=(3, N+1))
    elif coupling_scheme_str == "implicit-cps":
        run_simulation = run_implicit_cps_simulation
        left_system = SameTimescales(left_system=True, result_shape=(3, N+1))
        right_system = SameTimescales(left_system=False, result_shape=(3, N+1))
    elif coupling_scheme_str == "implicit-css":
        run_simulation = run_implicit_css_simulation
        left_system = SameTimescales(left_system=True, result_shape=(3, N+1))
        right_system = SameTimescales(left_system=False, result_shape=(3, N+1))
    else:
        raise NotImplementedError(f"Coupling scheme {coupling_scheme_str} not implemented!")
    alpha_m = 0.2
    alpha_f = 0.5
    gamma = 0.5 - alpha_m + alpha_f
    beta = 0.25 * (gamma + 0.5)**2
    # create solvers
    solver_left = GeneralizedAlpha(
        left_system.A_second_order, left_system.M, left_system.K,
        beta, gamma, alpha_f, alpha_m, left_system.second_order_force
    )
    solver_right = GeneralizedAlpha(
        right_system.A_second_order, right_system.M, right_system.K,
        beta, gamma, alpha_f, alpha_m, right_system.second_order_force
    )
    # run simulation
    left_result, right_result = run_simulation(left_system, solver_left, right_system, solver_right, t_end, N)
    full_result = np.array(
        [left_result[0],
        right_result[0],
        left_result[1],
        right_result[1]]
    )
    return full_result

def partitioned_erk(t_end: float, N: int, order: int = 1, coupling_scheme_str: str = ""):
    if coupling_scheme_str == "css":
        run_simulation = run_css_simulation
        left_system = SameTimescales(left_system=True, result_shape=(2, N+1))
        right_system = SameTimescales(left_system=False, result_shape=(2, N+1))
    elif coupling_scheme_str == "cps":
        run_simulation = run_cps_simulation
        left_system = SameTimescales(left_system=True, result_shape=(2, N+1))
        right_system = SameTimescales(left_system=False, result_shape=(2, N+1))
    elif coupling_scheme_str == "strang":
        run_simulation = run_strang_simulation
        left_system = SameTimescales(left_system=True, result_shape=(2, 2*N+1))
        right_system = SameTimescales(left_system=False, result_shape=(2, N+1))
    elif coupling_scheme_str == "implicit-cps":
        run_simulation = run_implicit_cps_simulation
        left_system = SameTimescales(left_system=True, result_shape=(2, N+1))
        right_system = SameTimescales(left_system=False, result_shape=(2, N+1))
    elif coupling_scheme_str == "implicit-css":
        run_simulation = run_implicit_css_simulation
        left_system = SameTimescales(left_system=True, result_shape=(2, N+1))
        right_system = SameTimescales(left_system=False, result_shape=(2, N+1))
    else:
        raise NotImplementedError(f"Coupling scheme {coupling_scheme_str} not implemented!")
    # create solvers
    solver_left = ERK(left_system.A_first_order, left_system.first_order_force, order)
    solver_right = ERK(right_system.A_first_order, right_system.first_order_force, order)
    left_result, right_result = run_simulation(left_system, solver_left, right_system, solver_right, t_end, N)
    full_result = np.array(
        [left_result[0],
        right_result[0],
        left_result[1],
        right_result[1]]
    )
    return full_result
