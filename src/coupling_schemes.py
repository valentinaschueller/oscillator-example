import numpy as np

def run_cps_simulation(left_system, left_solver, right_system, right_solver, t_end, N):
    dt = t_end / N
    t = 0
    iter = 0
    while iter < N:
        left_system.result[:, iter + 1] = np.squeeze(left_solver.compute_timestep(dt, t, left_system.result[:, iter]))
        right_system.result[:, iter + 1] = np.squeeze(right_solver.compute_timestep(dt, t, right_system.result[:, iter]))
        iter += 1
        t += dt
        left_system.other_u = right_system.result[0, iter]
        right_system.other_u = left_system.result[0, iter]
    return left_system.result, right_system.result

# def run_implicit_simulation(left_system, left_solver, right_system, right_solver, t_end, N):
#     tol = 1e-4
#     max_steps = 10
#     tol_
#     dt = t_end / N
#     t = 0
#     iter = 0
#     while iter < N:
#         # do left and right step
        
#         # check if both are 
#         left_system.result[:, iter + 1] = np.squeeze(left_solver.compute_timestep(dt, t, left_system.result[:, iter]))
#         right_system.result[:, iter + 1] = np.squeeze(right_solver.compute_timestep(dt, t, right_system.result[:, iter]))
#         iter += 1
#         t += dt
#         left_system.other_u = right_system.result[0, iter]
#         right_system.other_u = left_system.result[0, iter]
#     return left_system.result, right_system.result

def run_css_simulation(left_system, left_solver, right_system, right_solver, t_end, N):
    """left system does the first step"""
    dt = t_end / N
    t = 0
    iter = 0
    while iter < N:
        left_system.result[:, iter + 1] = np.squeeze(left_solver.compute_timestep(dt, t, left_system.result[:, iter]))
        right_system.other_u = left_system.result[0, iter + 1]
        right_system.result[:, iter + 1] = np.squeeze(right_solver.compute_timestep(dt, t, right_system.result[:, iter]))
        left_system.other_u = right_system.result[0, iter + 1]
        iter += 1
        t += dt
    return left_system.result, right_system.result

def run_strang_simulation(left_system, left_solver, right_system, right_solver, t_end, N):
    """left system = f1, right system = f2"""
    dt = t_end / N
    t = 0
    right_iter = 0
    left_iter = 0
    while right_iter < N:
        left_system.result[:, left_iter + 1] = np.squeeze(left_solver.compute_timestep(dt/2, t, left_system.result[:, left_iter]))
        left_iter += 1
        right_system.other_u = left_system.result[0, left_iter]
        right_system.result[:, right_iter + 1] = np.squeeze(right_solver.compute_timestep(dt, t, right_system.result[:, right_iter]))
        right_iter += 1
        left_system.other_u = right_system.result[0, right_iter]
        left_system.result[:, left_iter + 1] = np.squeeze(left_solver.compute_timestep(dt/2, t, left_system.result[:, left_iter]))
        left_iter += 1
        t += dt
    return left_system.result[:, ::2], right_system.result
