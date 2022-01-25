import numpy as np

from utility import l2_norm

def run_cps_simulation(left_system, left_solver, right_system, right_solver, t_end, N, **kwargs):
    if "sc" in kwargs:
        sc = kwargs['sc']
    else:
        sc = 1
    dt = t_end / N
    dt_sc = dt/sc
    t = 0
    right_iter = 0
    left_iter = 0
    while right_iter < N:
        for i in range(sc):
            left_system.result[:, left_iter + 1] = np.squeeze(left_solver.compute_timestep(dt_sc, t + i*dt_sc, left_system.result[:, left_iter]))
            left_iter += 1
        right_system.result[:, right_iter + 1] = np.squeeze(right_solver.compute_timestep(dt, t, right_system.result[:, right_iter]))
        right_iter += 1
        t += dt
        left_system.other_u[right_iter] = right_system.result[0, right_iter]
        right_system.other_u[left_iter-(sc-1):left_iter+1] = left_system.result[0, left_iter-(sc-1):left_iter+1]
    return left_system.result[:,::sc], right_system.result

def run_css_simulation(left_system, left_solver, right_system, right_solver, t_end, N, **kwargs):
    """left system does the first step"""
    if "sc" in kwargs:
        sc = kwargs['sc']
    else:
        sc = 1
    dt = t_end / N
    dt_subcycling = dt/sc
    t = 0
    right_iter = 0
    left_iter = 0
    while right_iter < N:
        for i in range(sc):
            left_system.result[:, left_iter + 1] = np.squeeze(left_solver.compute_timestep(dt_subcycling, t + i*dt_subcycling, left_system.result[:, left_iter]))
            left_iter += 1
        right_system.other_u[left_iter-(sc-1):left_iter+1] = left_system.result[0, left_iter-(sc-1):left_iter+1]
        right_system.result[:, right_iter + 1] = np.squeeze(right_solver.compute_timestep(dt, t, right_system.result[:, right_iter]))
        right_iter += 1
        t += dt
        left_system.other_u[right_iter] = right_system.result[0, right_iter]
    return left_system.result[:,::sc], right_system.result

def run_implicit_cps_simulation(left_system, left_solver, right_system, right_solver, t_end, N, **kwargs):
    if "sc" in kwargs:
        sc = kwargs['sc']
    else:
        sc = 1
    tol = 1e-8
    max_iters = 10
    dt = t_end / N
    dt_sc = dt/sc
    t = 0
    right_iter = 0
    left_iter = 0
    while right_iter < N:
        fp_iter = 0
        res = np.inf
        while (fp_iter < max_iters) and (res > tol):
            left_temp = np.full((left_system.result.shape[0], sc + 1), np.inf)
            left_temp[:, 0] = left_system.result[:, left_iter]
            for i in range(sc):
                left_temp[:, i + 1] = np.squeeze(left_solver.compute_timestep(dt_sc, t + i*dt_sc, left_temp[:, i]))
            right_temp = np.squeeze(right_solver.compute_timestep(dt, t, right_system.result[:, right_iter]))
            res = l2_norm(left_temp[:,-1] - left_system.result[:, left_iter + sc]) + l2_norm(right_temp - right_system.result[:, right_iter + 1])
            # print(f"res: {res}, iter: {fp_iter}")
            # print(f"new val: {repr(left_temp[:,-1])}")
            # print(f"old val: {repr(left_system.result[:, left_iter + sc])}")
            # print(f"diff: {repr(left_temp[:,-1] - left_system.result[:, left_iter + sc])}")
            # print(f"updating elements {right_iter + 1} and {list(range(left_iter,left_iter+1+sc))}")
            assert(np.sum(np.abs(left_system.result[:,left_iter]-left_temp[:,0])) == 0)
            left_system.result[:, left_iter:(left_iter+1+sc)] = left_temp.copy()
            right_system.result[:, right_iter + 1] = right_temp
            left_system.other_u[right_iter + 1] = right_temp[0]
            right_system.other_u[left_iter:(left_iter+1+sc)] = left_temp[0].copy()
            fp_iter += 1
        right_iter += 1
        left_iter += sc
        t += dt
    assert(left_iter == sc * N)
    assert(right_iter == N)
    assert(np.sum(np.abs(left_system.result[0] - right_system.other_u)) == 0)
    assert(np.sum(np.abs(right_system.result[0] - left_system.other_u)) == 0)
    assert(np.abs(t - t_end) < 1e-10)
    return left_system.result[:,::sc], right_system.result


def run_strang_simulation(left_system, left_solver, right_system, right_solver, t_end, N, **kwargs):
    """left system = f1, right system = f2"""
    del kwargs
    dt = t_end / N
    t = 0
    right_iter = 0
    left_iter = 0
    while right_iter < N:
        left_system.result[:, left_iter + 1] = np.squeeze(left_solver.compute_timestep(dt/2, t, left_system.result[:, left_iter]))
        left_iter += 1
        right_system.other_u[right_iter + 1] = left_system.result[0, left_iter]
        right_system.result[:, right_iter + 1] = np.squeeze(right_solver.compute_timestep(dt, t, right_system.result[:, right_iter]))
        right_iter += 1
        left_system.other_u[left_iter] = right_system.result[0, right_iter]
        left_system.other_u[left_iter + 1] = right_system.result[0, right_iter]
        left_system.result[:, left_iter + 1] = np.squeeze(left_solver.compute_timestep(dt/2, t + dt/2, left_system.result[:, left_iter]))
        left_iter += 1
        t += dt
    return left_system.result[:, ::2], right_system.result
