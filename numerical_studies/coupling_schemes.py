import numpy as np
from system_partition import SystemPartition
from timestepping import TimesteppingMethod
from utility import l2_norm


def run_cps_simulation(
    partition_1: SystemPartition,
    solver_1: TimesteppingMethod,
    partition_2: SystemPartition,
    solver_2: TimesteppingMethod,
    t_end: float,
    N: int,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    run a partitioned simulation with the conventional parallel staggered (CPS) scheme

    both partitions are updated using the interface values from the previous time step.
    In theory, the computations could thus happen in parallel (although they are sequential here).

    supports subcycling using the optional parameter 'sc', denoting the subcycling factor.
    If sc > 1: partition_1 does sc*N time steps (partition_2 does N time steps).

    :param partition_1: the first partition
    :param solver_1: the timestepping method associated with partition_1
    :param partition_2: the second system partition
    :param solver_2: the timestepping method associated with partition_2
    :param t_end: the simulation is executed until t >= t_end
    :param N: the number of time steps
    :returns: a tuple containing the result of partition_1 and partition_2 (N values)
    :raises ValueError: if subcycling factor is invalid
    """
    sc = kwargs.get("sc", 1)
    if type(sc) not in (int, np.int64) or sc < 1:
        raise ValueError("subcycling factor must be a positive integer")

    dt = t_end / N
    dt_sc = dt / sc
    t = 0

    iter_1 = 0
    iter_2 = 0
    while iter_2 < N:
        # partition_1 does sc time steps per time window
        for i in range(sc):
            partition_1.result[:, iter_1 + 1] = np.squeeze(
                solver_1.compute_timestep(
                    dt_sc, t + i * dt_sc, partition_1.result[:, iter_1]
                )
            )
            iter_1 += 1
        # partition_2 does 1 time step per time window
        partition_2.result[:, iter_2 + 1] = np.squeeze(
            solver_2.compute_timestep(dt, t, partition_2.result[:, iter_2])
        )
        iter_2 += 1
        # update the interface values handed to the partitions *after* both computations are done
        # note: partition_1 provides 'sc' interface values in each time step (as opposed to one)
        partition_1.other_u[iter_2] = partition_2.result[0, iter_2]
        partition_2.other_u[iter_1 - (sc - 1) : iter_1 + 1] = partition_1.result[
            0, iter_1 - (sc - 1) : iter_1 + 1
        ]
        t += dt
    return partition_1.result[:, ::sc], partition_2.result


def run_css_simulation(
    partition_1: SystemPartition,
    solver_1: TimesteppingMethod,
    partition_2: SystemPartition,
    solver_2: TimesteppingMethod,
    t_end: float,
    N: int,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    run a partitioned simulation with the conventional serial staggered (CSS) scheme

    partition_1 does the first step, partition_2 uses the newly computed interface value(s).

    supports subcycling using the optional parameter 'sc', denoting the subcycling factor.
    If sc > 1: partition_1 does sc*N time steps (partition_2 does N time steps).

    :param partition_1: the first partition
    :param solver_1: the timestepping method associated with partition_1
    :param partition_2: the second system partition
    :param solver_2: the timestepping method associated with partition_2
    :param t_end: the simulation is executed until t >= t_end
    :param N: the number of time steps
    :returns: a tuple containing the result of partition_1 and partition_2 (N values)
    :raises ValueError: if subcycling factor is invalid
    """
    sc = kwargs.get("sc", 1)
    if type(sc) not in (int, np.int64) or sc < 1:
        raise ValueError("subcycling factor must be a positive integer")

    dt = t_end / N
    dt_subcycling = dt / sc
    t = 0

    iter_1 = 0
    iter_2 = 0
    while iter_2 < N:
        # partition_1 does sc time steps per time window
        for i in range(sc):
            partition_1.result[:, iter_1 + 1] = np.squeeze(
                solver_1.compute_timestep(
                    dt_subcycling,
                    t + i * dt_subcycling,
                    partition_1.result[:, iter_1],
                )
            )
            iter_1 += 1
        # update interface values for partition_2 (sc values)
        partition_2.other_u[iter_1 - (sc - 1) : iter_1 + 1] = partition_1.result[
            0, iter_1 - (sc - 1) : iter_1 + 1
        ]
        # partition_2 does 1 time step per time window
        partition_2.result[:, iter_2 + 1] = np.squeeze(
            solver_2.compute_timestep(dt, t, partition_2.result[:, iter_2])
        )
        iter_2 += 1
        # update interface value for partition_1 (1 value)
        partition_1.other_u[iter_2] = partition_2.result[0, iter_2]
        t += dt
    return partition_1.result[:, ::sc], partition_2.result


def run_implicit_cps_simulation(
    partition_1: SystemPartition,
    solver_1: TimesteppingMethod,
    partition_2: SystemPartition,
    solver_2: TimesteppingMethod,
    t_end: float,
    N: int,
    **kwargs,
):
    """
    run a partitioned simulation with the implicit conventional parallel staggered (CPS) scheme

    Implicit --> fixed-point iterations are used until the interface values converge between iterations.
    both partitions are updated using the interface values from the previous time step.
    In theory, the computations could thus happen in parallel (although they are sequential here).

    supports subcycling using the optional parameter 'sc', denoting the subcycling factor.
    If sc > 1: partition_1 does sc*N time steps (partition_2 does N time steps).

    Set convergence tolerance using optional param 'tol'
    Set max. amount of fixed-point iterations using optional param 'max_iters'

    :param partition_1: the first partition
    :param solver_1: the timestepping method associated with partition_1
    :param partition_2: the second system partition
    :param solver_2: the timestepping method associated with partition_2
    :param t_end: the simulation is executed until t >= t_end
    :param N: the number of time steps
    :returns: a tuple containing the result of partition_1 and partition_2 (N values)
    :raises ValueError: if subcycling factor is invalid
    """
    sc = kwargs.get("sc", 1)
    if type(sc) not in (int, np.int64) or sc < 1:
        raise ValueError("subcycling factor must be a positive integer")

    tol = kwargs.get("tol", 1e-8)
    max_iters = kwargs.get("max_iters", 10)

    dt = t_end / N
    dt_sc = dt / sc
    t = 0

    iter_1 = 0
    iter_2 = 0
    while iter_2 < N:
        fixed_point_iter = 0
        res = np.inf
        while (fixed_point_iter < max_iters) and (res > tol):
            # compute temporary new values of partition_1
            # partition_1 does sc steps per time window
            temp_1 = np.full((partition_1.result.shape[0], sc + 1), np.inf)
            temp_1[:, 0] = partition_1.result[:, iter_1]
            for i in range(sc):
                temp_1[:, i + 1] = np.squeeze(
                    solver_1.compute_timestep(dt_sc, t + i * dt_sc, temp_1[:, i])
                )
            # compute temporary new value of partition_2
            # partition_1 does 1 step per time window
            temp_2 = np.squeeze(
                solver_2.compute_timestep(dt, t, partition_2.result[:, iter_2])
            )
            # compute residual: l2 norm of result difference between previous and current iteration
            # evaluated at the end of the time window
            res_1 = l2_norm(temp_1[:, -1] - partition_1.result[:, iter_1 + sc])
            res_2 = l2_norm(temp_2 - partition_2.result[:, iter_2 + 1])
            res = res_1 + res_2
            # copy temporary values to result arrays
            partition_1.result[:, iter_1 : (iter_1 + 1 + sc)] = temp_1.copy()
            partition_2.result[:, iter_2 + 1] = temp_2
            # update interface values of partitions
            partition_1.other_u[iter_2 + 1] = temp_2[0]
            partition_2.other_u[iter_1 : (iter_1 + 1 + sc)] = temp_1[0].copy()
            fixed_point_iter += 1
        iter_1 += sc
        iter_2 += 1
        t += dt
    # some assertions to make sure we didn't do anything wrong
    assert iter_1 == sc * N
    assert iter_2 == N
    assert np.sum(np.abs(partition_1.result[0] - partition_2.other_u)) == 0
    assert np.sum(np.abs(partition_2.result[0] - partition_1.other_u)) == 0
    assert np.abs(t - t_end) < 1e-10
    return partition_1.result[:, ::sc], partition_2.result


def run_strang_simulation(
    partition_1: SystemPartition,
    solver_1: TimesteppingMethod,
    partition_2: SystemPartition,
    solver_2: TimesteppingMethod,
    t_end: float,
    N: int,
    **kwargs,
):
    """
    run a partitioned simulation with Strang splitting

    1. partition_1 does a half dt/2 step
    2. partition_2 does a full dt step
    3. partition_1 does a half dt/2 step

    does not support subcycling!

    :param partition_1: the first partition
    :param solver_1: the timestepping method associated with partition_1
    :param partition_2: the second system partition
    :param solver_2: the timestepping method associated with partition_2
    :param t_end: the simulation is executed until t >= t_end
    :param N: the number of time steps
    :returns: a tuple containing the result of partition_1 and partition_2 (N values)
    """
    del kwargs

    dt = t_end / N
    t = 0

    iter_1 = 0
    iter_2 = 0
    while iter_2 < N:
        # half step
        partition_1.result[:, iter_1 + 1] = np.squeeze(
            solver_1.compute_timestep(dt / 2, t, partition_1.result[:, iter_1])
        )
        iter_1 += 1
        # update interface value for full step
        partition_2.other_u[iter_2 + 1] = partition_1.result[0, iter_1]
        # full step
        partition_2.result[:, iter_2 + 1] = np.squeeze(
            solver_2.compute_timestep(dt, t, partition_2.result[:, iter_2])
        )
        iter_2 += 1
        # update interface value for this step *and* the next step
        partition_1.other_u[iter_1] = partition_2.result[0, iter_2]
        partition_1.other_u[iter_1 + 1] = partition_2.result[0, iter_2]
        # half step
        partition_1.result[:, iter_1 + 1] = np.squeeze(
            solver_1.compute_timestep(dt / 2, t + dt / 2, partition_1.result[:, iter_1])
        )
        iter_1 += 1
        t += dt
    return partition_1.result[:, ::2], partition_2.result
