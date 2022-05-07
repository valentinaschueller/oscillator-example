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
    **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    run a partitioned simulation with the conventional parallel staggered (CPS) scheme

    both partitions are updated using the interface values from the previous time step.
    In theory, the computations could thus happen in parallel (although they are sequential here).

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
    n = 0
    while n < N:
        partition_1.result[:, n + 1] = np.squeeze(
            solver_1.compute_timestep(dt, t, partition_1.result[:, n])
        )
        partition_2.result[:, n + 1] = np.squeeze(
            solver_2.compute_timestep(dt, t, partition_2.result[:, n])
        )
        n += 1
        t += dt
        # update the interface values handed to the partitions *after* both computations are done
        partition_1.other_u[n] = partition_2.result[0, n]
        partition_2.other_u[n] = partition_1.result[0, n]
    return partition_1.result, partition_2.result


def run_css_simulation(
    partition_1: SystemPartition,
    solver_1: TimesteppingMethod,
    partition_2: SystemPartition,
    solver_2: TimesteppingMethod,
    t_end: float,
    N: int,
    **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    run a partitioned simulation with the conventional serial staggered (CSS) scheme

    partition_1 does the first step, partition_2 uses the newly computed interface value.

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

    n = 0
    while n < N:
        partition_1.result[:, n + 1] = np.squeeze(
            solver_1.compute_timestep(dt, t, partition_1.result[:, n])
        )
        # update interface value for partition_2
        partition_2.other_u[n + 1] = partition_1.result[0, n + 1]
        partition_2.result[:, n + 1] = np.squeeze(
            solver_2.compute_timestep(dt, t, partition_2.result[:, n])
        )
        # update interface value for partition_1
        partition_1.other_u[n + 1] = partition_2.result[0, n + 1]
        n += 1
        t += dt
    return partition_1.result, partition_2.result


def run_implicit_cps_simulation(
    partition_1: SystemPartition,
    solver_1: TimesteppingMethod,
    partition_2: SystemPartition,
    solver_2: TimesteppingMethod,
    t_end: float,
    N: int,
    **kwargs
):
    """
    run a partitioned simulation with the implicit conventional parallel staggered (CPS) scheme

    Implicit --> fixed-point iterations are used until the interface values converge between iterations.
    both partitions are updated using the interface values from the previous time step.
    In theory, the computations could thus happen in parallel (although they are sequential here).

    Set convergence tolerance using optional param 'tol'
    Set max. amount of fixed-point iterations using optional param 'max_iters'

    :param partition_1: the first partition
    :param solver_1: the timestepping method associated with partition_1
    :param partition_2: the second system partition
    :param solver_2: the timestepping method associated with partition_2
    :param t_end: the simulation is executed until t >= t_end
    :param N: the number of time steps
    :returns: a tuple containing the result of partition_1 and partition_2 (N values)
    """

    tol = kwargs.get("tol", 1e-8)
    max_iters = kwargs.get("max_iters", 100)  # larger number of iterations is needed for waveform iterations

    dt = t_end / N
    t = 0
    n = 0

    while n < N:
        k = 0
        res = 1
        while (k < max_iters) and (res > tol):
            # do left and right step
            temp_1 = np.squeeze(
                solver_1.compute_timestep(dt, t, partition_1.result[:, n])
            )
            temp_2 = np.squeeze(
                solver_2.compute_timestep(dt, t, partition_2.result[:, n])
            )
            res = l2_norm(temp_1 - partition_1.result[:, n + 1]) + l2_norm(
                temp_2 - partition_2.result[:, n + 1]
            )
            partition_1.result[:, n + 1] = temp_1
            partition_2.result[:, n + 1] = temp_2
            partition_1.other_u[n + 1] = temp_2[0]
            partition_2.other_u[n + 1] = temp_1[0]
            k += 1
        n += 1
        t += dt
        if(k == max_iters):
            print("WARNING!")
            print("dt={}".format(dt))
    return partition_1.result, partition_2.result


def run_strang_simulation(
    partition_1: SystemPartition,
    solver_1: TimesteppingMethod,
    partition_2: SystemPartition,
    solver_2: TimesteppingMethod,
    t_end: float,
    N: int,
    **kwargs
):
    """
    run a partitioned simulation with Strang splitting

    1. partition_1 does a half dt/2 step
    2. partition_2 does a full dt step
    3. partition_1 does a half dt/2 step

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

    n_1 = 0
    n_2 = 0
    while n_2 < N:
        # half step
        partition_1.result[:, n_1 + 1] = np.squeeze(
            solver_1.compute_timestep(dt / 2, t, partition_1.result[:, n_1])
        )
        n_1 += 1
        # update interface value for full step
        partition_2.other_u[n_2 + 1] = partition_1.result[0, n_1]
        # full step
        partition_2.result[:, n_2 + 1] = np.squeeze(
            solver_2.compute_timestep(dt, t, partition_2.result[:, n_2])
        )
        n_2 += 1
        # update interface value for the next two half steps
        partition_1.other_u[n_1] = partition_2.result[0, n_2]
        partition_1.other_u[n_1 + 1] = partition_2.result[0, n_2]
        # half step
        partition_1.result[:, n_1 + 1] = np.squeeze(
            solver_1.compute_timestep(dt / 2, t + dt / 2, partition_1.result[:, n_1])
        )
        n_1 += 1
        t += dt
    return partition_1.result[:, ::2], partition_2.result
