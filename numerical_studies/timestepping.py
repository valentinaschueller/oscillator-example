"""
Interface and implementations of time integration methods.
"""

from abc import ABC, abstractmethod

import numpy as np


class TimesteppingMethod(ABC):
    """
    Interface class for one-step time integration methods.
    """

    @abstractmethod
    def compute_timestep(
        self, dt: float, t_n: float, last_values: np.ndarray
    ) -> np.ndarray:
        """
        do a single time step.

        :param dt: time step size
        :param t_n: current simulation time
        :param last_values: values of the unknowns at time t_n
        :returns: new values of the unknowns at time t_n + dt
        """
        pass

    def integrate(self, dt: float, N: int, io_array: np.ndarray) -> None:
        """
        integrate from 0 to N*dt.

        repeatedly calls compute_timestep and stores the values in io_array.

        :param dt: time step size
        :param N: number of time steps that shall be executed
        :param io_array: preallocated array to store the results in. (should be N+1 long and contain an initial condition)
        """
        t_n = 0.0
        for n in range(0, N):
            io_array[:, n + 1] = self.compute_timestep(dt, t_n, io_array[:, n])
            t_n = t_n + dt


class GeneralizedAlpha(TimesteppingMethod):
    """
    Generalized-alpha method for a system Mu'' + Ku = F.
    """

    def __init__(
        self,
        A_second_order: np.ndarray,
        M: np.ndarray,
        K: np.ndarray,
        beta: float,
        gamma: float,
        alpha_f: float,
        alpha_m: float,
        F: callable,
    ):
        # for the formulation: u'' =  Au:
        self.A = A_second_order
        # for the formulation: Mu'' + Ku = 0:
        self.M = M
        self.K = K
        self.beta = beta
        self.gamma = gamma
        self.alpha_f = alpha_f
        self.alpha_m = alpha_m
        self.F = F

    def compute_timestep(self, dt: float, t_n: float, last_values: np.ndarray):
        if last_values.size == 6:
            # for the monoltihic system
            u_n = last_values[[0, 1]]
            v_n = last_values[[2, 3]]
            a_n = last_values[[4, 5]]
        elif last_values.size == 3:
            # for the partitioned system
            u_n = last_values[0]
            v_n = last_values[1]
            a_n = last_values[2]
        else:
            raise ValueError("Unknown shape of last_values?")

        force = self.alpha_f * self.F(t_n, t_n) + (1 - self.alpha_f) * self.F(
            t_n + dt, t_n
        )

        # solve for u_next
        m1 = (1 - self.alpha_m) / (self.beta * dt**2)
        m2 = (1 - self.alpha_m) / (self.beta * dt)
        m3 = (1 - self.alpha_m - 2 * self.beta) / (2 * self.beta)
        k1 = 1 - self.alpha_f
        rhs = (
            force
            - self.alpha_f * np.dot(self.K, u_n)
            + np.dot(self.M, m1 * u_n + m2 * v_n + m3 * a_n)
        )
        system_matrix = k1 * self.K + m1 * self.M
        u_next = np.linalg.solve(system_matrix, rhs)

        # compute a_next
        a_next = (
            1.0 / (self.beta * dt**2) * (u_next - u_n - dt * v_n)
            - (1 - 2 * self.beta) / (2 * self.beta) * a_n
        )

        # compute v_next
        v_next = v_n + dt * ((1 - self.gamma) * a_n + self.gamma * a_next)

        return np.concatenate([u_next, v_next, a_next])


class NewmarkBeta(GeneralizedAlpha):
    """
    Newmark-beta method for a system Mu'' + Ku = F
    """

    def __init__(
        self,
        A_second_order: np.ndarray,
        M: np.ndarray,
        K: np.ndarray,
        beta: float,
        gamma: float,
        F: callable,
    ):
        super().__init__(A_second_order, M, K, beta, gamma, 0.0, 0.0, F)


class ERK(TimesteppingMethod):
    """
    Explicit Runge-Kutta methods.

    can be instantiated by providing the order.
    supported:
    - order=1: explicit Euler
    - order=2: Heun's method
    - order=4: classical Runge-Kutta method
    """

    def __init__(
        self,
        first_order_matrix: np.ndarray,
        force_function: callable = None,
        order: int = 1,
    ):
        if not force_function:
            self.force_function = lambda t: 0 * t
        else:
            self.force_function = force_function
        # for the formulation: y' = Ay:
        self.first_order_matrix = first_order_matrix
        if order == 1:
            self.erk_step = self._erk1
        elif order == 2:
            self.erk_step = self._erk2
        elif order == 4:
            self.erk_step = self._erk4
        else:
            raise NotImplementedError(
                f"Currently only supports order 1,2,4, not {order}!"
            )

    def compute_timestep(self, dt: float, t_n: float, last_values: np.ndarray):
        next_values = self.erk_step(last_values, dt, t_n)
        return next_values

    def du_dt(self, last_values: np.ndarray, t_n: float, t_lower: float) -> np.ndarray:
        return np.dot(self.first_order_matrix, last_values) + self.force_function(
            t_n, t_lower
        )

    def _erk1(self, u_i: np.ndarray, dt: float, t_i=0) -> np.ndarray:
        """Explicit Euler method"""
        A = np.array([0])
        b = np.array([0])
        c = np.array([1])
        u_next = self._erk_gen(A, b, c, u_i, dt, t_i)
        return u_next

    def _erk2(self, u_i: np.ndarray, dt: float, t_i=None) -> np.ndarray:
        """Method of Heun (2nd order accurate)"""
        A = np.array([[0, 0], [1, 0]])
        b = np.array([0, 1])
        c = np.array([0.5, 0.5])
        u_next = self._erk_gen(A, b, c, u_i, dt, t_i)
        return u_next

    def _erk4(self, u_i: np.ndarray, dt: float, t_i=None) -> np.ndarray:
        """RK4 scheme"""
        A = np.zeros((4, 4), dtype=float)
        A[1, 0] = 0.5
        A[2, 1] = 0.5
        A[3, 2] = 1
        b = np.array([0, 0.5, 0.5, 1])
        c = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        u_next = self._erk_gen(A, b, c, u_i, dt, t_i)
        return u_next

    def _erk_gen(
        self, A: np.ndarray, b, c, u_i: np.ndarray, dt: float, t_i: float
    ) -> np.ndarray:
        """
        implements the general explicit Runge-Kutta step.
        """
        k = np.zeros(b.shape + u_i.shape, dtype=u_i.dtype)
        k[0] = self.du_dt(u_i, t_i, t_i)
        if len(b) > 1:
            for i in range(1, len(b)):
                u_eval = u_i + dt * np.tensordot(A[i], k, axes=1)
                t_eval = t_i + b[i] * dt
                try:
                    k[i] = self.du_dt(u_eval, t_eval, t_i)
                except ValueError:
                    k[i] = self.du_dt(u_eval, t_eval, t_i)[i]
        u_next = u_i + dt * np.tensordot(c, k, axes=1)
        return u_next


class ImplicitMidpoint(TimesteppingMethod):
    """
    implicit midpoint method for a system y'=Ay.

    O(dt**2), symplectic.
    """

    def __init__(
        self,
        first_order_matrix: np.ndarray,
        force_function: callable = None,
    ):
        if not force_function:
            self.force_function = lambda t: 0 * t
        else:
            self.force_function = force_function
        # for the formulation: y' = Ay:
        self.A = first_order_matrix

    def compute_timestep(self, dt: float, t_n: float, last_values: np.ndarray):
        # rhs: (I+0.5*dt*A) * last_values + b = R * last_values + b
        # compute contribution of force function to rhs
        b = dt * self.force_function(t_n + dt / 2, t_n)
        R = np.eye(*self.A.shape) + 0.5 * dt * self.A
        rhs = np.dot(R, last_values) + b
        # finish system and solve: L * next_values = rhs
        L = np.eye(*self.A.shape) - 0.5 * dt * self.A
        next_values = np.linalg.solve(L, rhs.astype(float))
        return next_values


class SemiImplicitEuler(TimesteppingMethod):
    """
    semi-implicit Euler method/symplectic Euler method/Euler-A method

    O(dt), symplectic.
    https://en.wikipedia.org/wiki/Semi-implicit_Euler_method
    """

    def __init__(
        self,
        A: np.ndarray,
        F: callable,
    ):
        # for the formulation: y' = Ay + b, where b = [0,F]^T, A = [0, I; K/M, 0]:
        half_size = int(len(A) / 2)
        # extract lower left subblock of A
        self.KM = A[half_size:, :half_size]
        self.F = F

    def compute_timestep(self, dt: float, t_n: float, last_values: np.ndarray):
        if last_values.size == 4:
            u_n = last_values[[0, 1]]
            v_n = last_values[[2, 3]]
        elif last_values.size == 2:
            u_n = last_values[0]
            v_n = last_values[1]
        else:
            raise ValueError("Unknown shape of last_values?")
        v_next = v_n + dt * (np.dot(self.KM, u_n) + self.F(t_n, t_n))
        u_next = u_n + dt * v_next
        return np.concatenate([u_next, v_next])
