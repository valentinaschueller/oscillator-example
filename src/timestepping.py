from abc import ABC, abstractmethod

import numpy as np

class TimesteppingMethod(ABC):
    @abstractmethod
    def compute_timestep(self, dt: float, t_n: float, last_values: np.ndarray):
        pass

    def integrate(self, dt: float, N: int, io_array: np.ndarray) -> None:
        t_n = 0.
        for n in range(0, N):
            io_array[:, n+1] = self.compute_timestep(dt, t_n, io_array[:, n])
            t_n = t_n + dt


class GeneralizedAlpha(TimesteppingMethod):
    def __init__(self,
                A_second_order: np.ndarray, M: np.ndarray, K: np.ndarray, 
                beta: float, gamma: float,
                alpha_f: float, alpha_m: float,
                F: callable):
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
        u_n = last_values[[0, 1]]
        v_n = last_values[[2, 3]]
        a_n = last_values[[4, 5]]
        force = self.F(t_n) + (1 - self.alpha_f) * self.F(t_n + dt)

        # solve for u_next
        m1 = (1 - self.alpha_m) / (self.beta * dt**2)
        m2 = (1 - self.alpha_m) / (self.beta * dt)
        m3 = (1 - self.alpha_m - 2 * self.beta) / (2 * self.beta)
        k1 = 1 - self.alpha_f
        rhs = force - self.alpha_f * np.dot(self.K, u_n) + np.dot(self.M, m1 * u_n + m2 * v_n + m3 * a_n)
        system_matrix = k1 * self.K + m1 * self.M
        u_next = np.linalg.solve(system_matrix, rhs)

        # compute a_next
        a_next = np.dot(self.A, u_next)

        # compute v_next
        v_next = v_n + dt * ((1 - self.gamma) * a_n + self.gamma * a_next)

        return np.concatenate([u_next, v_next, a_next])

class NewmarkBeta(TimesteppingMethod):
    def __init__(self, A: np.ndarray, beta: float, gamma: float):
        self.A = A
        self.beta = beta
        self.gamma = gamma
    
    def compute_timestep(self, dt: float, t_n: float, last_values: np.ndarray):
        del t_n
        u_n = last_values[[0, 1]]
        v_n = last_values[[2, 3]]
        a_n = last_values[[4, 5]]

        # solve for u_next
        M = np.eye(2, 2) - dt**2 * self.beta * self.A
        rhs = u_n + dt * v_n + dt**2 * (0.5 - self.beta) * a_n
        u_next = np.linalg.solve(M, rhs)

        # compute a_next
        a_next = np.dot(self.A, u_next)

        # compute v_next
        v_next = v_n + dt * ((1 - self.gamma) * a_n + self.gamma * a_next)

        return np.concatenate([u_next, v_next, a_next])


class ERK(TimesteppingMethod):
    def __init__(self, first_order_matrix: np.ndarray, order: int = 1):
        # for the formulation: y' = Ay:
        self.first_order_matrix = first_order_matrix
        if order == 1:
            self.erk_step = self._erk1
        elif order == 2:
            self.erk_step = self._erk2
        elif order == 4:
            self.erk_step = self._erk4
        else:
            raise NotImplementedError(f"Currently only supports order 1,2,4, not {order}!")
        
    
    def compute_timestep(self, dt: float, t_n: float, last_values: np.ndarray):
        next_values = self.erk_step(last_values, dt, t_n)
        return next_values

    def du_dt(self, last_values: np.ndarray, t_n: float) -> np.ndarray:
        del t_n
        return np.dot(self.first_order_matrix, last_values)

    def _erk1(self, u_i: np.ndarray, dt: float, t_i = 0) -> np.ndarray:
        """Explicit Euler method"""
        A = np.array([0])
        b = np.array([0])
        c = np.array([1])
        u_next = self._erk_gen(A, b, c, u_i, dt, t_i)
        return u_next

    def _erk2(self, u_i: np.ndarray, dt: float, t_i = None) -> np.ndarray:
        """Method of Heun (2nd order accurate)"""
        A = np.array([[0,0],[1,0]])
        b = np.array([0,1])
        c = np.array([0.5, 0.5])
        u_next = self._erk_gen(A, b, c, u_i, dt, t_i)
        return u_next

    def _erk4(self, u_i: np.ndarray, dt: float, t_i = None) -> np.ndarray:
        """RK4 scheme"""
        A = np.zeros((4,4), dtype=float)
        A[1,0] = 0.5
        A[2,1] = 0.5
        A[3,2] = 1
        b = np.array([0,.5,.5,1])
        c = np.array([1/6, 1/3, 1/3, 1/6])
        u_next = self._erk_gen(A, b, c, u_i, dt, t_i)
        return u_next
    
    def _erk_gen(self, A: np.ndarray, b, c, u_i: np.ndarray, dt: float, t_i: float) -> np.ndarray:
        k = np.zeros(b.shape + u_i.shape, dtype=u_i.dtype)
        k[0] = self.du_dt(u_i, t_i)
        if len(b) > 1:
            for i in range(1, len(b)):
                u_eval = u_i + dt * np.tensordot(A[i], k, axes=1)
                t_eval = t_i + b[i] * dt
                try:
                    k[i] = self.du_dt(u_eval, t_eval)
                except ValueError:
                    k[i] = self.du_dt(u_eval, t_eval)[i]
        u_next = u_i + dt * np.tensordot(c, k, axes=1)
        return u_next