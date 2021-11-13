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
                A: np.ndarray, M: np.ndarray, K: np.ndarray, 
                beta: float, gamma: float,
                alpha_f: float, alpha_m: float,
                F: callable):
        self.A = A
        self.M = M
        self.K = -K
        self.beta = beta
        self.gamma = gamma
        self.alpha_f = alpha_f
        self.alpha_m = alpha_m
        self.F = F

    def compute_timestep(self, dt: float, t_n: float, last_values: np.ndarray):
        u_n = last_values[[0, 1]]
        v_n = last_values[[2, 3]]
        a_n = last_values[[4, 5]]
        force = self.F(t_n + (1 - self.alpha_f) * dt)

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
