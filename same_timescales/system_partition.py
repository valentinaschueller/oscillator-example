import numpy as np
from utility import interpolate_linear

class SystemPartition:
    def __init__(self,
                left_system: bool,
                t_end: float = 0.,
                N: int = 0,
                result_values: int = 0,
                k1: int = 1,
                k2: int = 1,
                k12: int = 1,
                m1: int = 1,
                m2: int = 1,
                **kwargs):
        self.left_system_bool = left_system
        self.k12 = k12
        self.interpolation_order = kwargs.get("interpolation_order", 0)
        
        if left_system:
            # for the formulation: Mu'' + Ku = 0:
            self.M = m1 * np.eye(1, 1, dtype=float)
            self.K = (k1 + k12) * np.eye(1, 1, dtype=float)
            # for the formulation: y' = A_first_order * y:
            self.A_first_order = np.array(
                [[0., 1.],
                [-(k1 + k12)/m1, 0.]]
            )
            # for the formulation: u'' = A_second_order * u:
            self.A_second_order = -(k1 + k12) * np.eye(1, 1, dtype=float)
        else:
            # for the formulation: Mu'' + Ku = 0:
            self.M = m2 * np.eye(1, 1, dtype=float)
            self.K = (k2 + k12) * np.eye(1, 1, dtype=float)
            # for the formulation: y' = A_first_order * y:
            self.A_first_order = np.array(
                [[0., 1.],
                [-(k2 + k12)/m2, 0.]]
            )
            # for the formulation: u'' = A_second_order * u:
            self.A_second_order = -(k2 + k12) * np.eye(1, 1, dtype=float)

        self.result = np.zeros((result_values, N+1))
        self.t_values = np.linspace(0, t_end, N+1)
        self.dt = t_end/N
        self.other_u = np.full(N+1, np.inf)
        self.other_u[0] = self._initial_other_u()
        try:
            self.result[:, 0] = self._initial_conditions()
        except ValueError:
            self.result[:, 0] = self._initial_conditions()[:2]

    def other_u_at(self, t, t_lower):
        idx = np.where(np.abs(self.t_values - t_lower) < 1e-6)[0][0]
        # print(f"t_lower: {t_lower}, idx: {idx}")
        if self.interpolation_order == 0:
            if self.other_u[idx + 1] == np.inf:
                self.other_u[idx + 1] = self.other_u[idx]
            del t
            return self.other_u[idx + 1]
        elif self.interpolation_order == 1:
            previous_other_u = self.other_u[idx]
            if self.other_u[idx + 1] == np.inf:
                self.other_u[idx + 1] = previous_other_u
            next_other_u = self.other_u[idx + 1]
            # print(f"Interpolating between {previous_other_u}, {next_other_u} at t = {t}")
            percentage = (t - t_lower) / self.dt
            # print(f"Percentage: {percentage}")
            interpolated_u = interpolate_linear(previous_other_u, next_other_u, percentage)
            # print(f"interpolated value: {interpolated_u}")
            return interpolated_u
        else:
            raise NotImplementedError
    
    def second_order_force(self, t, t_lower):
        if t == 0:
            return self.k12 * self._initial_other_u()
        else:
            return self.k12 * self.other_u_at(t, t_lower)

    def first_order_force(self, t, t_lower):
        if t == 0:
            return np.array([0., self.k12 * self._initial_other_u()], dtype=object)
        else:
            return np.array([0., self.k12 * self.other_u_at(t, t_lower)], dtype=object)

    def _initial_conditions(self):
        if self.left_system_bool:
            u0 = np.array([1.])
        else:
            u0 = np.array([0.])
        v0 = np.array([0.])
        a0 = np.dot(self.A_second_order, u0) + self.k12 * self._initial_other_u()
        return np.concatenate([u0, v0, a0])
    
    def _initial_other_u(self):
        if not self.left_system_bool:
            return 1.
        else:
            return 0.


class SameTimescales(SystemPartition):
    def __init__(self,
                left_system: bool,
                t_end: float = 0.,
                N: int = 0,
                result_values: int = 0,
                **kwargs):
        super().__init__(left_system, t_end, N, result_values, 1, 1, 1, 1, 1, **kwargs)
