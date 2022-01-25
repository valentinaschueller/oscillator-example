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
        if "sc" in kwargs:
            self.sc = kwargs["sc"]
        else:
            print("Default: No subcycling")
            self.sc = 1
        
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

        if self.left_system_bool:
            self.result = np.full((result_values, self.sc*N+1), np.inf)
            self.t_values = np.linspace(0, t_end, self.sc*N+1)
        else:
            self.result = np.full((result_values, N+1), np.inf)
            self.t_values = np.linspace(0, t_end, N+1)
        if self.left_system_bool:
            self.other_u = np.full(N + 1, np.inf)
            self.other_t_values = np.linspace(0, t_end, N+1)
            self.other_dt = t_end/N
        else:
            self.other_u = np.full(self.sc*N + 1, np.inf)
            self.other_t_values = np.linspace(0, t_end, self.sc*N+1)
            self.other_dt = t_end/(self.sc*N)
        self.other_u[0] = self._initial_other_u()
        try:
            self.result[:, 0] = self._initial_conditions()
        except ValueError:
            self.result[:, 0] = self._initial_conditions()[:2]

    def other_u_at(self, t, t_lower):
        # get id of "left" u-value in current interval
        idx = np.where(self.other_t_values == self.other_t_values[self.other_t_values <= (t_lower+1e-10)].max())[0][0]
        if self.interpolation_order == 0:
            if self.other_u[idx + 1] == np.inf:
                self.other_u[idx + 1] = self.other_u[idx]
            del t
            return self.other_u[idx + 1]
        elif self.interpolation_order == 1:
            previous_other_u = self.other_u[idx]
            if self.other_u[idx + 1] == np.inf:
                self.other_u[idx + 1] = previous_other_u # constant extrapolation in first iteration
            next_other_u = self.other_u[idx + 1]
            percentage = (t - t_lower) / self.other_dt
            interpolated_u = interpolate_linear(previous_other_u, next_other_u, percentage)
            # print(f"Using interpolated {interpolated_u} instead of {previous_other_u}, {next_other_u}")
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
        u0 = np.array([0.5])
        v0 = np.array([0.])
        a0 = np.dot(self.A_second_order, u0) + self.k12 * self._initial_other_u()
        return np.concatenate([u0, v0, a0])
    
    def _initial_other_u(self):
        return 0.5

class DiffTimeScales(SystemPartition):
    def __init__(self,
                left_system: bool,
                t_end: float = 0.,
                N: int = 0,
                result_values: int = 0,
                **kwargs
                ):
        super().__init__(left_system, t_end, N, result_values, 20, 0.1, 1, 1, 1, **kwargs)