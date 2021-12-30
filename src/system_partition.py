import numpy as np

class SystemPartition:
    def __init__(self,
                left_system: bool,
                result_shape: tuple = (3,1),
                k1: int = 1,
                k2: int = 1,
                k12: int = 1,
                m1: int = 1,
                m2: int = 1):
        self.left_system_bool = left_system
        self.k12 = k12
        
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

        self.result = np.zeros(result_shape)
        self.other_u = self._initial_other_u()
        try:
            self.result[:, 0] = self._initial_conditions()
        except ValueError:
            self.result[:, 0] = self._initial_conditions()[:2]

    def second_order_force(self, t):
        del t # no time-dependent force so far
        return self.k12 * self.other_u

    def first_order_force(self, t):
        del t # no time-dependent force so far
        return np.array([0., self.k12 * self.other_u], dtype=object)

    def _initial_conditions(self):
        if self.left_system_bool:
            u0 = np.array([1.])
        else:
            u0 = np.array([0.])
        v0 = np.array([0.])
        a0 = np.dot(self.A_second_order, u0) + self.second_order_force(0)
        return np.concatenate([u0, v0, a0])
    
    def _initial_other_u(self):
        if not self.left_system_bool:
            return np.array([1.])
        else:
            return np.array([0.])


class SameTimescales(SystemPartition):
    def __init__(self,
                left_system: bool,
                result_shape: tuple = (3,1)):
        super().__init__(left_system, result_shape, 1, 1, 1, 1, 1)