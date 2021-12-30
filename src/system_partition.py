import numpy as np

class SystemPartition:
    def __init__(self, left_system: bool, result_shape: tuple = (3,1)):
        self.left_system_bool = left_system
        # for the formulation: u'' = A_second_order * u:
        self.A_second_order = (-2) * np.eye(1, 1, dtype=float)
        # for the formulation: Mu'' + Ku = 0:
        self.M = np.eye(1, 1, dtype=float)
        self.K = 2 * np.eye(1, 1, dtype=float)
        # for the formulation: y' = A_first_order * y:
        self.A_first_order = np.array(
            [[0., 1.],
            [-2., 0.]]
        )
        self.result = np.zeros(result_shape)
        self.other_u = self._initial_other_u()
        try:
            self.result[:, 0] = self._initial_conditions()
        except ValueError:
            self.result[:, 0] = self._initial_conditions()[:2]
        # print("--- System Setup ---")
        # print(f"A_first_order: {self.A_first_order}")
        # print(f"A_second_order: {self.A_second_order}")
        # print(f"K: {self.K}")
        # print(f"M: {self.M}")
        # print(f"Initial Conditions: {self.result[:, 0]}")
        # print(f"Other u: {self.other_u}")

    def second_order_force(self, t):
        del t # no time-dependent force so far
        # k12 * other_u
        # print(f"second order, t = {t}, left system: {self.left_system_bool}: {self.other_u}")
        return self.other_u

    def first_order_force(self, t):
        del t # no time-dependent force so far
        # [0, k12 * other_u]
        # print(f"first order, t = {t}, left system: {self.left_system_bool}: {self.other_u}")
        return np.array([0., self.other_u], dtype=object)

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