import numpy as np

from timestepping import TimesteppingMethod

class MonolithicSystem:
    def __init__(self):
        # for the formulation: u'' = A_second_order * u:
        self.A_second_order = np.array([[-2., 1.],[1., -2.]])
        # for the formulation: Mu'' + Ku = 0:
        self.M = np.eye(2, 2, dtype=float)
        self.K = - self.A_second_order.copy()
        # for the formulation: y' = A_first_order * y:
        self.A_first_order = np.array(
            [[0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [-2., 1., 0., 0.],
            [1., -2., 0., 0.]]
        )
    
    def second_order_force(self, t):
        del t # no time-dependent force for this system
        return np.array([0., 0.])

    def first_order_force(self, t):
        del t # no time-dependent force for this system
        return np.array([0., 0., 0., 0.])


    def _initial_conditions(self):
        u0 = np.array([1., 0.])
        v0 = np.array([0., 0.])
        a0 = np.dot(self.A_second_order, u0)
        return np.concatenate([u0, v0, a0])

    def analytical_solution(self, t_end: float, N: int):
        t = np.linspace(0, t_end, N+1)
        result = np.array([
            0.5 * (np.cos(t) + np.cos(np.sqrt(3)*t)),
            0.5 * (np.cos(t) - np.cos(np.sqrt(3)*t)),
            0.5 * (- np.sin(t) - np.sqrt(3) * np.sin(np.sqrt(3)*t)),
            0.5 * (- np.sin(t) + np.sqrt(3) * np.sin(np.sqrt(3)*t)),
        ])
        return result

    def numerical_solution(self, t_end: float, N: int, solver: TimesteppingMethod):
        dt = t_end/N
        try:
            result = np.zeros((6, N+1))
            result[:,0] = self._initial_conditions()
            solver.integrate(dt, N, result)
        except ValueError:
            result = np.zeros((4, N+1))
            result[:,0] = self._initial_conditions()[:4]
            solver.integrate(dt, N, result)
        return result
