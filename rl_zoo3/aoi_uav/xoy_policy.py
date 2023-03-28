import numpy as np
import matplotlib.pyplot as plt
from rl_zoo3.aoi_uav.env import AoIUavTrajectoryPlanningEnv


class XoYPolicy:

    def __init__(self, env:AoIUavTrajectoryPlanningEnv):
        self.K = env.K
        self.qk = env.qk

    def epsilon(self):
        return 1e-2


    def propose_movement(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError


class DecisiveRandomizedTraveling(XoYPolicy):

    def __init__(self, env: AoIUavTrajectoryPlanningEnv):
        super(DecisiveRandomizedTraveling, self).__init__(env)
        self._calculate_transition_matrix()

    @property
    def name(self):
        return 'Decisive RT'

    def _calculate_transition_matrix(self):
        pass

    def propose_movement(self):
        return None


if __name__ == '__main__':
    pass