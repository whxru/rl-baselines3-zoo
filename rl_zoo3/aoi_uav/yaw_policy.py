import numpy as np
import matplotlib as plt
from rl_zoo3.aoi_uav.env import AoIUavTrajectoryPlanningEnv


class YawPolicy:

    def __init__(self, env:AoIUavTrajectoryPlanningEnv):
        self.K = env.K

    def yaw(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError
