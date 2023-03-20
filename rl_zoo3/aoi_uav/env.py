import gym
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time


class AoIUavTrajectoryPlanningEnv(gym.Env):

    def __init__(self, observation_size=50, world_size=2e4, K=20, T=2000, seed=0, pos_seed=None):
        self.observation_size = observation_size
        self.world_size = world_size
        self.K = K
        self.T = T
        if pos_seed is None:
            pos_seed = seed + 1
        self.config_rs = np.random.RandomState(seed=pos_seed)
        self.qk = self.config_rs.random((self.K, 3)) * world_size
        self.qk[:, -1] = 0
        self.w = self.config_rs.random(self.K)

        self.env_name = 'AoI-UAV'
        self.rs = np.random.RandomState(seed=seed) if seed > 0 else np.random
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=0, high=self.T * self.K, shape=(self.observation_size, self.observation_size))

        self.beta = np.deg2rad(70)
        self.eta = 4 / 3
        self.z_min = 3
        self.z_max = 5e2
        self.lamb_xoy = 9  # meters (per second)
        self.lamb_z = 4  # meters (per second)
        self.AoI_record = []

        self.tan_half_beta = np.tan(self.beta / 2)
        self.arccot_eta = np.arctan(1 / self.eta)
        self.arctan_eta = np.arctan(self.eta)

        self._prepare_p()
        self._reset_stat_vars()

    def _reset_stat_vars(self):
        self.q_u = np.array([0, 0, self.z_min])
        self.yaw = 0  #
        self.t = 0
        self.expect_h = np.zeros(self.K)
        self.expect_sum_h = np.zeros(self.K)
        self.actual_h = np.zeros(self.K)
        self.actual_sum_h = np.zeros(self.K)
        self.actual_peak_h = [[] for _ in range(self.K)]
        self.z_history = []

    @property
    def gamma(self):
        return self.yaw - self.arccot_eta

    def r_u(self, z=None):
        if z is None:
            z = self.q_u[2]
        return self.q_u[2] / self.tan_half_beta

    def l_x(self, z=None):
        if z is None:
            z = self.q_u[2]
        r = self.r_u(z)
        return r * np.sin(self.arctan_eta)

    def l_y(self, z=None):
        if z is None:
            z = self.q_u[2]
        r = self.r_u(z)
        return r * np.cos(self.arctan_eta)

    def _prepare_p(self):
        dataset = np.array([
            # [self.z_min, 1],
            [10, 0.91],
            [20, 0.83],
            [30, 0.63]
        ])
        self.p_func_params, _ = curve_fit(self._p_func, dataset[:, 0].flatten(), dataset[:, 1].flatten())

    def _p_func(self, x, a, b):
        return 1 / (1 + np.exp((x - a) / b))

    def p(self, z=None):
        if z is None:
            z = self.q_u[2]
        if self.z_min <= z <= self.z_max:
            return self._p_func(z, *self.p_func_params)
        return 0

    def covered(self, q=None):
        if q is None:
            q = self.q_u.copy()
        q_hat_u = np.array([q[0], q[1], 0])
        b_x = np.array([np.cos(self.yaw), np.sin(self.yaw), 0])
        b_y = np.array([np.sin(self.yaw), np.cos(self.yaw), 0])
        x_covered = np.abs(np.dot(self.qk - q_hat_u, b_x)) <= self.l_x(q[2]) / 2
        y_covered = np.abs(np.dot(self.qk - q_hat_u, b_y)) <= self.l_y(q[2]) / 2
        return np.logical_and(x_covered, y_covered).astype(int)

    def display_p_func(self):
        z = np.arange(self.z_min, self.z_max, 1)
        y = self._p_func(z, *self.p_func_params)
        plt.plot(z, y)
        plt.show()

    def step(self, action):
        delta_z = action * self.lamb_z
        self.q_u[2] += delta_z
        self.z_history.append(self.q_u[2])

        current_p = self.p()
        current_poi_covered = self.covered()

        # todo: Control the XOY movement for the UAV
        # todo: Control the yaw of the UAV

        update_prob = current_p * current_poi_covered
        self.expect_sum_h += self.expect_h
        self.expect_h = update_prob * np.ones(self.K) + (1 - update_prob) * (self.expect_h + 1)

        self.actual_sum_h += self.actual_h
        self.actual_update_result = (np.random.random(self.K) <= update_prob).astype(int)
        for k in range(self.K):
            if self.actual_update_result[k] == 1:
                self.actual_peak_h[k].append(self.actual_h[k])
                self.actual_h[k] = 1
            else:
                self.actual_h[k] += 1

        obs = self._compute_obs()
        reward = np.dot(self.w, self.expect_h)
        done = self.t >= self.T
        info = None

        return obs, reward, done, info

    def _compute_obs(self):
        # todo: compute observation
        return None

    def reset(self):
        self._reset_stat_vars()
        return self._compute_obs()


if __name__ == '__main__':
    env = AoIUavTrajectoryPlanningEnv()
    env.display_p_func()

