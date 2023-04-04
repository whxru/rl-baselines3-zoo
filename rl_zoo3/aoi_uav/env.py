import gym
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time
import seaborn as sns
from rl_zoo3.aoi_uav.epsilon_rt import EpsilonRT


class AoIUavTrajectoryPlanningEnv(gym.Env):

    def __init__(self, observation_size=50, world_size=1000, K=50, T=2000, seed=0, pos_seed=2096, x0=0):
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
        self.x0 = x0

        self.env_name = 'AoI-UAV'
        self.rs = np.random.RandomState(seed=seed) if seed > 0 else np.random
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=0, high=self.K, shape=(2, self.observation_size, self.observation_size))

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

        self.grid_indices = np.array(np.meshgrid(range(self.observation_size), range(self.observation_size), [0])).T.reshape(self.observation_size**2, 3)
        self.grid_size = self.world_size / self.observation_size
        self.grid_q = (self.grid_indices + .5) * self.grid_size
        self.grid_q[:, 2] = 0
        self.poi_in_grid = np.zeros((self.observation_size**2, len(self.qk)))
        for k, q in enumerate(self.qk):
            self.poi_in_grid[int((q[0] // self.grid_size) * self.observation_size + q[1] // self.grid_size), k] = 1

        self._xoy_cmds = []
        self._current_delta_qu = np.zeros(3)

        V, E = EpsilonRT.make_graph(self.w, self.qk, EpsilonRT.build_edges(self.w, self.qk, self.x0))
        xoy_agent = EpsilonRT(V, E, epsilon=0.002895393667369076)
        t0 = time.time()
        self.set_agent(xoy_agent, 'xoy')

    def set_agent(self, agent, agent_type='xoy'):
        if agent_type == 'xoy':
            self._xoy_agent = agent
        elif agent_type == 'yaw':
            self._yaw_agent = agent

    def _reset_stat_vars(self):
        self.qu = np.array([self.qk[self.x0][0], self.qk[self.x0][1], self.z_min])
        self.yaw = 0  #
        self.t = 0
        self.visited_times = np.zeros(self.K)
        self.expect_h = np.zeros(self.K)
        self.expect_sum_h = np.zeros(self.K)
        self.actual_h = np.zeros(self.K)
        self.actual_sum_h = np.zeros(self.K)
        self.actual_peak_h = [[] for _ in range(self.K)]
        self.z_history = [self.z_min]

    @property
    def q_u_ground(self):
        return np.array([self.qu[0], self.qu[1], 0])

    @property
    def gamma(self):
        return self.yaw - self.arccot_eta

    @property
    def estimated_r(self):
        return self.z_history[-1] * np.tan(self.beta / 2)

    def r_u(self, z=None):
        if z is None:
            z = self.qu[2]
        return self.qu[2] / self.tan_half_beta

    def l_x(self, z=None):
        if z is None:
            z = self.qu[2]
        r = self.r_u(z)
        return r * np.sin(self.arctan_eta)

    def l_y(self, z=None):
        if z is None:
            z = self.qu[2]
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
            z = self.qu[2]
        if self.z_min <= z <= self.z_max:
            return self._p_func(z, *self.p_func_params)
        return 0

    def covered(self, q=None, qk=None):
        if q is None:
            q = self.qu.copy()
        if qk is None:
            qk = self.qk.copy()
        q_hat_u = np.array([q[0], q[1], 0])
        b_x = np.array([np.cos(self.yaw), np.sin(self.yaw), 0])
        b_y = np.array([np.sin(self.yaw), -np.cos(self.yaw), 0])
        lx_half = self.l_x(q[2]) / 2
        ly_half = self.l_y(q[2]) / 2
        x_covered = np.abs(np.dot(qk - q_hat_u, b_x)) <= lx_half
        y_covered = np.abs(np.dot(qk - q_hat_u, b_y)) <= ly_half
        return np.logical_and(x_covered, y_covered).astype(int)

    def display_p_func(self):
        z = np.arange(self.z_min, self.z_max, 1)
        y = self._p_func(z, *self.p_func_params)
        plt.plot(z, y)
        plt.show()

    def step(self, action):
        delta_z = action * self.lamb_z
        self.qu[2] += delta_z
        self.qu[2] = max(self.qu[2], self.z_min)
        self.z_history.append(self.qu[2])

        current_p = self.p()

        self.qu += self._current_delta_qu

        self.yaw = np.random.random() * np.pi * 2

        current_poi_covered = self.covered()
        update_prob = current_p * current_poi_covered
        self.visited_times += update_prob
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
                if self.t == self.T - 1:
                    self.actual_peak_h[k].append(self.actual_h[k])

        if len(self._xoy_cmds) == 0 or np.linalg.norm(self.q_u_ground - self._xoy_cmds[-1]) <= 1e-2:
            if len(self._xoy_cmds) > 0:  # to avoid cumulative bias
                self.qu[0], self.qu[1], _ = self._xoy_cmds[-1]
            q_current = self.q_u_ground
            q_next, delta_t = self._xoy_agent.step(q_current, info={
                'covered': current_poi_covered,
                'estimated_r': self.estimated_r,
                'visited_times': self.visited_times
            })
            self._current_delta_qu = (q_next - q_current) / delta_t
            self._xoy_cmds.append(q_next)

        self.t += 1
        obs = self._compute_obs()
        reward = np.dot(self.w, self.expect_h)
        done = self.t >= self.T
        info = {}

        if done:
            av_pAoI = np.dot(self.w, np.array([np.mean(pAoI) for pAoI in self.actual_peak_h])) / self.K
            av_AoI = np.dot(self.w, self.expect_sum_h) / self.T / self.K
            print(f'Average peak AoI: {av_pAoI}, average AoI: {av_AoI}')
            self.AoI_record.append(av_AoI)

        return obs, reward, done, info

    def _compute_obs(self, q_u=None, h=None):
        if q_u is None:
            q_u = self.qu.copy()
        if h is None:
            h = self.expect_h.copy()
        qk = self.qk.copy()
        assert len(h) == len(qk)
        h /= self.T
        K = len(h)

        grid_indices = np.array(np.meshgrid(range(self.observation_size), range(self.observation_size), [0])).T.reshape(self.observation_size**2, 3)
        grid_size = self.world_size / self.observation_size
        grid_q = (grid_indices + .5) * grid_size
        # estimated_r = min(self.l_x(self.z_history[-1]), self.l_y(self.z_history[-1])) / 2
        grid_covered_poi = cdist(grid_q, qk) <= self.estimated_r + 1e-5
        grid_covered_poi_h = np.dot(np.logical_or(grid_covered_poi, self.poi_in_grid), h.reshape((K, 1))).reshape(self.observation_size, self.observation_size)
        grid_covered_poi_h /= self.T

        p = self.p()
        grid_covered_by_uav = self.covered(q_u, self.grid_q).reshape((self.observation_size, self.observation_size)) * p
        uav_grid_x, uav_grid_y = int(q_u[0] // self.grid_size), int(q_u[1] // self.grid_size)
        try:
            grid_covered_by_uav[uav_grid_x, uav_grid_y] = min(grid_covered_by_uav[uav_grid_x, uav_grid_y] + p, p)
        except IndexError:
            pass
        return np.stack((grid_covered_poi_h, grid_covered_by_uav)).astype(np.float32)

    def plot_obs(self, obs):
        # print(f'Current height: {self.qu[2]} meter')
        sns.heatmap(obs[0])
        plt.show()
        sns.heatmap(obs[1])
        plt.show()

    def render(self, mode="human"):
        self.plot_obs(self._compute_obs())

    def reset(self):
        self._reset_stat_vars()
        return self._compute_obs()


if __name__ == '__main__':
    np.random.seed(2096)
    from rl_zoo3.aoi_uav.epsilon_rt import EpsilonRT
    from stable_baselines3.common.env_checker import check_env

    env = AoIUavTrajectoryPlanningEnv(world_size=1000, observation_size=50, T=1800, K=100)
    check_env(env)
    t0 = time.time()
    obs = env.reset()
    done = False
    while not done:
        # action = env.action_space.sample()
        action = 1 if env.qu[2] <= 40 else 0
        obs, reward, done, info = env.step(action)
        if env.t % 200 == 0:
            env.render()
    print(f'One episode consumes {time.time() - t0} seconds')
    # env.display_p_func()

