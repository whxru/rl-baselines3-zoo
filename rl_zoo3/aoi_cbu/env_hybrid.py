import gym
import numpy as np
import pandas as pd
import os
import torch
import torch.cuda
from gym.spaces import Box, Discrete, Dict


class HybridCentralizedAoICbuEnv(gym.Env):
    def __init__(
            self,
            target='gowalla',
            training_set_split_ratio=0.7,
            manual_set_instance_idx=None,
            seed=1997,
    ):

        self.load_config(target)
        self.env_name = target

        self._is_training = True
        self._evaluation_inst_idx = 0

        self.K = self.beta.shape[0]
        self.N = self.beta.shape[1]
        self.device = torch.device('cpu')
        self.manual_set_instance_idx = manual_set_instance_idx

        self.max_train_inst_idx = int(self.N_instance * training_set_split_ratio) - 1

        self.action_space = Discrete(self.K)
        self.observation_space = Dict({
            "poi_active": Box(low=0, high=1, shape=(self.N,)),
            "weight": Box(0, 1, (self.N, )),
            "AoI": Box(0, self.T, (self.N, )),
            "t": Box(0, self.T, (1, ))
        })

        self.AoI_record = []
        self.reset_stat_vars()

        np.random.seed(seed)

    def load_config(self, target):

        def real_path(file_name):
            return os.path.join(os.path.dirname(__file__), 'config', target, file_name)

        res_config = []
        NK_suffix = 'N=80_K=26' if target == 'gowalla' else ''
        for config_name in ['p', 'w', 'o', 'beta', 'active']:
            res_config.append(np.load(real_path(f'{config_name}_{NK_suffix}.npy')))
        self.p, self.w, self.o, self.beta, self.poi_active = res_config
        self.o = self.o * self.beta
        self.N_instance = self.poi_active.shape[0]
        self.T = self.poi_active[0].shape[1]

    @property
    def current_instance_idx(self):
        return self.last_selected_inst_idx if self._is_training else self._evaluation_inst_idx

    @property
    def active_poi_indices(self):
        return np.argwhere(self.poi_active[self.current_instance_idx, :, self.t] == 1).ravel()

    @property
    def current_w(self):
        res = self.w[self.current_instance_idx, :, self.t]
        res[res == 0] = 1
        return res

    def enable_evaluation(self, enable=True, inst_idx=0):
        self._is_training = not enable
        self._evaluation_inst_idx = inst_idx

    def _compute_obs(self, change_selected_inst_idx=False):
        if self._is_training and change_selected_inst_idx:
            if self.manual_set_instance_idx is not None:
                self.last_selected_inst_idx = np.random.choice(self.manual_set_instance_idx)
            else:
                self.last_selected_inst_idx = np.random.randint(low=0, high=self.max_train_inst_idx)

        poi_active = np.zeros(self.N)
        poi_active[self.active_poi_indices] = 1

        res_obs = {
            "poi_active": poi_active,
            "weight": self.current_w,
            "AoI": self.target_AoIs,
            "t": [self.t]
        }
        return res_obs

    def _compute_reward(self):
        w = self.current_w[self.active_poi_indices]
        AoI = self.target_AoIs[self.active_poi_indices]
        return - np.dot(w, AoI)

    def reset_stat_vars(self):
        self.t = 0
        self.last_selected_inst_idx = 0
        self.target_AoIs = np.zeros(self.N)
        self.sum_AoI = 0

        self.AoI_distrib = np.zeros(self.T)
        self.weight_distrib = np.zeros(1000)

    def step(self, action):
        cur_active_target_AoIs = self.target_AoIs[self.active_poi_indices].copy()
        self.target_AoIs = np.zeros(self.N)
        # print(f'# of new PoIs: {np.count_nonzero(cur_active_target_AoIs == 0)}')
        cur_active_target_AoIs += 1
        self.target_AoIs[self.active_poi_indices] = cur_active_target_AoIs

        selected_source = int(action)

        w = self.current_w[self.active_poi_indices]
        w = w
        old_aoi_sum = np.dot(w, self.target_AoIs[self.active_poi_indices])

        if np.random.random() <= self.p[selected_source]:
            # successfully_update = np.random.random(self.N) < self.o[selected_source]
            # successfully_update_indices = np.argwhere(successfully_update).ravel()
            self.target_AoIs[np.random.random(self.N) < self.o[selected_source]] = 1

        if len(self.active_poi_indices) > 0:
            aoi = self.target_AoIs[self.active_poi_indices]
            self.AoI_distrib[aoi.astype(int)] += 1
            self.weight_distrib[w.astype(int)] += 1
            new_aoi_sum = np.dot(w, aoi)
            # print(f'Weighted AoI reduced by: {old_aoi_sum - new_aoi_sum}, to: {new_aoi_sum}')
            self.sum_AoI += new_aoi_sum

        reward = self._compute_reward()
        obs = self._compute_obs()
        done = self.t >= self.T - 1
        info = {}

        self.t += 1

        if done:
            self.AoI_record.append(self.sum_AoI / self.T)
            # print(self.sum_AoI / self.T)

        return obs, reward, done, info

    def render(self, mode="human"):
        pass

    def reset(self):
        self.reset_stat_vars()
        return self._compute_obs(change_selected_inst_idx=True)


if __name__ == '__main__':
    env = HybridCentralizedAoICbuEnv()

    print(np.sum(env.beta, axis=1))
    print(np.sum(env.beta, axis=0))

    obs = env.reset()
    done = False

    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
    # print(env.AoI_record)
    AoI_distrib = env.AoI_distrib
    weight_distrib = env.weight_distrib

    import matplotlib.pyplot as plt
    for distrib in [AoI_distrib]:
        distrib = np.array(distrib[0:50])
        x = 0.5 + np.arange(len(distrib))
        y = distrib / np.sum(distrib)
        plt.bar(x, y, width=0.5)
        plt.xticks(0.5 + np.arange(len(distrib)), np.arange(len(x)))
        for _x, _y in zip(x, y):
            plt.text(_x, _y, np.round(_y * 100, 1))
        plt.show()
