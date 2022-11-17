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
            env_name='',
            manual_set_instance_idx=None,
            seed=1997,
    ):

        self.load_config(target)
        self.T = None
        self.env_name = env_name

        self._is_training = True
        self._evaluation_inst_idx = 0

        self.K = self.beta.shape[0]
        self.N = self.beta.shape[1]
        self.device = torch.device('cpu')
        self.manual_set_instance_idx = manual_set_instance_idx

        self.T = self.poi_active[0].shape[1]
        self.N_instance = self.poi_active.shape[0]
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
        computed_config_path = real_path(f'computed_config_{NK_suffix}.npy')
        if not os.path.isfile(computed_config_path):
            self._compute_config(save_to=computed_config_path)
        self._edges, self._vtx_props, self._poi_indices, self._source_indices = np.load(computed_config_path, allow_pickle=True).tolist()

    @property
    def current_instance_idx(self):
        return self.last_selected_inst_idx if self._is_training else self._evaluation_inst_idx

    @property
    def active_poi_indices(self):
        return self._poi_indices[self.current_instance_idx][self.t]

    @property
    def active_source_indices(self):
        return self._source_indices[self.current_instance_idx][self.t]

    @property
    def current_w(self):
        res = self.w[self.current_instance_idx, :, self.t]
        res[res == 0] = 1
        return res

    @property
    def current_vtx_prop(self):
        return self._vtx_props[self.current_instance_idx][self.t]

    @staticmethod
    def norm_graph_index(edges):
        uniq_source_indices = np.unique(edges[:, 0])
        uniq_poi_indices = np.unique(edges[:, 1])
        source_map = dict(zip(uniq_source_indices, np.arange(len(uniq_source_indices))))
        poi_map = dict(zip(uniq_poi_indices, np.arange(len(uniq_source_indices), len(uniq_source_indices) + len(uniq_poi_indices) + 1)))
        res_props = [[1, 0]] * len(uniq_source_indices) + [[0, 1]] * len(uniq_poi_indices)
        res_edges = []
        for source, poi in edges:
            res_edges.append([source_map[source], poi_map[poi]])
        return res_edges, res_props, uniq_poi_indices, uniq_source_indices

    def enable_evaluation(self, enable=True, inst_idx=0):
        self._is_training = not enable
        self._evaluation_inst_idx = inst_idx

    def _compute_config(self, save_to):
        df_coverage = pd.DataFrame(self.beta).stack().reset_index().rename(columns={'level_0': 'source', 'level_1': 'PoI', 0: 'covered'})
        res_edges = [[] for _ in range(self.N_instance)]
        res_props = [[] for _ in range(self.N_instance)]
        res_poi_indices = [[] for _ in range(self.N_instance)]
        res_source_indices = [[] for _ in range(self.N_instance)]
        for inst_idx, poi_ac in enumerate(self.poi_active):
            df_ac = pd.DataFrame(poi_ac).stack().reset_index().rename(columns={'level_0': 'PoI', 'level_1': 't', 0: 'active'})
            df_merge = df_ac.merge(df_coverage, on='PoI', how='outer', suffixes=('', ''))
            df_merge.drop(df_merge[df_merge['active'] + df_merge['covered'] < 2].index, inplace=True)
            for t in range(self.T):
                print(f'Day {inst_idx}, Minute {t}')
                original_record = df_merge[df_merge['t'] == t][['source', 'PoI']].to_numpy()
                r_edges, r_props, r_poi_indices, r_source_indices = self.norm_graph_index(original_record)
                res_edges[inst_idx].append(r_edges)
                res_props[inst_idx].append(r_props)
                res_poi_indices[inst_idx].append(r_poi_indices)
                res_source_indices[inst_idx].append(r_source_indices)
        np.save(save_to, [res_edges, res_props, res_poi_indices, res_source_indices])

    def _compute_obs(self, change_selected_inst_idx=False):
        if self._is_training and change_selected_inst_idx:
            if self.manual_set_instance_idx is not None:
                self.last_selected_inst_idx = np.random.choice(self.manual_set_instance_idx)
            else:
                self.last_selected_inst_idx = np.random.randint(low=0, high=self.max_train_inst_idx)

        poi_active = np.zeros(self.N)
        poi_active[self.active_poi_indices] = 1
        # res_obs = {
        #     "poi_active": torch.tensor(poi_active, dtype=torch.int),
        #     "weight": torch.tensor(self.current_w, dtype=torch.float),
        #     "AoI": torch.tensor(self.target_AoIs, dtype=torch.float),
        #     "t": torch.tensor([self.t])
        # }
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