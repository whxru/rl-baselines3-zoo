# Version 1: adapt KKT analysis of SRS policy

import gym
import numpy as np
import torch
import pandas as pd
from torch.nn.parameter import Parameter
from torch.nn import init, GRU, Sequential, Linear
# from torch.multiprocessing import Pool, set_start_method
import torch.nn.functional as F

import time

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env


try:
    from env.env_hybrid import HybridCentralizedAoICbuEnv
except ModuleNotFoundError:
    from rl_zoo3.aoi_cbu.env_hybrid import HybridCentralizedAoICbuEnv


class DynamicPoIFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            env_target='gowalla',
            num_out_channel_feat=4,
            num_out_channel_aoi=4,
            computation_config={},
            net_dim=[1024, 512],
    ):
        env = HybridCentralizedAoICbuEnv(target=env_target)
        self.beta = torch.tensor(env.beta, dtype=torch.int)
        self.n_parent = torch.sum(self.beta, dim=0)
        self.p = torch.tensor(env.p, dtype=torch.float32)
        self.o = torch.tensor(env.o, dtype=torch.float32)
        self.K = self.beta.shape[0]
        self.N = self.beta.shape[1]
        super(DynamicPoIFeatureExtractor, self).__init__(observation_space, features_dim=net_dim[-1])

        self.device = None
        self.computation_config = computation_config

        self.gru_w = GRU(self.K * self.K, self.K * self.K)
        self.grw_w_2 = GRU(self.K * self.K, self.K * self.K)

        activation_func = torch.nn.Sigmoid if 'activation_func' not in self.computation_config else {
            'relu': torch.nn.ReLU,
            'relu6': torch.nn.ReLU6,
            'sigmoid': torch.nn.Sigmoid,
            'leaky_relu': torch.nn.Sigmoid,
            'rrelu': torch.nn.RReLU,
            'tanh': torch.nn.Tanh
        }[self.computation_config['activation_func']]
        modules = []
        net_dim = [self.K * self.K * 4] + net_dim
        for i in range(1, len(net_dim)):
            modules.append(Linear(net_dim[i-1], net_dim[i]))
            modules.append(activation_func())
        self.seq = Sequential(*modules)

    def expand_to_out_channels(self, x, num=None):
        if num is None:
            num = self.num_out_channel_feat
        return torch.reshape(x, (len(x), 1)).expand((-1, num))

    def unbatch_forward(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation)
        if self.device is None:
            self.device = observation['AoI'].device
            self.p = self.p.to(self.device)
            self.o = self.o.to(self.device)
            self.beta = self.beta.to(self.device)
            self.n_parent = self.n_parent.to(self.device)

        indices = torch.argwhere(observation['poi_active'] == 1).squeeze_()
        if indices.ndim == 0:
            return self.seq(torch.zeros(self.K * self.K * 4).to(self.device))
        w = observation['weight'][indices]  # feature of PoIs
        AoI = observation['AoI'][indices]  # feature of PoIs
        beta = self.beta[:, indices].to(dtype=torch.float)
        beta_o = self.beta[:, indices] * self.o[:, indices]
        p = torch.squeeze(torch.mm(torch.t(beta_o), torch.reshape(self.p, (self.K, 1))))  # p for each PoI, (N(t), 1)
        n_parent = self.n_parent[indices]
        w_AoI_p = w * AoI / p

        overlapped_w = torch.mm(beta, torch.mm(self.expand_to_out_channels(w, len(indices)), torch.t(beta)))
        gru_out_1 = torch.squeeze(self.gru_w(torch.reshape(torch.flatten(overlapped_w), (1, 1, self.K * self.K)))[0])
        overlapped_w_2 = torch.mm(beta, torch.mm(self.expand_to_out_channels(w * n_parent, len(indices)), torch.t(beta)))
        gru_out_2 = torch.squeeze(self.grw_w_2(torch.reshape(torch.flatten(overlapped_w_2), (1, 1, self.K * self.K)))[0])
        wAoI_out_1 = torch.flatten(torch.mm(beta, torch.mm(self.expand_to_out_channels(w_AoI_p, len(indices)), torch.t(beta))))
        wAoI_out_2 = torch.flatten(torch.mm(beta, torch.mm(self.expand_to_out_channels(w_AoI_p * n_parent, len(indices)), torch.t(beta))))
        return self.seq(torch.cat((gru_out_1, gru_out_2, wAoI_out_1, wAoI_out_2)))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        seq_len = observations['AoI'].shape[0]
        list_of_obs = [dict(zip(observations, t)) for t in zip(*observations.values())]
        res = torch.stack([self.unbatch_forward(obs) for obs in list_of_obs])

        return res


class SimplifiedWeightedAoIFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            env_target='gowalla',
            linear_dims=[],
            act_func='tanh',
    ):
        env = HybridCentralizedAoICbuEnv(target=env_target)
        self.beta = torch.tensor(env.beta, dtype=torch.int)
        self.p = torch.tensor(env.p, dtype=torch.float32)
        self.o = torch.tensor(env.o, dtype=torch.float32)
        self.K = self.beta.shape[0]
        self.N = self.beta.shape[1]

        self.linear_dims = linear_dims
        self.act_func = act_func
        self.out_dims = self.K
        self.linear_net = None
        self.act_func = {
            'relu': torch.nn.ReLU,
            'relu6': torch.nn.ReLU6,
            'sigmoid': torch.nn.Sigmoid,
            'leaky_relu': torch.nn.LeakyReLU,
            'rrelu': torch.nn.RReLU,
            'tanh': torch.nn.Tanh
        }[act_func]
        if len(self.linear_dims) > 0:
            self.out_dims = self.linear_dims[-1]
        super().__init__(observation_space, features_dim=self.out_dims)

        if len(self.linear_dims) > 0:
            linear_nets = []
            for i in range(len(self.linear_dims)):
                in_dim = self.K if i == 0 else self.linear_dims[i - 1]
                out_dim = self.linear_dims[i]
                linear_nets.append(torch.nn.Linear(in_dim, out_dim))
                linear_nets.append(self.act_func())
            self.linear_net = torch.nn.Sequential(*linear_nets)
        self.gru = GRU(self.K, self.K)
        self.device = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.device is None:
            self.device = observations['AoI'].device
            self.p = self.p.to(self.device)
            self.o = self.o.to(self.device)
            self.beta = self.beta.to(self.device)
            self.gru = self.gru.to(self.device)
            if self.linear_net is not None:
                self.linear_net = self.linear_net.to(self.device)

        seq_len = observations['AoI'].shape[0]
        list_of_obs = [dict(zip(observations, t)) for t in zip(*observations.values())]
        res = torch.stack([self.unbatch_forward(obs) for obs in list_of_obs])
        return res

    def unbatch_forward(self, observation) -> torch.Tensor:
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation)

        indices = torch.argwhere(observation['poi_active'] == 1).squeeze_()
        if indices.ndim == 0:
            return torch.zeros(self.out_dims).to(self.device)
        w = observation['weight'][indices]  # feature of PoIs
        AoI = observation['AoI'][indices]  # feature of PoIs
        w_AoI = torch.reshape(w * AoI, (len(indices), 1))
        beta = self.beta[:, indices] * self.o[:, indices] * torch.reshape(self.p, (self.K, 1)).repeat(1, len(indices))
        w_AoI = torch.reshape(torch.mm(beta, w_AoI), (1, 1, self.K))
        w_AoI, hidden_w = self.gru(w_AoI)
        w_AoI = torch.squeeze(w_AoI)
        if self.linear_net is not None:
            w_AoI = self.linear_net(w_AoI)
        return w_AoI


if __name__ == "__main__":
    evaluation = False
    model = None
    env = None

    # extractor = DynamicPoIFeatureExtractor(env.observation_space)
    env = HybridCentralizedAoICbuEnv(target='gowalla')
    policy_kwargs = dict(
        net_arch=[64, 128, 64],
        features_extractor_class=DynamicPoIFeatureExtractor,
        features_extractor_kwargs=dict(
            env_target='gowalla',
            num_out_channel_feat=64,
            num_out_channel_aoi=32
        )
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=1440 * 1000,
        save_path="env/res/check_point/larger_net/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    if not evaluation:
        model = PPO(
            "MultiInputPolicy",
            env,
            seed=1203,
            policy_kwargs=policy_kwargs,
            verbose=1,
            batch_size=512,
            n_steps=10240,
        )
        model.learn(int(3e4) * 1440, callback=checkpoint_callback)
        np.save('./res/aoi_record/larger_net/PPO_with_KKT_feature_extractor', env.AoI_record)
        model.save('./res/model/larger_net/PPO_with_KKT_feature_extractor')
    else:
        model = PPO.load('env/res/check_point/rl_model_57600000_steps.zip', print_system_info=True)
        model.set_env(env)
        model.learn(int(4e4) * 1440, callback=checkpoint_callback)
        np.save('./res/aoi_record/PPO_with_KKT_feature_extractor_2', env.AoI_record)
        model.save('./res/model/PPO_with_KKT_feature_extractor')

    env.enable_evaluation(True, 28)
    obs = env.reset()
    done = False
    while not done:
        action, _state = model.predict(obs)
        obs, reward, done, info = env.step(action)
    print(env.AoI_record)
