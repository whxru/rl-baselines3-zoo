import time

import numpy as np
import math, random, heapq
import matplotlib.pyplot as plt
from rl_zoo3.aoi_uav.decisive_rt import DecisiveRT
from multiprocessing import Pool


class Vertex:

    def __init__(self, w=1, pi=1, is_virtual=False, index=0, q: np.ndarray=None):
        self.w = w
        self.q = q
        self._index = index
        self.is_virtual = is_virtual
        self._neighbors = []
        self._connected_PoIs = []

        self.pi = pi

    @property
    def k(self):
        return self._index

    @property
    def deg(self):
        return len(self._neighbors)

    def is_neighbor(self, v):
        return v in self._neighbors

    def is_for_poi_pair(self, v1, v2):
        assert not v1.is_virtual and not v2.is_virtual
        self._connected_PoIs.append(v1)
        self._connected_PoIs.append(v2)

    def leads_to_poi(self, start_v):
        assert not start_v.is_virtual and len(self._connected_PoIs) == 2
        return self._connected_PoIs[0 if start_v == self._connected_PoIs[1] else 1]

    def connect_to(self, v):
        if self.is_neighbor(v):
            assert v.is_neighbor(self)
            return False
        self._neighbors.append(v)
        v.connect_to(self)
        return True

    def __eq__(self, other):
        return self.k == other.k


class EpsilonRT:

    def __init__(self, V, E, T=1800, lamb_xoy=9, epsilon=None):
        self.V = V
        self.E = E
        self.T = T
        self.lamb_xoy = lamb_xoy

        self.epsilon = epsilon
        self._cal_vals()
        self.Vu, self.Eu = self.build_uDTG(self.epsilon)
        self.p = EpsilonRT.metropolis_hastings(self.Vu, self.Eu)

    def build_uDTG(self, epsilon):
        Vu = []
        Eu = []
        one_minus_epsilon_mag_E_pos = 1 - epsilon * len(self.E_positive)
        for v in self.V:
            v_pi = one_minus_epsilon_mag_E_pos * np.sqrt(v.w) / self.sum_sqrt_w
            Vu.append(Vertex(is_virtual=False, index=v.k, q=v.q, pi=v_pi))

        for v1, v2 in self.E:
            v1 = Vu[v1.k]
            v2 = Vu[v2.k]
            d = self.d[v1.k, v2.k]
            if d == -1:
                continue
            elif d == 0:
                Eu.append((v1, v2))
                v1.connect_to(v2)
            else:  # d >= 1
                delta_q = (v2.q - v1.q) / (1 + d)
                pi_v = epsilon / d
                for n in range(1, 1 + d):
                    v_intermediate = Vertex(pi=pi_v, is_virtual=True, index=len(Vu), q=v1.q+n*delta_q)
                    v_intermediate.is_for_poi_pair(v1, v2)
                    v_last = Vu[-1]
                    Vu.append(v_intermediate)
                    if n == 1:
                        v1.connect_to(v_intermediate)
                        Eu.append((v1, v_intermediate))
                    if n == d:
                        v_intermediate.connect_to(v2)
                        Eu.append((v_intermediate, v2))
                    if 1 < n <= d:
                        v_last.connect_to(v_intermediate)
                        Eu.append((v_last, v_intermediate))
        return Vu, Eu

    def _cal_vals(self):
        self.E_positive = []
        self.E_one = []
        self.E_two = []
        self.d_max = -2
        self.d = np.zeros((len(self.V), len(self.V))).astype(int)
        for v1, v2 in self.E:
            d = np.ceil(np.linalg.norm(v1.q - v2.q) / self.lamb_xoy) - 1
            # print(f'Node {v1.k} and {v2.k}: distance = {np.linalg.norm(v1.q - v2.q)} / position {v1.q} and {v2.q}')
            self.d[v1.k, v2.k] = d
            self.d[v2.k, v1.k] = d
            self.d_max = max(self.d_max, d)
            if d == 1:
                self.E_one.append((v1, v2))
                self.E_positive.append((v1, v2))
            elif d > 1:
                self.E_two.append((v1, v2))
                self.E_positive.append((v1, v2))
        self.sum_sqrt_w = np.sum([np.sqrt(v.w) for v in self.V])
        if self.epsilon is None:
            self._pick_epsilon()
        self.one_minus_epsilon_mag_E_pos = 1 - self.epsilon * len(self.E_positive)

    def _pick_epsilon(self, padding=1e-5, T_exp=2000):
        t0 = time.time()
        K = 50
        b = 1 / len(self.E_positive)
        # candidate_epsilons_left = [padding]
        # candidate_epsilons_right = [b - padding]
        # sampling_func = lambda x: 1 / x / (b - x)
        # y_max = sampling_func(padding)
        # y_min = sampling_func(b / 25)
        # num_sample_on_slope = int(K * 0.3)
        # num_sample_on_plane = K - num_sample_on_slope * 2
        # delta_y = (y_max - y_min) / (num_sample_on_slope - 1)
        # for k in range(num_sample_on_slope - 1):
        #     progress = (k + 1) / num_sample_on_slope
        #     y = y_max * (1 - progress) + y_min * progress
        #     x = .5 * (b - np.sqrt(b**2 - 4 / y))
        #     candidate_epsilons_left.append(x)
        #     candidate_epsilons_right.insert(0, b - x)
        # delta_x = (candidate_epsilons_right[0] - candidate_epsilons_left[-1]) / (num_sample_on_plane - 1)
        # candidate_epsilons_median = np.linspace(candidate_epsilons_left[-1] + delta_x, candidate_epsilons_right[0], num=num_sample_on_plane, endpoint=False).tolist()
        # candidate_epsilons = candidate_epsilons_left + candidate_epsilons_median + candidate_epsilons_right
        # Plot the candidate epsilons
        # plt.scatter(candidate_epsilons, [sampling_func(x) for x in candidate_epsilons])
        # plt.show()
        # plt.scatter(candidate_epsilons_left, [sampling_func(x) for x in candidate_epsilons_left])
        # plt.show()
        candidate_epsilons = np.linspace(padding, 1 / len(self.E_positive), K, endpoint=False)
        candidate_dRT_agents = [DecisiveRT(EpsilonRT(self.V, self.E, self.T, self.lamb_xoy, epsilon)) for epsilon in candidate_epsilons]

        # [ICML'2013] Almost Optimal Exploration in Multi-Armed Bandits
        global arm_reward
        def arm_reward(arm):
            return candidate_dRT_agents[arm].simu_average_pAoI()
        arms = list(range(K))
        empirical_sums = np.zeros(K)
        sample_counters = np.zeros(K)
        for r in range(0, math.ceil(np.log2(K))):
            sampled_times = math.floor(T_exp / len(arms) / math.ceil(np.log2(K)))
            print(f'Sampling times: {sampled_times} for arm set {arms}')
            for arm in arms:
                with Pool() as pool:
                    empirical_sums[arm] += np.sum(pool.map(arm_reward, [arm] * sampled_times))
                    pool.close()
                    pool.join()
                # for _ in range(sampled_times):
                #     empirical_sums[arm] += candidate_dRT_agents[arm].simu_average_pAoI()
            sample_counters[arms] += sampled_times
            num_arms_to_save = math.ceil(len(arms) / 2)
            arms = heapq.nsmallest(num_arms_to_save, arms, key=lambda arm: empirical_sums[arm] / sample_counters[arm])
        best_arm = arms[0]
        plt.plot(np.arange(K), empirical_sums / sample_counters)
        plt.scatter(best_arm, empirical_sums[best_arm] / sample_counters[best_arm])
        plt.title('Candidate epsilons $\\to$$ simulated Av PAoI')
        plt.show()
        self.epsilon = candidate_epsilons[best_arm]
        print(f'Hyperparamter optimization time: {time.time() - t0} seconds')

    @staticmethod
    def metropolis_hastings(Vu, Eu):
        p = np.zeros((len(Vu), len(Vu)))
        for v1, v2 in Eu:
            if v1 != v2:
                p[v1.k, v2.k] = 1 / v1.deg * min(1.0, (v2.pi * v1.deg) / (v1.pi * v2.deg))
                p[v2.k, v1.k] = 1 / v2.deg * min(1.0, (v1.pi * v2.deg) / (v2.pi * v1.deg))
        for v in Vu:
            p[v.k, v.k] = 1 - np.sum(p[v.k])
            assert p[v.k, v.k] >= 0
        return p

    # length == 0 means stops until a new vertex (no matter virtual or not) is met
    # length > 0 means samples for a fixed number of steps
    def step(self, start: int, length=1):
        x = self.Vu[start]
        step_count = 0
        while True:
            step_count += 1
            x_next = np.random.choice(self.Vu, p=self.p[x.k])
            # if x_next != x:
            #     print(f'{x.k} to {x_next.k}')
            x = x_next
            if length == 0 and not x.k == start:
                x = x.leads_to_poi(self.Vu[start])
                break
            if step_count == length or step_count >= self.T:
                break
        return x, step_count

    @staticmethod
    def filter_edges(w, qk, x0=0):
        visited_nodes = [x0]
        x_last = x0
        K = len(w)
        k_large_w = heapq.nlargest(K // 10, range(K), key=lambda k: w[k])

        visited_seq = {k: 0 for k in k_large_w}
        E = []
        while len(visited_nodes) < K:
            d_min = np.inf
            nearest_node = None
            for k in range(K):
                if k == x_last:
                    continue
                if k in visited_nodes:
                    continue
                d_now = np.linalg.norm(qk[k] - qk[x_last])
                if d_now < d_min:
                    d_min = d_now
                    nearest_node = k
            visited_nodes.append(nearest_node)
            E.append((x_last, nearest_node))
            x_last = nearest_node

            if nearest_node in visited_seq:
                visited_seq[nearest_node] = len(E) - 1

        E.append((x_last, x0))

        def dis_div_hop(k, hop):
            if hop % K == 0:
                return np.inf
            target_k = E[(visited_seq[k] + hop) % len(E)][1]
            return np.ceil(np.linalg.norm(qk[k] - qk[target_k])) / hop
        for k in k_large_w:
            target_step = heapq.nsmallest(1, range(K // 4, K * 3 // 4), lambda hop: dis_div_hop(k, hop))[0]
            # E.append((k, E[(visited_seq[k] + target_step) % len(E)][1]))
            # print(f'Connect Node {E[-1]}')
        return E

    @staticmethod
    def make_graph(w, qk: [np.ndarray], E: [(int, int)]):
        res_V = []
        res_E = []
        for i, q in enumerate(qk):
            res_V.append(Vertex(w=w[i], is_virtual=False, index=i, q=q))
        for i, j in E:
            v1, v2 = res_V[i], res_V[j]
            res_E.append((v1, v2))
            v1.connect_to(v2)
        return res_V, res_E

    def plot_func(self, func, title, use_recip=True):
        max_epsilon = 1 / len(self.E_positive) - 1e-8
        min_epsilon = 1e-8
        epsilon = np.arange(min_epsilon, max_epsilon, max_epsilon / 100)
        vals = [func(1 / ep if use_recip else ep) for ep in epsilon]
        plt.plot(epsilon, vals)
        plt.title(title)
        plt.show()

    @staticmethod
    def display_graph(V: [Vertex], E: [(Vertex, Vertex)], K=None):
        for v1, v2 in E:
            plt.plot([v1.q[0], v2.q[0]], [v1.q[1], v2.q[1]], c='g')
        for v in V:
            plt.scatter(v.q[0], v.q[1], c='r' if v.is_virtual else 'b')
        if K is not None:
            for k in range(K):
                plt.text(V[k].q[0], V[k].q[1], str(k))
        plt.show()


if __name__ == '__main__':
    K = 50
    w = np.random.random(K)
    qk = np.random.random((K, 3)) * 100
    qk[:, 2] = np.zeros(K)
    # E = []
    # while len(E) <= 40:
    #     v1, v2 = np.random.choice(K, 2)
    #     if v1 == v2 or (v1, v2) in E or (v2, v1) in E:
    #         continue
    #     E.append((v1, v2))
    E = EpsilonRT.filter_edges(w, qk)
    V, E = EpsilonRT.make_graph(w, qk, E)
    G = EpsilonRT(V=V, E=E)
    # print(G.epsilon, 1 / G.candidate_epsilon_recip)
    # G.display_graph(G.Vu, G.Eu, len(G.V))
    G.display_graph(G.V, G.E, len(G.V))
