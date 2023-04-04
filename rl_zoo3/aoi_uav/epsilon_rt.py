import time

import numpy as np
import math, random, heapq
import matplotlib.pyplot as plt
from multiprocessing import Pool
from rl_zoo3.aoi_uav.disk_cluster import disk_coverage_cluster

class Vertex:

    def __init__(self, w=1, pi=1, is_virtual=False, index=0, q: np.ndarray=None):
        self.w = w
        self.q = q
        self._index = index
        self.is_virtual = is_virtual
        self.neighbors = []

        self.pi = pi

    @property
    def k(self):
        return self._index

    @property
    def deg(self):
        return len(self.neighbors)

    def is_neighbor(self, v):
        return v in self.neighbors

    def connect_to(self, v):
        if self.is_neighbor(v):
            assert v.is_neighbor(self)
            return False
        self.neighbors.append(v)
        return True

    def __eq__(self, other):
        return self.k == other.k


class EpsilonRT:

    def __init__(self, V, E, T=1800, lamb_xoy=9, epsilon=None, seed=1997, x0=0):
        self.V = V
        self.E = E
        self.T = T
        self.lamb_xoy = lamb_xoy
        self.x0 = x0

        self.K = len(V)
        self.w = np.array([v.w for v in self.V])
        self.qk = np.array([v.q for v in self.V])

        self.epsilon = epsilon
        self.rs = np.random.RandomState(seed=seed)
        self._cal_vals()
        print(f'Epsilon: {self.epsilon}')

    def _cal_vals(self):
        self.E_positive = []
        self.E_one = []
        self.E_two = []
        self.d_max = -2
        self.d = np.zeros((self.K, self.K)).astype(int) - 1
        self.o = np.zeros((self.K, self.K))
        self.p_raw = np.zeros((self.K, self.K))
        for v1, v2 in self.E:
            d = np.ceil(np.linalg.norm(v1.q - v2.q) / self.lamb_xoy) - 1
            # print(f'Node {v1.k} and {v2.k}: distance = {np.linalg.norm(v1.q - v2.q)} / position {v1.q} and {v2.q}')
            self.d[v1.k, v2.k] = d
            self.o[v1.k, :] += d + 1
            self.d_max = max(self.d_max, d)
            if d == 1:
                self.E_one.append((v1, v2))
                self.E_positive.append((v1, v2))
            elif d > 1:
                self.E_two.append((v1, v2))
                self.E_positive.append((v1, v2))

        self.o = (1 + self.d) / self.o
        for v1, v2 in self.E:
            # assert self.o[v1.k, v2.k] > 0 and self.o[v2.k, v1.k] > 0
            if self.o[v2.k, v1.k] == 0:
                assert self.o[v1.k, v2.k] > 0
                self.p_raw[v1.k, v2.k] = 1 / self.o[v1.k, v2.k]
            else:
                self.p_raw[v1.k, v2.k] = 1 / self.o[v1.k, v2.k] * min(1, np.sqrt(v2.w) * self.o[v1.k, v2.k] / np.sqrt(v1.w) / self.o[v2.k, v1.k])

        # Search the bound of epsilon
        right_bound = 2
        search_speed = 0.9
        while True:
            mid_bound = right_bound * search_speed
            p_now = mid_bound * self.p_raw
            pi_now = self._pi(p_now)
            feasible_now = True
            for v in self.V:
                if pi_now[v.k] < np.dot(pi_now, p_now[:, v.k]):
                    feasible_now = False
                    break
            right_bound = mid_bound
            if feasible_now:
                self.epsilon_bound = right_bound
                break
        # Tune a good epsilon within the range
        if self.epsilon is None or self.epsilon > self.epsilon_bound:
            self._pick_epsilon()
        # self.epsilon = self.epsilon_bound / 2
        self.p = self.p_raw * self.epsilon
        for k in range(self.K):
            self.p[k, k] = 1 - np.sum(self.p[k, :])
        self.pi = self._pi(self.p)
        self.sum_pi = np.sum(self.pi)
        for v in self.V:
            v.pi = self.pi[v.k]
        # Check optimality gap
        pi_opt = np.sqrt(self.w) / np.sum(np.sqrt(self.w))
        print(f'Opt: {np.sum(self.w / pi_opt) / self.K}, ours: {np.sum(self.w / self.pi) / self.K}')

    def _pi(self, p):
        param_agg = np.ones(self.K)  # 1 + sum_{k\neq k'}d_{k,k'}p_{k,k'}^\text{raw} for each PoI k
        for v1, v2 in self.E:
            param_agg[v1.k] += self.d[v1.k, v2.k] * p[v1.k, v2.k]
        pi_res = np.sqrt(self.w / param_agg) / np.sum(np.sqrt(self.w * param_agg))
        return pi_res

    def _pick_epsilon(self, padding=1e-5, T_exp=1000):
        t0 = time.time()
        K = 50
        b = self.epsilon_bound
        candidate_epsilons_left = [padding]
        candidate_epsilons_right = [b - padding]
        sampling_func = lambda x: 1 / x / (b - x)
        y_max = sampling_func(padding)
        y_min = sampling_func(b / 25)
        num_sample_on_slope = int(K * 0.3)
        num_sample_on_plane = K - num_sample_on_slope * 2
        delta_y = (y_max - y_min) / (num_sample_on_slope - 1)
        for k in range(num_sample_on_slope - 1):
            progress = (k + 1) / num_sample_on_slope
            y = y_max * (1 - progress) + y_min * progress
            x = .5 * (b - np.sqrt(b**2 - 4 / y))
            candidate_epsilons_left.append(x)
            candidate_epsilons_right.insert(0, b - x)
        delta_x = (candidate_epsilons_right[0] - candidate_epsilons_left[-1]) / (num_sample_on_plane - 1)
        candidate_epsilons_median = np.linspace(candidate_epsilons_left[-1] + delta_x, candidate_epsilons_right[0], num=num_sample_on_plane, endpoint=False).tolist()
        candidate_epsilons = candidate_epsilons_left + candidate_epsilons_median + candidate_epsilons_right
        # Plot the candidate epsilons
        # plt.scatter(candidate_epsilons, [sampling_func(x) for x in candidate_epsilons])
        # plt.show()
        # plt.scatter(candidate_epsilons_left, [sampling_func(x) for x in candidate_epsilons_left])
        # plt.show()
        # candidate_epsilons = np.linspace(padding, self.epsilon_bound, K, endpoint=True)

        # [ICML'2013] Almost Optimal Exploration in Multi-Armed Bandits
        global arm_reward
        def arm_reward(epsilon, metric='AoI'):
            p = self.p_raw * epsilon
            for k in range(self.K):
                p[k, k] = 1 - np.sum(p[k, :])
            x = self.x0
            t = 0
            h = np.ones(self.K)
            sum_h = np.zeros(self.K)
            visited_times = np.ones(self.K)
            while t < self.T:
                visited_times[x] += 1
                x_next = np.random.choice(self.K, p=p[x])
                delta_t = 1 if x_next == x else self.d[x, x_next] + 1
                for _ in range(delta_t):
                    sum_h += h
                    h += 1
                h[x_next] = 1

                t += delta_t
                x = x_next
            if metric == 'peak_AoI':
                return np.sum(self.w / visited_times * self.T)
            if metric == 'AoI':
                return np.sum(self.w * sum_h / self.T)

        arms = list(range(K))
        empirical_sums = np.zeros(K)
        sample_counters = np.zeros(K)
        for r in range(0, math.ceil(np.log2(K))):
            sampled_times = math.floor(T_exp / len(arms) / math.ceil(np.log2(K)))
            print(f'Sampling times: {sampled_times} for arm set {arms}')
            for arm in arms:
                with Pool(processes=2) as pool:
                    empirical_sums[arm] += np.sum(pool.map(arm_reward, candidate_epsilons))
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
        print(f'Epsilon: {self.epsilon}, optimization time: {time.time() - t0} seconds')

    def step(self, q_current: np.ndarray, info):
        poi_covered = info['covered']
        r = info['estimated_r']
        empirical_pi = info['visited_times'] / np.sum(info['visited_times']) * self.sum_pi

        # Calculate the inter-cluster transition probability p_aggregate
        poi_cluster_id = np.zeros(self.K).astype(int) - 1
        uncovered_cluster_id, cluster_centers = disk_coverage_cluster(self.qk[poi_covered == 0], r)
        poi_cluster_id[poi_covered == 0] = uncovered_cluster_id
        cluster_centers[-1] = q_current
        num_clusters = len(cluster_centers.keys())
        # poi_cluster_id[poi_covered == 1] = num_clusters - 1
        cluster_ids = list(cluster_centers.keys())
        cluster_id_to_idx = {cluster_ids[n]: n for n in range(num_clusters)}

        p_aggregate = np.zeros((num_clusters, num_clusters))
        adj = np.diag([1] * num_clusters).astype(int)
        for v1, v2 in self.E:
            cid1, cid2 = poi_cluster_id[v1.k], poi_cluster_id[v2.k]
            idx1, idx2 = cluster_id_to_idx[cid1], cluster_id_to_idx[cid2]
            p_aggregate[idx1, idx2] += self.p[v1.k, v2.k]
            adj[idx1, idx2] = 1
        for v in self.V:
            idx = cluster_id_to_idx[poi_cluster_id[v.k]]
            p_aggregate[idx, idx] += self.p[v.k, v.k]
        for k in range(num_clusters):
            p_aggregate[k, :] /= np.sum(p_aggregate[k, :])
        # adj_n_hops = [adj]
        # adj_final = adj
        # for _ in range(4):
        #     adj_n_hops.append(np.dot(adj, adj[-1]))
        #     adj_final += adj_n_hops[-1]

        # Calculate the empirical pi for each cluster
        empirical_pi_aggregate = np.zeros(num_clusters)
        expected_pi_aggregate = np.zeros(num_clusters)
        for v in self.V:
            cid = poi_cluster_id[v.k]
            idx = cluster_id_to_idx[cid]
            empirical_pi_aggregate[idx] += empirical_pi[v.k]
            expected_pi_aggregate[idx] += self.pi[v.k]

        delta_pi_aggregate = empirical_pi_aggregate - expected_pi_aggregate
        most_overvisted_cluster = np.argmax(delta_pi_aggregate * adj[-1])
        overflow_trans_prob = p_aggregate[most_overvisted_cluster, most_overvisted_cluster] * 0.4
        p_aggregate[most_overvisted_cluster, most_overvisted_cluster] -= overflow_trans_prob
        total_lack_pi = np.sum(np.abs(delta_pi_aggregate) * (delta_pi_aggregate <= 0) * (adj[-1] == 1))
        if total_lack_pi > 0:
            for m in range(num_clusters):
                if m == most_overvisted_cluster or adj[-1, m] == 0 or delta_pi_aggregate[m] >= 0:
                    continue
                p_aggregate[most_overvisted_cluster, m] += overflow_trans_prob * (-delta_pi_aggregate[m]) / total_lack_pi

            cluster_next = self.rs.choice(num_clusters, p=p_aggregate[-1, :])
            q_next = cluster_centers[cluster_ids[cluster_next]]
        else:  # if no PoI is covered
            q_next = cluster_centers[cluster_ids[int(np.argmin(delta_pi_aggregate))]]
        return q_next, max(1, np.ceil(np.linalg.norm(q_next - q_current) / self.lamb_xoy))

    @staticmethod
    def build_edges(w, qk, x0=0):
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
            # E.append((nearest_node, x_last))
            x_last = nearest_node

            if nearest_node in visited_seq:
                visited_seq[nearest_node] = len(E) - 1

        E.append((x_last, x0))
        # E.append((x0, x_last))

        def dis_div_hop(k, hop):
            if hop % K == 0:
                return np.inf
            target_k = E[(visited_seq[k] + hop) % len(E)][1]
            return np.ceil(np.linalg.norm(qk[k] - qk[target_k])) / hop
        for k in k_large_w:
            target_step = heapq.nsmallest(1, range(K // 4, K * 3 // 4), lambda hop: dis_div_hop(k, hop))[0]
            v1, v2 = k, E[(visited_seq[k] + target_step) % len(E)][1]
            if (v1, v2) not in E and (v2, v1) not in E:
                E.append((v1, v2))
                E.append((v2, v1))
            # print(f'Connect Node {E[-1]}')
        return E

    @staticmethod
    def make_graph(w, qk: [np.ndarray], E: [(int, int)]):
        res_V = []
        res_E = []
        for i, q in enumerate(qk):
            res_V.append(Vertex(w=w[i], is_virtual=False, index=i, q=q))
        for i, j in E:
            if i == 96:
                pass
            v1, v2 = res_V[i], res_V[j]
            res_E.append((v1, v2))
            v1.connect_to(v2)
        return res_V, res_E

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
    np.random.seed(2024)
    K = 50
    w = np.random.random(K)
    qk = np.random.random((K, 3)) * 100
    qk[:, 2] = np.zeros(K)

    E = EpsilonRT.build_edges(w, qk)
    V, E = EpsilonRT.make_graph(w, qk, E)
    G = EpsilonRT(V=V, E=E, epsilon=0.091053067655531)
    G.display_graph(G.V, G.E, len(G.V))
