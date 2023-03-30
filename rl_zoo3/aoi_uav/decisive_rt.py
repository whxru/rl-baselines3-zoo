import numpy as np
import math, time


class DecisiveRT:

    def __init__(self, epsilon_rt_agent):
        self._epsilon_rt_agent = epsilon_rt_agent

    def step(self, start):
        x_cur = self._epsilon_rt_agent.Vu[start]
        x_next, t_first_step = self._epsilon_rt_agent.step(start=start, length=0)
        dist_in_ts = math.ceil(np.linalg.norm(x_next.q - x_cur.q) / self._epsilon_rt_agent.lamb_xoy)
        return x_next.k, t_first_step - 1, t_first_step - 1 + dist_in_ts

    def simu_average_pAoI(self, x0=0):
        # t0 = time.time()
        t = 0
        x = x0
        visited_times = np.zeros(len(self._epsilon_rt_agent.V))
        while True:
            x_next, t_launch, t_arrive = self.step(x)
            visited_times[x0] += min(t_launch, self._epsilon_rt_agent.T - t)
            t += t_arrive
            if t >= self._epsilon_rt_agent.T:
                break
            x = x_next
        w = np.array([v.w for v in self._epsilon_rt_agent.V])
        return np.sum(w / (1 + visited_times))
