import numpy as np
import math, time


class DecisiveRT:

    def __init__(self, epsilon_rt_agent):
        self._epsilon_rt_agent = epsilon_rt_agent

    def step(self, start):
        x_cur = self._epsilon_rt_agent.Vu[start]
        x_next, t_arrive = self._epsilon_rt_agent.step(start=start, length=0)
        return x_next, t_arrive - math.ceil(np.linalg.norm(x_next.q - x_cur.q) / self._epsilon_rt_agent.lamb_xoy), t_arrive

    def simu_average_pAoI(self, x0=0):
        # t0 = time.time()
        t = 0
        visited_times = np.zeros(len(self._epsilon_rt_agent.V))
        x_next, t_launch, t_arrive = self.step(x0)
        while True:
            t += t_arrive
            if t >= self._epsilon_rt_agent.T:
                break
            visited_times[x_next.k] += 1
            x_next, t_launch, t_arrive = self.step(x_next.k)
        w = np.array([v.w for v in self._epsilon_rt_agent.V])
        return np.sum(w / (1 + visited_times))
