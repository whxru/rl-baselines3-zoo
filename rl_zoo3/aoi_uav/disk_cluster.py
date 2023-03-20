import matplotlib.pyplot as plt
import numpy as np
import time


# Efficient approximation algorithms for offline and online unit disk multiple coverage
def disk_coverage_cluster(qk, r):  # returns [the cluster id assigned to each PoI] and {cluster_id: position of each cluster's center}
    res_cluster_centers = {}
    res_cluster_assigned = []

    grid_size = r / np.sqrt(2)
    qk_g = (qk // grid_size).astype(int)
    min_x_g, min_y_g = np.min(qk_g[:, 0]), np.min(qk_g[:, 1])
    max_x_g, max_y_g = np.max(qk_g[:, 0]), np.max(qk_g[:, 1])
    B = np.zeros((max_x_g - min_x_g + 1, max_y_g - min_y_g + 1, 1))
    for i, q in enumerate(qk):
        # q_xoy = np.array([qk[0], qk[1]])
        q_g = np.array([qk_g[i][0] - min_x_g, qk_g[i][1] - min_y_g, 0])
        already_covered = False
        # for offset_x, offset_y in zip(range(-2, 3), range(-2, 3)):
        for offset_x in range(-2, 3):
            for offset_y in range(-2, 3):
                if q_g[0] + offset_x < 0 or q_g[1] + offset_y < 0:
                    continue
                if q_g[0] + offset_x >= B.shape[0] or q_g[1] + offset_y >= B.shape[1]:
                    continue
                if abs(offset_x) == 2 and abs(offset_y) == 2:
                    continue
                grid_g = q_g + np.array([offset_x, offset_y, 0])
                grid_center = (grid_g + np.array([min_x_g + .5, min_y_g + .5, 0])) * grid_size
                if B[grid_g[0], grid_g[1], grid_g[2]] == 1 and np.linalg.norm(grid_center - q) <= r:
                    already_covered = True
                    res_cluster_assigned.append(grid_g[0] * B.shape[0] + grid_g[1])
                    break
            if already_covered:
                break
        if not already_covered:
            grid_g = q_g
            grid_center = (grid_g + np.array([min_x_g + .5, min_y_g + .5, 0])) * grid_size
            B[grid_g[0], grid_g[1], grid_g[2]] = 1
            grid_id = grid_g[0] * B.shape[0] + grid_g[1]
            res_cluster_centers[grid_id] = grid_center
            res_cluster_assigned.append(grid_id)

    return res_cluster_assigned, res_cluster_centers


def test_cluster_func():
    t0 = time.time()
    qk = np.random.random((50, 3)) * 100
    qk[:, -1] = np.zeros(50)
    radius = 30
    cid, centers = disk_coverage_cluster(qk, radius)
    print(f'Alg time: {time.time() - t0} seconds')
    for i, q in enumerate(qk):
        center = centers[cid[i]]
        plt.plot([q[0], center[0]], [q[1], center[1]], c='g')
        plt.scatter(q[0], q[1], c='b')
    for j, circle_center in enumerate(centers.values()):
        circle = plt.Circle((circle_center[0], circle_center[1]), radius, color=(.1, .1, .1, .1))
        plt.gca().add_patch(circle)
        plt.scatter(circle_center[0], circle_center[1], c='r')
    plt.show()


if __name__ == '__main__':
    test_cluster_func()
