import numpy as np

def make_fitness_function(func):
    def temp(array):
        result = []
        for element in array:
            result.append(func(element))
        return np.array(result)  
    return temp

def iterarSRO(maxIter, iter, dim, population, fitness, best, fo, lb, ub):
    # 1. Algorithm parameters
    N = len(population)
    maxIter = maxIter
    ff = make_fitness_function(fo)

    k = 2e-5
    K_com = 2
    groupCount = 4
    C = 1.0

    # noImproveIter = 0 #vel[0]

    # 2. Group index initialization
    groupSize = N // groupCount
    remainder = N % groupCount
    groups = []
    start_idx = 0

    for g in range(groupCount):
        current_size = groupSize + (remainder if g == groupCount - 1 else 0)
        end_idx = start_idx + current_size
        groups.append(np.arange(start_idx, end_idx))
        start_idx = end_idx

    # 3. Variable initialization
    F = 2 * np.random.randint(0, 2, N) - 1
    omega = np.zeros(N)
    delta = np.zeros((N, dim))
    angle = np.zeros((N, dim))
    ship_vel = np.zeros((N, dim))

    # 4. Group communication
    if iter % 20 == 0:
        for g in range(groupCount):
            indices = groups[g]

            g_pos = population[indices, :]
            g_fitness = fitness[indices]
            g_best = g_pos[np.argmin(g_fitness)]

            worst_indices = np.argsort(-g_fitness)[:int(np.ceil(len(indices) / 3))]
            c1 = K_com * np.random.rand()
            c2 = K_com * np.random.rand()

            for w in worst_indices:
                idx = indices[w]

                population[idx, :] = np.clip(population[idx, :] + (
                    c1 * (g_best - population[idx, :]) +
                    c2 * (best  - population[idx, :])
                ), lb, ub)

    # 5. Ship movement
    for i in range(N):
        if np.linalg.norm(population[i, :]) > 0 and np.linalg.norm(best) > 0:
            cos_angle = np.dot(population[i, :], best) / (np.linalg.norm(population[i, :]) * np.linalg.norm(best))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle[i, :] = np.arccos(cos_angle)

    c = -2 + 4 * np.random.rand(N)
    delta = (c * F)[:, None] * angle
    omega = omega + k * delta[:, 0]

    ship_vel += (omega[:, None]) * np.random.randn(N, dim) # * (ub - lb)

    newPopulation = np.clip(population + C * (ship_vel + delta * np.random.randn(N, dim) * (best - population)), lb, ub)
    # indices = ff(newPopulation) < ff(population)
    # population[indices] = newPopulation[indices]

    # noImproveIter = 0 if np.min(ff(population)) < ff(best) else noImproveIter + 1
    # vel[0] = noImproveIter

    return population