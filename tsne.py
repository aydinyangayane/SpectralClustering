import numpy as np

def pairwise_affinities(X, perplexity):
    n = X.shape[0]
    affinities = np.zeros((n, n))
    for i in range(n):
        distances = np.linalg.norm(X - X[i], axis = 1)
        pij = np.exp(-distances * perplexity)
        pij[i] = 0.0
        sum_pij = np.sum(pij)
        pij = pij / sum_pij
        affinities[i] = (pij + pij[i]) / (2.0 * n)
    return affinities

def low_dimensional_affinities(Y):
    n = Y.shape[0]
    qij = np.zeros((n, n))
    for i in range(n):
        distances = np.linalg.norm(Y - Y[i], axis = 1)
        qij[i] = 1.0 / (1.0 + distances ** 2)
        qij[i, i] = 0.0
        sum_qij = np.sum(qij[i])
        qij[i] = qij[i] / sum_qij
    return qij

def gradient(X, Y, affinities):
    n = X.shape[0]
    gradient = np.zeros_like(Y)
    for i in range(n):
        diff = Y[i] - Y
        pairwise_diff = affinities[i] - low_dimensional_affinities(Y)[i]
        gradient[i] = 2.0 * np.dot(pairwise_diff, diff)
    return gradient

def t_sne(X, perplexity = 30, learning_rate = 200, momentum = 0.5, num_iterations = 1000):
    n = X.shape[0]
    Y = np.random.randn(n, 2) * 1e-4
    for i in range(1, num_iterations + 1):
        affinities = pairwise_affinities(X, perplexity)
        gradient = gradient(X, Y, affinities)
        Y_prev = Y.copy()
        Y = Y + learning_rate * gradient + momentum * (Y - Y_prev)
        if i % 100 == 0:
            cost = np.sum(affinities * np.log(affinities / low_dimensional_affinities(Y)))
            print(f"Iteration {i}, Cost: {cost}")
    return Y