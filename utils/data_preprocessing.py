import numpy as np
import random
from scipy.stats import bernoulli

class bernoulli_function():
    def pmf(x ,p):
        f = p** x * (1 - p) ** (1 - x)
        return f

    def mean(p):
        return p

    def var(p):
        return p * (1 - p)

    def std(p):
        return bernoulli.var(p) ** (1 / 2)

    def rvs(p, size=1):
        rvs = np.array([])
        for i in range(0, size):
            if np.random.rand() <= p:
                a = 1
                rvs = np.append(rvs, a)
            else:
                a = 0
                rvs = np.append(rvs, a)
        return rvs


def generated_noise_data(dataset_train, dict_users, noise_degree, num_class):
    dataset_train_noise = dataset_train
    noise_label = list()
    for i in range(len(dataset_train)):
        noise_label.append(dataset_train[i][1])
    noise_info = {k: [0, len(v)] for k, v in dict_users.items()}
    for i in range(len(noise_degree)):
        for j in range(len(dict_users[i])):
            sample_idx = dict_users[i][j]
            random_change = random.random()
            if (random_change < noise_degree[i]):
                noise_info[i][0] += 1
                noise_label[sample_idx] = random.randint(0, num_class - 1)  # change label
    dict_users_train = list()
    # dict_users_val=list()
    for i in range(len(noise_degree)):
        np.random.shuffle(dict_users[i])
        dict_users_train.append(dict_users[i])
    return noise_label, dict_users_train, noise_info
