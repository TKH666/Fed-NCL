#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = list(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = list(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = list(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users

def non_iid_dirichlet_sampling(y_train, num_classes, p, num_users, seed, alpha_dirichlet=100):
    np.random.seed(seed)
    Phi = np.random.binomial(1, p, size=(num_users, num_classes))  # indicate the classes chosen by each client
    n_classes_per_client = np.sum(Phi, axis=1)
    while np.min(n_classes_per_client) == 0:
        invalid_idx = np.where(n_classes_per_client==0)[0]
        Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
        n_classes_per_client = np.sum(Phi, axis=1)
    Psi = [list(np.where(Phi[:, j]==1)[0]) for j in range(num_classes)]   # indicate the clients that choose each class
    print(Psi)
    num_clients_per_class = np.array([len(x) for x in Psi])
    dict_users = {}
    for class_i in range(num_classes):
        all_idxs = np.where(y_train==class_i)[0]
        p_dirichlet = np.random.dirichlet([alpha_dirichlet] * num_clients_per_class[class_i])
        assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())

        for client_k in Psi[class_i]:
            if client_k in dict_users:
                dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]))
                # dict_users[client_k]=list(dict_users[client_k] )
            else:
                dict_users[client_k] = set(all_idxs[(assignment == client_k)])
                # dict_users[client_k] = list(dict_users[client_k])
    for ck in range(num_users):
        dict_users[ck]=list(dict_users[ck])

    return dict_users

def unbalance_iid(dataset,args):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # Draw from lognormal distribution
    n_data_per_clnt= int(len(dataset)/args.num_users)
    clnt_data_list = (np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=args.unbalanced_sgm, size=args.num_users))
    clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(dataset)).astype(int)
    diff = np.sum(clnt_data_list) - len(dataset)

    # Add/Subtract the excess number starting from first client
    if diff != 0:
        for clnt_i in range(args.num_users):
            if clnt_data_list[clnt_i] > diff:
                clnt_data_list[clnt_i] -= diff
                break
    # num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(args.num_users):
        dict_users[i] = list(np.random.choice(all_idxs, clnt_data_list[i], replace=False))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users


