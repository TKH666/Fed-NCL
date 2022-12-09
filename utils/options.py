#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=150, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")    #0.1##########################
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")  #10
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")  #0.01
    parser.add_argument('--lr_decay_per_round', type=float, default=0.99, help="learning rate")  #0.998
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")  #0.5
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='VGG16', help='model name')  #LenetMnist  mlp   VGG16Cifar10  ResNet18Cifar10 ResNet18Cifar100  #ResNet18Tiny, VGG16Cifar10#################

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")              # mnist cifar10 cifar100  tiny_imagenet fashion_mnist ##########################
    parser.add_argument('--iid', action='store_true',default=False, help='whether i.i.d or not')         #################################
    parser.add_argument('--alpha_dirichlet', type=int,default=1000, help='whether i.i.d or not')         #################################
    parser.add_argument('--p_dirichlet', type=float,default=1, help='whether i.i.d or not')         #################################
    parser.add_argument('--unbalance', type=bool, default=False,
                        help="unbalanced_sgm")  # 10 100 200                ##############################
    parser.add_argument('--unbalanced_sgm', type=float, default=0.3, help="unbalanced_sgm")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")             #  10 100 200                ##############################
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping') 
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--std_num', type=float, default=1, help='挑选标准差')
    parser.add_argument('--penalty', type=int, default=50, help='惩罚因子')
    parser.add_argument('--start_penalty', type=int, default=10, help='惩罚因子')
    parser.add_argument('--pl_ratio', type=float, default=0.5, help='大于pl_ratio*pl_epoch的比例')
    parser.add_argument('--save_logpath', type=str, default='/home/tamkahou/Documents/fed/federated-learning-master_noise_cvpr', help='log save path')
    parser.add_argument('--all_clients',  default=False, action='store_true', help='aggregation over all clients')  ##
    parser.add_argument('--save_checkpoint', type=bool, default=True, help='whether save the weight distance and weight matrix or not')

    parser.add_argument('--bernoulli', type=bool, help='1 is bernoulli; 0 for Guss')
    parser.add_argument('--bernoulli_p', type=float, default=0.7, help='clean client probability for bernoulli Distribution')
    parser.add_argument('--gaussian', type=bool,  help='1 is bernoulli; 0 for Guss')
    parser.add_argument('--gaus_mu', type=float, default=0.3, help='mean for Truncated Gaussian Distribution')
    parser.add_argument('--gaus_sigma', type=float, default=0.4, help='variance for Truncated Gaussian Distribution')
    parser.add_argument('--avg_w', type=int, default=2, help='2 for layer weight agg; 1 for dis weight agg; 0 for FedAvg, 3 for FedAvg with noise client detection')
    parser.add_argument('--avg_l', type=int, default=0, help='1 for loss weight agg; 0 for FedAvg')
    parser.add_argument('--pl_epoch', type=int, default=60, help='什么时候介入pseudo label')

    parser.add_argument('--reweight_classifier', type=bool, default=False, help='是否只对classifier 进行layer reweight')
    parser.add_argument('--layer_agg', type=bool, default=True, help='使用layer divergence作为weight还是整个model的divergence作为weight')
    parser.add_argument('--exp_note', type=str, default="fedprox测试利用上一个global mode avg, 1 frac，guss 0.3/0.4 noise,10个client，VGG16Cifar10，fedavg,layer agg 并且增大20差距,大为标签之后为1.1-1.5，70 psudo label,150 epoch,,Data cifar10，测试0，1正态分布筛选", help='exp note')

    parser.add_argument('--fedprox_mu', type=float, default=0.01, help='FedProx mu')
    parser.add_argument('--fedprox', type=bool, default=False, help='FedProx ,记得也修改 avg_w,avg_l 都是0')

    parser.add_argument('--mode', type=str, default='fedncl', help='FedProx ,记得也修改 avg_w,avg_l 都是0')
    parser.add_argument('--pseudo_label', type=bool, default=True, help='是否介入pseudo label')


    parser.add_argument('--feddyn_alpha', type=float, default=0.01, help='FedProx mu')
    parser.add_argument('--feddyn', type=bool, default=False, help='FedProx ,记得也修改 avg_w,avg_l 都是0')
    args = parser.parse_args()
    return args

