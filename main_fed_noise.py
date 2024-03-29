#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import datetime
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
from utils.sampling import mnist_iid,  cifar_iid, non_iid_dirichlet_sampling,unbalance_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from utils.data_preprocessing import generated_noise_data
from nets.models import  Lenet ,  VGG16
from models.Fed import FedAvg, FedAvg_noise_layer_weight, trimmed_mean, agg_feddyn
from models.test import test_img



from models.noise import bernoulli_function


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args)

    # load dataset and split users  #mnist cifar10 cifar100  tiny_imagenet
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=False, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=False, transform=trans_mnist)
        # sample users
        if args.unbalance:
            dict_users = unbalance_iid(dataset_train, args)
        elif args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            y_train = np.array(dataset_train.targets)
            dict_users = non_iid_dirichlet_sampling(y_train=y_train, num_classes=args.num_classes, p=args.p_dirichlet,
                                                    num_users=args.num_users, seed=42,
                                                    alpha_dirichlet=args.alpha_dirichlet)
    elif args.dataset == 'cifar10':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if args.unbalance:
            dict_users = unbalance_iid(dataset_train, args)
        elif args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            y_train=np.array(dataset_train.targets)
            dict_users = non_iid_dirichlet_sampling(y_train=y_train, num_classes=args.num_classes, p=args.p_dirichlet,
                                                    num_users=args.num_users, seed=42, alpha_dirichlet=args.alpha_dirichlet)

    elif args.dataset == 'fashion_mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('./data/fashionmnist', train=True, download=True, transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST('./data/fashionmnist', train=False, download=True, transform=trans_fashion_mnist)
        if args.unbalance:
            dict_users = unbalance_iid(dataset_train, args)
        elif args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            y_train = np.array(dataset_train.targets)
            dict_users = non_iid_dirichlet_sampling(y_train=y_train, num_classes=args.num_classes, p=args.p_dirichlet,
                                                    num_users=args.num_users, seed=42,
                                                    alpha_dirichlet=args.alpha_dirichlet)

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model:  Lenet  VGG16
    if args.model == 'Lenet':
        net_glob = Lenet(args=args).to(args.device)
    elif args.model == 'VGG16':
        net_glob = VGG16(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()


    #settings
    #add noise-----------------------------------------------------------
    np.random.seed(42)
    if args.gaussian:
        mu = args.gaus_mu #0.3
        sigma= args.gaus_sigma #0.4
        noise_degree = (np.random.normal(mu,sigma,args.num_users))  #Guss
    elif args.bernoulli:
        noise_prob = args.bernoulli_p
        noise_degree = bernoulli_function.rvs(1 - noise_prob, args.num_users)  # bernoulli
    # bernoulli=0
    ############

    # bernoulli=1
    
    #avg method---------------------------------------------------------------
    avg_w=args.avg_w
    avg_l=args.avg_l


    experiment=""
    if args.bernoulli==0:
        experiment+="Guss"
    else:
        experiment+="Bernoulli"
    experiment= experiment+("_"+args.dataset + "_" +args.model+ "_iid_" +str(args.iid) + "_num_users" +str(args.num_users)+ "_frac" +str(args.frac)+"_avgwl_"+str(avg_w)+str(avg_l))

    print(experiment)
    exp_time=datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    logpath = f'{args.save_logpath}/{args.dataset}_log'
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    if args.gaussian:
        f1 = open(logpath + '/%s_BL_%s_result_%s_mu_%s_sigma_%s' % (exp_time, str(args.unbalance), experiment, str(mu), str(sigma)) + '.txt','a+')
    elif args.bernoulli:
        f1 = open(logpath + '/%s_BL_%s_result_%s_noise_level_%s' % (exp_time, str(args.unbalance), experiment, str(args.bernoulli_p)) + '.txt', 'a+')

    f1.write(str(args))
    f1.write("\n")
    # f2.write(str(args))
    f1.write(str(noise_degree))
    f1.write("\n")
    noise_idx = [j for j in range(len(noise_degree.tolist())) if noise_degree.tolist()[j] > 0]
    print(noise_idx)
    f1.write(str(noise_idx))
    f1.write("\n")

    # f2.write(str(noise_degree))
    # noise_degree = list()
    # for i in range(args.num_users): 
    #     noise_degree.append(0.5)
    clean_label=copy.deepcopy(dataset_train.targets)
    noise_label, dict_users_train, client_Noise_Info= generated_noise_data(dataset_train, dict_users, noise_degree, args.num_classes)

    f1.write(str(client_Noise_Info))
    # f2.write(str(client_Noise_Info))
    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    test_acc_list, test_loss_list = [], []
    noise_client_list=[]
    true_noise_client_list=[]
    noise_client_count={c:0 for c in range(args.num_users)}
    # print(noise_label)
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    if args.feddyn:
        local_list = [LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[cl], noise_label=noise_label,
                            clean_label=clean_label, learning_rate_iter=args.lr,model=copy.deepcopy(net_glob).to(args.device)) for cl in range(args.num_users)]
        server_stat=copy.deepcopy(net_glob).to(args.device)
    for epoch in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
            client_datalen = []
        if args.feddyn:
            w_locals = [w_glob for i in range(args.num_users)]

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)


        ###########
        weight_list= list()
        data_quality= list()

        learning_rate_iter= args.lr * (args.lr_decay_per_round ** epoch)


        for idx in idxs_users:
            #local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            if (idx in true_noise_client_list) and epoch> args.pl_epoch:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx],
                                     noise_label=noise_label,clean_label=clean_label,
                                    learning_rate_iter=learning_rate_iter)
                pseudo_label,accuracy,class_accuracy=local.correct_label(net=copy.deepcopy(net_glob).to(args.device))
                print(f"CLIENT {idx} pseudo_label accuracy{accuracy},class accuracy {class_accuracy}")
                new_labels=copy.deepcopy(pseudo_label)
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx],
                                     noise_label=new_labels,clean_label=clean_label,
                                    learning_rate_iter=learning_rate_iter)
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            elif args.feddyn:
                local = local_list[idx]
                local.learning_rate_iter = learning_rate_iter
                if epoch==0:
                    w,loss=local.train(net=copy.deepcopy(net_glob).to(args.device))
                else:
                    w,loss=local.train_feddyn(net=copy.deepcopy(net_glob).to(args.device))
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx], noise_label= noise_label, clean_label=clean_label,learning_rate_iter=learning_rate_iter)
                if args.fedprox:
                    w, loss= local.train_fedProx(net=copy.deepcopy(net_glob).to(args.device))
                else:
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients or args.feddyn:
                w_locals[idx] = copy.deepcopy(w)

            else:
                w_locals.append(copy.deepcopy(w))
                client_datalen.append(len(dict_users_train[idx]))
            loss_locals.append(copy.deepcopy(loss))
            # data_quality.append(copy.deepcopy(eval_loss))

        selected_noise=[]  
        for idx in idxs_users:
            selected_noise.append(noise_degree[idx])


        print("selected noise", selected_noise)
        print("selected'clients loss", loss_locals)




        # update global weights
        if args.mode== "fedavg":
            w_glob = FedAvg(w_locals)
        elif args.mode== "fedncl":
            w_glob,weight_dis,wc,noise_clients=FedAvg_noise_layer_weight(args,w_locals, w_glob,epoch,client_datalen,loss_locals) #也要返回异常的client的id，然后记下来，下次抽到整个client，就开始打pseudo label
            print("noise_clients",noise_clients)
            print("noise_clients_idx",[idxs_users[cl] for cl in noise_clients])
            if args.pseudo_label and epoch < args.pl_epoch:
                for cl in noise_clients:
                    if idxs_users[cl] not  in noise_client_list:
                        noise_client_list.append(idxs_users[cl])
                        noise_client_count[idxs_users[cl]] += 1
                    else:
                        noise_client_count[idxs_users[cl]]+=1
                print("noise_clients_all_list", noise_client_list)
                print("noise_clients_all_count", noise_client_count)
                for cl_id, count_num in noise_client_count.items():
                    if (count_num >args.pl_ratio * args.pl_epoch) and (cl_id not in true_noise_client_list):
                        true_noise_client_list.append(cl_id)
                print("final client:", true_noise_client_list)

                detect_acc = 0
                for noise_k in true_noise_client_list:
                    if noise_k in noise_idx:
                        detect_acc += 1
                f1.write(f"Epoch{epoch} final client:" + str(
                    true_noise_client_list) + f"\nEpoch {epoch} Detect Accuracy: {detect_acc / len(noise_idx)}\n")


        else :
            exit("Error Unknown Aggregation Method")


        #w_glob = FedAvg(w_locals)

        # print(w_glob)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        loss_train.append(loss_avg)

        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        test_acc_list.append(acc_test.item())
        test_loss_list.append(loss_test)



        # f1.write(str(loss_test))
        # f1.write('\n')
        # f2.write(str(acc_test))
        # f2.write('\n')


        print("test loss: {:.2f}".format(loss_test))
        print("test accuracy: {:.2f}".format(acc_test))
        f1.write(f"Epoch {epoch} test loss: {loss_test} ,test accuracy: {acc_test} Average loss: {loss_avg}\n")
        f1.flush()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.xlabel("Epoch")
    plt.ylabel('avg_train_loss')
    plt.savefig('./save/{}fed_{}_{}_{}_C{}_iid{}_avg_train_loss.png'.format(exp_time,args.dataset, args.model, args.epochs, args.frac, args.iid))

    plt.figure()
    plt.plot(range(len(test_acc_list)), test_acc_list)
    plt.xlabel("Epoch")
    plt.ylabel('Teat_AUC')
    plt.savefig('./save/{}_{}_{}_fed_{}_{}_{}_C{}_iid{}_test_accuracy.png'.format(exp_time, experiment,str(avg_w)+"+"+str(avg_l),args.dataset, args.model, args.epochs, args.frac,
                                                             args.iid))
    plt.figure()
    plt.plot(range(len(test_loss_list)), test_loss_list)
    plt.xlabel("Epoch")
    plt.ylabel('Teat_loss')
    plt.savefig(
        './save/{}_{}_{}_fed_{}_{}_{}_C{}_iid{}_test_loss.png'.format(exp_time, experiment,str(avg_w)+"+"+str(avg_l), args.dataset, args.model,
                                                                       args.epochs, args.frac,
                                                                       args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    


    f1.close()
    # f2.close()

    #remove_file(r"./log/", r"./loged/")



