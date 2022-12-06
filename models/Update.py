#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import numpy as np
import random

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class DatasetSplit_noise(Dataset):
    def __init__(self, dataset, idxs, noise_label):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.noise_label = noise_label

    def __len__(self):
        return len(self.idxs)


    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        label = self.noise_label[self.idxs[item]]

        #print(label,label2)#####################################################################
        return image, label, self.idxs[item]



#def generated_noise_data(dataset=dataset_train, idxs=dict_users, noise_degree, num_class):
def generated_noise_data(dataset_train, dict_users, noise_degree, num_class):
    dataset_train_noise = dataset_train

    #print("-------------------------------",dataset_train_noise)
    
    noise_label=list()
    for i in range(len(dataset_train)):
        noise_label.append(dataset_train[i][1])
    #print("-------------------------------",noise_label)
    noise_info={k:[0,len(v)] for k,v in dict_users.items()}
    for i in range(len(noise_degree)):
        for j in range(len(dict_users[i])):
            sample_idx = dict_users[i][j]
            random_change = random.random()
            #print(dataset_train_noise[sample_idx]) 
            if(random_change<noise_degree[i]):
                noise_info[i][0]+=1
                #dataset_train_noise[sample_idx][1]=random.randint(0,num_class)#change label
                noise_label[sample_idx]=random.randint(0,num_class-1)#change label
            # else:
            #     noise_label[sample_idx]=dataset_train[sample_idx][1]#keep label
    #print("-------------------------------",noise_label)
    #return dataset_train_noise
    dict_users_train=list()
    # dict_users_val=list()
    for i in range(len(noise_degree)):
        np.random.shuffle(dict_users[i])
        # idx_train = dict_users[i][(len(dict_users[i])//10):]
        # idx_val = dict_users[i][:(len(dict_users[i])//10)]
        dict_users_train.append(dict_users[i])
        # dict_users_val.append(idx_val)
    return noise_label, dict_users_train,noise_info







class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, noise_label= None ,clean_label=None, learning_rate_iter=None,model=None):#learning_rate_iter=learning_rate_iter)
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.train_idxs=list(idxs)
        self.noise_label=noise_label #整个数据集的label
        self.pseudo_label=copy.deepcopy(noise_label)
        self.ldr_train = DataLoader(DatasetSplit_noise(dataset, idxs, noise_label), batch_size=self.args.local_bs, shuffle=True,pin_memory=True,num_workers=6)
        # self.ldr_eval = DataLoader(DatasetSplit_noise(dataset, idxs_eval, noise_label), batch_size=self.args.local_bs, shuffle=True,pin_memory=True,num_workers=6)
        self.learning_rate_iter=learning_rate_iter
        self.clean_label=clean_label

        #for FedDyn
        # self.global_model_vector = None
        # old_grad = copy.deepcopy(model)
        # old_grad = model_parameter_vector(old_grad)
        # self.old_grad = torch.zeros_like(old_grad)
        # print(len(self.train_idxs))

    def data_info_statistic(self):
        clean_label_distribute = {i: 0 for i in range(self.args.num_classes)}
        noise_label_distribute = {i: 0 for i in range(self.args.num_classes)}
        for sample_id in self.train_idxs:
            try:
                clean_label_distribute[self.clean_label[sample_id].item()] += 1
            except:
                clean_label_distribute[self.clean_label[sample_id]] += 1
            noise_label_distribute[self.noise_label[sample_id]] += 1
        return clean_label_distribute,noise_label_distribute


    def correct_label(self,net):
        net.eval()
        correct=0
        clean_label_distribute, noise_label_distribute=self.data_info_statistic()
        class_accuracy={i: 0 for i in range(self.args.num_classes)}

        with torch.no_grad():
            for batch_idx, (images, labels,sampleid) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logist_probs = net(images)
                prob=torch.softmax(logist_probs,dim=1)
                confidence_score,pse_labels=torch.max(prob,dim=1)
                correct_list=confidence_score.ge(0.9).tolist()
                for sample in range(len(sampleid)):
                    if correct_list[sample]:
                        self.pseudo_label[sampleid[sample].item()]=pse_labels[sample].item()
                        try:
                            if self.clean_label[sampleid[sample].item()].item()==pse_labels[sample].item():
                                correct+=1
                                try:
                                    class_accuracy[self.clean_label[sampleid[sample].item()].item()]+=1
                                except:
                                    class_accuracy[self.clean_label[sampleid[sample].item()]]+=1
                        except:
                            if self.clean_label[sampleid[sample].item()]==pse_labels[sample].item():
                                correct+=1
                                try:
                                    class_accuracy[self.clean_label[sampleid[sample].item()].item()]+=1
                                except:
                                    class_accuracy[self.clean_label[sampleid[sample].item()]]+=1


                        # print(pse_labels[sample])
        return self.pseudo_label, correct/len(self.train_idxs), {k:class_accuracy[k]/v for k,v in clean_label_distribute.items()}


    def train(self, net):
        net.train()
        torch.cuda.empty_cache()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.learning_rate_iter, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels,_) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            # del images,labels,log_probs
            epoch_loss.append(sum(batch_loss)/len(batch_loss))



        # net.eval()
        # eval_epoch_loss = []
        # for iter in range(self.args.local_ep):
        #     batch_loss = []
        #     for batch_idx, (images, labels,_) in enumerate(self.ldr_train):
        #         images, labels = images.to(self.args.device), labels.to(self.args.device)
        #         log_probs = net(images)
        #         loss = self.loss_func(log_probs, labels)
        #         if self.args.verbose and batch_idx % 10 == 0:
        #             print('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #                 iter, batch_idx * len(images), len(self.ldr_train.dataset),
        #                        100. * batch_idx / len(self.ldr_train), loss.item()))
        #         batch_loss.append(loss.item())
        #     eval_epoch_loss.append(sum(batch_loss)/len(batch_loss))


        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)#,  sum(eval_epoch_loss) / len(eval_epoch_loss)

    def train_fedProx(self,net):
        server_model=copy.deepcopy(net)
        server_model.to(self.args.device)
        net.train()
        torch.cuda.empty_cache()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.learning_rate_iter, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, _) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                if batch_idx>0:
                    w_diff = torch.tensor(0., device=self.args.device)
                    for w, w_t in zip(server_model.parameters(), net.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    loss += self.args.fedprox_mu / 2. * w_diff
                loss.backward()
                optimizer.step()
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            # del images,labels,log_probs
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def train_feddyn(self,net):
        # server_model = copy.deepcopy(net)
        self.global_model_vector = model_parameter_vector(net).detach().clone()
        # server_model.to(self.args.device)
        net.train()
        torch.cuda.empty_cache()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.learning_rate_iter, )

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, _) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                if self.global_model_vector != None:
                    v1 = model_parameter_vector(net)
                    loss += self.args.feddyn_alpha/2 * torch.norm(v1 - self.global_model_vector, 2)
                    loss -= torch.dot(v1, self.old_grad)
                loss.backward()
                optimizer.step()
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if self.global_model_vector != None:
            v1 = model_parameter_vector(net).detach()
            self.old_grad = self.old_grad - self.args.feddyn_alpha * (v1 - self.global_model_vector)
            # del images,labels,log_probs

        return net, sum(epoch_loss) / len(epoch_loss)

def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters()]
    return torch.concat(param, dim=0)


