#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import time

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def trimmed_mean(w, args):
    if args.bernoulli == 1:
        number_to_consider = int(args.num_users * args.bernoulli_p * args.frac) - 1
    else:
        number_to_consider = int(args.num_users * (1 - 0.5) * args.frac) - 1
    print(number_to_consider)
    w_avg = copy.deepcopy(w[0])
    # for k in w_avg.keys():
    #     tmp = []
    #     for i in range(len(w)):
    #         tmp.append(w[i][k].cpu().numpy())  # get the weight of k-layer which in each client
    #     tmp = np.array(tmp)
    #     med = np.median(tmp, axis=0)
    #     new_tmp = []
    #     for i in range(len(tmp)):  # cal each client weights - median
    #         new_tmp.append(tmp[i] - med)
    #     new_tmp = np.array(new_tmp)
    #     good_vals = np.argsort(abs(new_tmp), axis=0)[:number_to_consider]
    #     good_vals = np.take_along_axis(new_tmp, good_vals, axis=0)
    #     k_weight = np.array(np.mean(good_vals) + med)
    #     w_avg[k] = torch.from_numpy(k_weight).to(args.device)
    st=time.time()
    with torch.no_grad():
        for k in w_avg.keys():
            # k_st=time.time()
            torch.cuda.empty_cache()
            dim_ = [len(w)]
            for dim_shape in w[0][k].shape:
                dim_.append(dim_shape)
            tmp = torch.zeros(dim_)
            for i in range(len(w)):
                tmp[i] = w[i][k]  # get the weight of k-layer which in each client
            # tmp=tmp.cuda()
            med = tmp.median(dim=0).values
            tmp = tmp - med
            # new_tmp = []
            # for i in range(len(tmp)):# cal each client weights - median
            #     new_tmp.append(tmp[i]-med)
            # new_tmp = np.array(new_tmp)
            good_vals = torch.argsort(abs(tmp), dim=0)[:number_to_consider]
            good_vals = torch.take_along_dim(tmp, good_vals, dim=0)
            # del tmp
            w_avg[k] = (good_vals.mean() + med).cpu()
            # print(k,time.time() - k_st)
    print(time.time()-st)
    return w_avg


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedAvg_noise_loss(w, data_quality, w_glob):
    dc = list()
    wc = list()
    for i in range(len(data_quality)):
        dc.append(1 / data_quality[i])
    sum_dc = np.sum(dc)
    for i in range(len(data_quality)):
        dc[i] = dc[i] / sum_dc

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * dc[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * dc[i]
        # w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedAvg_noise_loss_and_weight(w, data_quality, w_glob):
    dc = list()
    wc = list()
    for i in range(len(data_quality)):
        dc.append(1 / data_quality[i])
    sum_dc = np.sum(dc)
    for i in range(len(data_quality)):
        dc[i] = dc[i] / sum_dc

    for i in range(len(data_quality)):
        wc.append(0)
        for k in w_glob.keys():
            wc[i] = wc[i] + (distance(w_glob[k], w[i][k]))

    for i in range(len(data_quality)):
        wc[i] = (1 / wc[i]).cpu()

    sum_wc = np.sum(wc)
    for i in range(len(data_quality)):
        wc[i] = wc[i] / sum_wc

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * (dc[0] + wc[0]) / 2
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * (dc[i] + wc[i]) / 2
        # w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedAvg_noise_weight(w, data_quality, w_glob):
    dc = list()
    wc = list()
    for i in range(len(data_quality)):
        dc.append(1 / data_quality[i])
    sum_dc = np.sum(dc)
    for i in range(len(data_quality)):
        dc[i] = dc[i] / sum_dc

    for i in range(len(data_quality)):
        wc.append(0)
        for k in w_glob.keys():
            # c[i] = wc[i]+(distance(w_glob[k], w[i][k]))
            wc[i] = wc[i] + distance(FedAvg(w)[k], w[i][k]).cpu()

    # print("dc  loss      ",data_quality)
    # print("wc   dis      ",wc)
    for i in range(len(data_quality)):
        wc[i] = 1 / wc[i]

    sum_wc = np.sum(wc)
    for i in range(len(data_quality)):
        wc[i] = wc[i] / sum_wc

    wc_list = list()
    for i in range(len(data_quality)):
        wc_list.append(wc[i].data)

    # print("dc            ",dc)
    # print("wc            ",wc_list)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * wc[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * wc[i]
        # w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedAvg_noise_layer_weight(arg, w, w_glob, epoch,client_datalen,loss_locals):
    wc = torch.zeros((len(w_glob.keys()), len(w)))
    new_glob_w = FedAvg(w)

    model_paratmeter_key_list = list(w_glob.keys())
    for client in range(len(w)):
        for k in w_glob.keys():
            # c[i] = wc[i]+(distance(w_glob[k], w[i][k]))
            wc[model_paratmeter_key_list.index(k), client] = distance(w_glob[k], w[client][k]).cpu() + 1

    # print("dc  loss      ",data_quality)
    weight_dis = copy.deepcopy(wc)
    # print("wc   dis      ",weight_dis)
    # for i in range(len(w)):
    #     wc[i] = 1 / wc[i]
    # abs_v=wc.max(dim=1).values -wc.min(dim=1).values
    # abs_v=abs_v.reshape(-1,1)
    # wc=torch.div(wc-wc.min(dim=1).values.reshape(-1,1),abs_v) +1

    all_w = wc.sum(dim=0)
    print("Total_unscale_divergence ", all_w)
    client_datasize = torch.from_numpy(np.array(client_datalen))
    loss_locals=torch.from_numpy(np.array(loss_locals))
    if arg.only_ce:
        all_w = loss_locals / client_datasize
    elif arg.only_modeldist:
        all_w = all_w / client_datasize
    else:
        all_w = all_w * loss_locals / client_datasize
    noise_client = all_w * (abs(all_w - all_w.mean()) > arg.std_num * all_w.std())
    noise_client = noise_client.nonzero().view(-1).tolist()
    if len(noise_client) > 0:
        if arg.wo_plently:
            current = 1
        elif epoch > arg.pl_epoch:
            # current = np.clip((epoch-arg.pl_epoch )/ 3, 1.1, 1.5)
            current = arg.penalty
        else:
            current = np.clip(epoch / arg.start_penalty, arg.start_penalty, arg.penalty)
            # current = np.clip(epoch / arg.start_penalty, 1, arg.penalty)
        # current=1
        for j in noise_client:
            wc[:, j] = current * wc[:, j]

    if arg.layer_agg:
        # layer aggeration
        wc = 1 / wc
        wc = wc / torch.sum(wc, dim=1, keepdim=True)
        # wc.softmax()
        print("Total_divergence ", all_w)
        print("layer wc ", wc)
        print("DataNum", client_datalen)
        # 正态分布差
    else:
        # total weight divergence
        all_w = all_w / all_w.sum()
        all_w = 1 / all_w
        all_w = all_w / all_w.sum()
        print("Total_divergence ", all_w)
        print("Total_wc ", all_w)

    # sum_wc = np.sum(wc)
    # for i in range(len(data_quality)):
    #     wc[i] = wc[i] / sum_wc
    #
    # wc_list = list()
    # for i in range(len(data_quality)):
    #     wc_list.append(wc[i].data)

    # print("dc            ",dc)
    #

    w_avg = copy.deepcopy(w[0])
    if arg.reweight_classifier:
        for layer_id in range(len(model_paratmeter_key_list)):
            if arg.layer_agg:
                if 'classifier' in model_paratmeter_key_list[layer_id]:
                    w_avg[model_paratmeter_key_list[layer_id]] = w_avg[model_paratmeter_key_list[layer_id]] * wc[
                        layer_id, 0]  # layer-wise agg
                else:
                    w_avg[model_paratmeter_key_list[layer_id]] = w_avg[model_paratmeter_key_list[layer_id]] * (
                                1 / len(w))
            else:
                w_avg[model_paratmeter_key_list[layer_id]] = w_avg[model_paratmeter_key_list[layer_id]] * all_w[
                    0]  # 总的divergence
        for layer_id in range(len(model_paratmeter_key_list)):
            for i in range(1, len(w)):
                # w_avg[k] += w[i][k] * wc[i]
                if arg.layer_agg:
                    if 'classifier' in model_paratmeter_key_list[layer_id]:
                        w_avg[model_paratmeter_key_list[layer_id]] += w[i][model_paratmeter_key_list[layer_id]] * wc[
                            layer_id, i]
                    else:
                        w_avg[model_paratmeter_key_list[layer_id]] += w[i][model_paratmeter_key_list[layer_id]] * (
                                    1 / len(w))
                else:
                    w_avg[model_paratmeter_key_list[layer_id]] += w[i][model_paratmeter_key_list[layer_id]] * all_w[
                        i]  ## 总的divergence
    else:
        for layer_id in range(len(model_paratmeter_key_list)):
            if arg.layer_agg:
                w_avg[model_paratmeter_key_list[layer_id]] = w_avg[model_paratmeter_key_list[layer_id]] * wc[
                    layer_id, 0]  # layer-wise agg
            else:
                w_avg[model_paratmeter_key_list[layer_id]] = w_avg[model_paratmeter_key_list[layer_id]] * all_w[
                    0]  # 总的divergence
        for layer_id in range(len(model_paratmeter_key_list)):
            for i in range(1, len(w)):
                # w_avg[k] += w[i][k] * wc[i]
                if arg.layer_agg:
                    w_avg[model_paratmeter_key_list[layer_id]] += w[i][model_paratmeter_key_list[layer_id]] * wc[
                        layer_id, i]
                else:
                    w_avg[model_paratmeter_key_list[layer_id]] += w[i][model_paratmeter_key_list[layer_id]] * all_w[
                        i]  ## 总的divergence
        # w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg, weight_dis, wc, noise_client


def FedAvg_withDetectionNoise(w, w_glob):
    wc = torch.zeros((len(w_glob.keys()), len(w)))
    # new_glob_w = FedAvg(w)

    model_paratmeter_key_list = list(w_glob.keys())
    for client in range(len(w)):
        for k in w_glob.keys():
            # c[i] = wc[i]+(distance(w_glob[k], w[i][k]))
            wc[model_paratmeter_key_list.index(k), client] = distance(w_glob[k], w[client][k]).cpu() + 1

    all_w = wc.sum(dim=0)
    noise_client = all_w * ((all_w - all_w.mean()) > 1 * all_w.std())
    noise_client = noise_client.nonzero().view(-1).tolist()

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg, noise_client


def distance(w_global, w_local):
    # print(torch.norm(w_global-w_local))
    #
    # print(w_global.shape, w_local.shape)
    # 
    # for i
    # return F.pairwise_distance(w_global, w_local)
    w_d = w_global - w_local
    w_d = w_d.float()
    return torch.norm(w_d, 2)  # L2
    # return torch.cosine_similarity(w_global[0],w_local[0]) #L2

def FedBN(server_model,client_weight):
    for key in server_model.state_dict().keys():
        if 'bn' not in key:
            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
            for client_idx in range(client_num):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
            server_model.state_dict()[key].data.copy_(temp)
            for client_idx in range(client_num):
                models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])


def agg_feddyn(args,global_model,uploaded_models,server_state):
    model_delta = copy.deepcopy(uploaded_models[0])
    for param in model_delta.parameters():
        param.data = torch.zeros_like(param.data)

    for client_model in uploaded_models:
        for server_param, client_param, delta_param in zip(global_model.parameters(), client_model.parameters(),
                                                           model_delta.parameters()):
            delta_param.data += (client_param - server_param) / args.num_users

    for state_param, delta_param in zip(server_state.parameters(), model_delta.parameters()):
        state_param.data -= args.feddyn_alpha * delta_param

    new_global_model = copy.deepcopy(uploaded_models[0])
    for param in new_global_model.parameters():
        param.data = torch.zeros_like(param.data)

    for client_model in uploaded_models:
        for server_param, client_param in zip(new_global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() / args.num_users
        # self.add_parameters(client_model)


    for server_param, state_param in zip(new_global_model.parameters(), server_state.parameters()):
        server_param.data -= (1 / args.feddyn_alpha) * state_param
    return server_state,new_global_model


