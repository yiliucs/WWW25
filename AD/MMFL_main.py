from __future__ import print_function

import copy
import os
import sys
import argparse
import time
import math
import torch.nn as nn
# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
import torch.nn.functional as F
from model import MySingleModel, My3Model,My2Model,Generator
import data_pre as data
import numpy as np
from sklearn.model_selection import train_test_split
# from communication import COMM
import random
from sampling_noniid import *
import itertools
from kmeans_pytorch import kmeans, kmeans_predict
# torch.backends.cudnn.enabled=False

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--distill', type=bool, default=True,
                        help='proportion_to_zero')
    parser.add_argument('--dataset', type=str, default='AD',
                        help='select dataset')
    parser.add_argument('--random_miss', type=bool, default=True,
                        help='proportion_to_zero')
    parser.add_argument('--random_pad', type=bool, default=False,
                        help='proportion_to_zero')
    parser.add_argument('--proportion_to_zero', type=float, default=0.8,
                        help='proportion_to_zero')
    parser.add_argument('--miss_zero', type=bool, default=False,
                        help='miss_zero')
    parser.add_argument('--cuda_device', type=int, default=1,
                        help='cuda')
    parser.add_argument('--usr_id', type=int, default=0,
                        help='user id')
    parser.add_argument('--local_modality', type=str, default='all',
                        choices=['audio', 'depth', 'radar', 'all'], help='local_modality')
    parser.add_argument('--server_address', type=str, default='192.168.83.1',
                        help='server_address')
    parser.add_argument('--fl_epoch', type=int, default=10,
                        help='communication to server after the epoch of local training')

    parser.add_argument('--print_freq', type=int, default=5,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate def:1e-40.00005')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.02,
                        help='decay rate for learning rate')

    # model dataset
    parser.add_argument('--model', type=str, default='MyMMmodel')
    parser.add_argument('--dataset', type=str, default='AD',
                        choices=['MHAD', 'FLASH', 'AD'], help='dataset')
    parser.add_argument('--num_class', type=int, default=11,
                        help='num_class')
    parser.add_argument('--num_of_train', type=int, default=200,
                        help='num_of_train')
    parser.add_argument('--num_of_test', type=int, default=50,
                        help='num_of_test')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    # set the path according to the environment
    opt.result_path = './save_unifl_results/node_{}/{}_results/'.format(opt.usr_id, opt.local_modality)

    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)

    return opt
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb

def set_loader_NON_iid(opt,usr,modality):
    # load labeled train and test data
    print("train labeled data:")

    total_dataset = data.Multimodal_dataset_train(usr)
    # idx=mixture_distribution_split_noniid(total_dataset,11,10,-1,0.1)


    train_sample=np.loadtxt('sample_index/idx_{}.txt'.format(usr))
    if len(train_sample)>=200:
        sam_size=200
        train_sample = np.random.choice(train_sample, size=sam_size, replace=False)
    else:
        pass

    print(len(train_sample))
    train_sample = list(map(int, train_sample))  # np.array(train_sample_index).

    train_sample = torch.utils.data.SubsetRandomSampler(train_sample, generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(
        total_dataset, batch_size=opt.batch_size, sampler=train_sample,
        num_workers=opt.num_workers,
        pin_memory=True,
        # shuffle=True,
        drop_last=True)
    # val_loader = torch.utils.data.DataLoader(
    #     total_dataset, batch_size=opt.batch_size, sampler=test_sample,
    #     num_workers=opt.num_workers,
    #     pin_memory=True,
    #     # shuffle=True,
    #     drop_last=True)

    return train_loader





def set_loader(opt,usr,modality):
    # load labeled train and test data
    print("train labeled data:")

    if modality == "all":
        total_dataset = data.Multimodal_dataset(usr)
        print('total dataset')


    else:
        total_dataset = data.Unimodal_dataset(usr, modality)

    sample_index_path = "sample_index/node_{}/".format(usr)
    all_train_sample_index = np.loadtxt(sample_index_path + "train_sample_index.txt")
    all_test_sample_index = np.loadtxt(sample_index_path + "test_sample_index.txt")

    if len(all_train_sample_index) < opt.num_of_train:
        opt.num_of_train = len(all_train_sample_index)
    if len(all_test_sample_index) < opt.num_of_test:
        opt.num_of_test = len(all_test_sample_index)

    train_sample_index = all_train_sample_index[0:opt.num_of_train]
    test_sample_index = all_test_sample_index[0:opt.num_of_test]
    # test_list.append(test_sample_index)

    ##set1, only add incomplete depth data on multimodal nodes
    if usr >= 6:
        if modality == "depth":  # opt.local_modality == "audio" or opt.local_modality == "depth":
            all_incomplete_sample_index = np.loadtxt(sample_index_path + "incomplete_sample_index.txt")
            train_sample_index = np.append(train_sample_index, all_incomplete_sample_index)
    print(len(train_sample_index))
    print(len(test_sample_index))
    print(train_sample_index[0:50])
    print(test_sample_index)

    train_sample_index = list(map(int, train_sample_index))  # np.array(train_sample_index).
    test_sample_index = list(map(int, test_sample_index))

    train_sample = torch.utils.data.SubsetRandomSampler(train_sample_index, generator=torch.Generator().manual_seed(42))
    test_sample = torch.utils.data.SubsetRandomSampler(test_sample_index, generator=torch.Generator().manual_seed(42))
    # print(train_sample)
    # print(test_sample)
    train_loader = torch.utils.data.DataLoader(
        total_dataset, batch_size=opt.batch_size, sampler=train_sample,
        num_workers=opt.num_workers,
        pin_memory=True,
        # shuffle=True,
        drop_last=True)
    print(len(train_sample_index))
    print(len(train_loader))
    val_loader = torch.utils.data.DataLoader(
        total_dataset, batch_size=opt.batch_size, sampler=test_sample,
        num_workers=opt.num_workers,
        pin_memory=True,
        # shuffle=True,
        drop_last=True)


    return train_loader, val_loader

def set_test(opt):
    test_dataset = data.Multimodal_dataset_test()
    random_list = np.random.choice(len(test_dataset), size=200, replace=False)
    train_sample = list(map(int, random_list))  # np.array(train_sample_index).

    test_sample = torch.utils.data.SubsetRandomSampler(train_sample, generator=torch.Generator().manual_seed(42))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,sampler=test_sample,
        num_workers=opt.num_workers,
        pin_memory=True,
        # shuffle=True,
        drop_last=True)
    return test_loader

def set_gen_model(opt,model):


    modality=model.miss_modal
    gen_model=Generator(modality,opt)
    print(gen_model)
    print('gen:',get_model_size(gen_model))

    if torch.cuda.is_available():
        gen_model = gen_model.cuda(opt.cuda_device)
        cudnn.benchmark = True

    return gen_model


def set_model(opt,modality):
    miss_num=np.random.choice(3)
    # miss_modal=sorted(np.random.choice([1,2,3],size=miss_num,replace=False))
    miss_modal=[3]

    miss_rate= np.random.choice([0.8])
    # miss_rate= 1.0



    if modality == "all":
        model = My3Model(opt.num_class,miss_modal,miss_rate)
        print('model:',get_model_size(model))

    # elif modality == 'AD' or 'AR' or 'DR':
    #     model = My2Model(num_classes=opt.num_class, modality=modality)
    else:
        model = MySingleModel(num_classes=opt.num_class, modality=modality)

    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda(opt.cuda_device)
        criterion = criterion.cuda(opt.cuda_device)
        cudnn.benchmark = True

    return model, criterion




def train_multi(train_loader, model, criterion, optimizer, epoch, opt,gen):
    """one epoch training"""
    model.train()
    miss=model.miss_modal
    print('-----------MISS MODAL-----------:',miss)
    ensemble_loss = nn.KLDivLoss(reduction="batchmean")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (input_data1, input_data2, input_data3, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # print(gen)
        if torch.cuda.is_available():
            input_data1 = input_data1.cuda(opt.cuda_device)
            input_data2 = input_data2.cuda(opt.cuda_device)
            input_data3 = input_data3.cuda(opt.cuda_device)
            labels = labels.cuda(opt.cuda_device)
        bsz = input_data1.shape[0]
        if opt.miss_zero:
            # print(model.miss_modal)
            # if len(model.miss_modal)!=0:
            #     print('input1==0')
            num_elements_to_zero = int(bsz *opt.proportion_to_zero)
            zero_indices = sorted(random.sample(range(bsz), num_elements_to_zero))
            input_data2[zero_indices]=torch.zeros_like(input_data2[zero_indices]).cuda(opt.cuda_device)
        elif opt.random_miss:
            num_elements_to_zero = int(bsz * model.miss_rate)
            zero_indices = sorted(random.sample(range(bsz), num_elements_to_zero))

            if len(model.miss_modal)>1:
                # print(model.miss_modal)
                if len(model.miss_modal)==2:
                    if model.miss_modal== [1,2]:
                        # print([1,2])
                        input_data1[zero_indices] = torch.zeros_like(input_data1[zero_indices].cuda(opt.cuda_device))
                        input_data2[zero_indices] = torch.zeros_like(input_data2[zero_indices].cuda(opt.cuda_device))

                    elif model.miss_modal == [1,3]:
                        # print([1, 3])
                        input_data1[zero_indices] = torch.zeros_like(input_data1[zero_indices].cuda(opt.cuda_device))
                        input_data3[zero_indices] = torch.zeros_like(input_data3[zero_indices].cuda(opt.cuda_device))
                    else:
                        # print([2, 3])
                        input_data2[zero_indices] = torch.zeros_like(input_data2[zero_indices].cuda(opt.cuda_device))
                        input_data3[zero_indices] = torch.zeros_like(input_data3[zero_indices].cuda(opt.cuda_device))

                else:
                    # print([1,2,3])
                    input_data1[zero_indices] = torch.zeros_like(input_data1[zero_indices].cuda(opt.cuda_device))
                    input_data2[zero_indices] = torch.zeros_like(input_data2[zero_indices].cuda(opt.cuda_device))
                    input_data3[zero_indices] = torch.zeros_like(input_data3[zero_indices].cuda(opt.cuda_device))


                #
                # variable_name1 = 'input_data' + str(model.miss_modal[0])
                # eval(variable_name1)[zero_indices] = torch.zeros_like(eval(variable_name1)[zero_indices]).cuda(opt.cuda_device)
                # variable_name2 = 'input_data' + str(model.miss_modal[1])
                # eval(variable_name2)[zero_indices] = torch.zeros_like(eval(variable_name2)[zero_indices]).cuda(opt.cuda_device)



            elif len(model.miss_modal)==1:
                if model.miss_modal[0]==1:
                    # print([1])
                    input_data1[zero_indices] = torch.zeros_like(input_data1[zero_indices].cuda(opt.cuda_device))
                elif model.miss_modal[0]==2:
                    # print([2])
                    input_data2[zero_indices] = torch.zeros_like(input_data2[zero_indices].cuda(opt.cuda_device))
                else:
                    # print([3])
                    input_data3[zero_indices] = torch.zeros_like(input_data3[zero_indices].cuda(opt.cuda_device))
                # variable_name1 = 'input_data' + str(model.miss_modal[0])
                # eval(variable_name1)[zero_indices] = torch.zeros_like(
                #     eval(variable_name1)[zero_indices]).cuda(opt.cuda_device)
            else:
                pass

        elif opt.random_pad:
            num_elements_to_zero = int(bsz * model.miss_rate)
            zero_indices = sorted(random.sample(range(bsz), num_elements_to_zero))

            if len(model.miss_modal) > 1:
                # print(model.miss_modal)
                if len(model.miss_modal) == 2:
                    if model.miss_modal == [1, 2]:
                        # print([1,2])
                        input_data1[zero_indices] = torch.rand_like(
                            input_data1[zero_indices].cuda(opt.cuda_device))
                        input_data2[zero_indices] = torch.rand_like(
                            input_data2[zero_indices].cuda(opt.cuda_device))

                    elif model.miss_modal == [1, 3]:
                        # print([1, 3])
                        input_data1[zero_indices] = torch.rand_like(
                            input_data1[zero_indices].cuda(opt.cuda_device))
                        input_data3[zero_indices] = torch.rand_like(
                            input_data3[zero_indices].cuda(opt.cuda_device))
                    else:
                        # print([2, 3])
                        input_data2[zero_indices] = torch.rand_like(
                            input_data2[zero_indices].cuda(opt.cuda_device))
                        input_data3[zero_indices] = torch.rand_like(
                            input_data3[zero_indices].cuda(opt.cuda_device))

                else:
                    # print([1,2,3])
                    input_data1[zero_indices] = torch.rand_like(input_data1[zero_indices].cuda(opt.cuda_device))
                    input_data2[zero_indices] = torch.rand_like(input_data2[zero_indices].cuda(opt.cuda_device))
                    input_data3[zero_indices] = torch.rand_like(input_data3[zero_indices].cuda(opt.cuda_device))

        else:
            pass



        # compute loss
        output,_ = model(input_data1, input_data2, input_data3)
        loss = criterion(output, labels)
        if opt.distill:
            gen_out = gen(labels)
            out_gen,_=model(gen_out['output'],None,None,latent=True)
            loss_kl=ensemble_loss(F.log_softmax(output/0.5, dim=1),F.softmax(out_gen/0.5, dim=1))
            sampled_y = np.random.choice(11,size=bsz)
            sampled_y = torch.tensor(sampled_y).cuda(opt.cuda_device)
            gen_out = gen(sampled_y)
            out_gen,_ = model(gen_out['output'],None, None, latent=True)
            loss_t = torch.mean(gen.crossentropy_loss(F.log_softmax(out_gen, dim=1),sampled_y))
            # print('KL',loss_kl)
            # print('CE',loss_t)
            loss_gen=1/2*loss_kl+1/2*loss_t
            loss=1/2*loss_gen+1/2*loss




        acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        # update metric
        losses.update(loss.item(), bsz)
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        # loss_gen.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'MISS_MODAL {miss})\t'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1,miss=miss))
        sys.stdout.flush()

    return losses.avg




def validate_multi(val_loader, model, criterion, opt):
    """validation"""
    model.cuda(opt.cuda_device)
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    confusion = np.zeros((opt.num_class, opt.num_class))

    with torch.no_grad():
        end = time.time()
        for idx, (input_data1, input_data2, input_data3, labels) in enumerate(val_loader):

            if torch.cuda.is_available():
                input_data1 = input_data1.float().cuda(opt.cuda_device)
                input_data2 = input_data2.float().cuda(opt.cuda_device)
                input_data3 = input_data3.float().cuda(opt.cuda_device)
                labels = labels.cuda(opt.cuda_device)
            bsz = labels.shape[0]

            # forward
            output,_ = model(input_data1, input_data2, input_data3)
            loss = criterion(output, labels)

            # update metric
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            # print(acc1)
            # print(acc5)
            losses.update(loss.item(), bsz)
            top1.update(acc1[0], bsz)

            # calculate and store confusion matrix
            rows = labels.cpu().numpy()
            cols = output.max(1)[1].cpu().numpy()
            for label_index in range(labels.shape[0]):
                confusion[rows[label_index], cols[label_index]] += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if idx % opt.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
            #         idx, len(val_loader), batch_time=batch_time,
            #         loss=losses, top1=top1))

    return losses.avg, top1.avg, confusion



def model_agg(opt,model_dict,val_loader,criterion,last_glob_acc):
    def get_model_name(model_dict):
        model_name = []
        para_num= {'e1':[],'e2':[],'e3':[],"c1":[],"c2":[],"c3":[]}
        a_num,d_num,r_num,c_num=0,0,0,0
        for model_id, model in model_dict.items():
            if model.modality=='all':
                model_name.append(3)
                para_num['e1'].append(0)
                para_num['e2'].append(0)
                para_num['e3'].append(0)
                para_num['c1'].append(0)
                para_num['c2'].append(0)
                para_num['c3'].append(0)
            elif model.modality=='audio' or 'depth' or 'radar':
                model_name.append(1)
                if model.modality=='audio':
                    para_num['e1'].append(0)
                    para_num['c1'].append(0)
                elif model.modality=='depth':
                    para_num['e2'].append(0)
                    para_num['c2'].append(0)
                else:
                    para_num['e3'].append(0)
                    para_num['c3'].append(0)
            else:
                model_name.append(2)

        return model_name,para_num

    num = 0
    local_contri = []
    global_contri = []
    select_list = []
    acc = 0
    gmodel = []
    gdict = []

    for model_id, model in copy.deepcopy(model_dict).items():
        for model_id, _ in model_dict.items():
            loss, _acc, confusion = validate_multi(val_loader[model_id], model, criterion, opt)
            acc += _acc.item()
        acc /= len(val_loader)
        model.cpu()
        local_contri.append(acc)
    print(local_contri)

    for i in range(10):
        g = dict(list(model_dict.items())[:(i + 1)])
        gdict.append(g)

    for i in range(10):
        # print(i)
        # glob=MyMMModel(opt.num_class, None, None)
        # for param in m.parameters():
        #     param.data.zero_()
        glob = gdict[i][0].cpu()
        acc2 = 0
        for id, (model_i, model) in enumerate(gdict[i].items()):
            model.cpu()
            if id == 0:
                pass
            else:
                for idx, ((name1, param), (name2, param_model)) in enumerate(
                        zip(glob.encoder.encoder_1.named_parameters(),
                            model.encoder.encoder_1.named_parameters())):
                    param.data = (param.data + param_model.data)

                for idx, ((name1, param), (name2, param_model)) in enumerate(
                        zip(glob.encoder.encoder_2.named_parameters(),
                            model.encoder.encoder_2.named_parameters())):
                    # if idx not in [2,3,6,7,10,11]:
                    param.data = (param.data + param_model.data)
                for idx, ((name1, param), (name2, param_model)) in enumerate(
                        zip(glob.encoder.encoder_3.named_parameters(),
                            model.encoder.encoder_3.named_parameters())):
                    # if idx not in [2,3,6,7,10,11]:
                    param.data = (param.data + param_model.data)
                for idx, ((name1, param), (name2, param_model)) in enumerate(
                        zip(glob.classifier.named_parameters(), model.classifier.named_parameters())):
                    # if idx not in [2,3]:
                    param.data = (param.data + param_model.data)

        for idx, (name, param) in enumerate(glob.encoder.encoder_1.named_parameters()):
            param.data = param.data / (i + 1)
        for idx, (name, param) in enumerate(glob.encoder.encoder_2.named_parameters()):
            param.data = param.data / (i + 1)
        for idx, (name, param) in enumerate(glob.classifier.named_parameters()):
            if idx == 0:
                param.data[:, :256] = param.data[:, :256] / (i + 1)
                param.data[:, 256:] = param.data[:, 256:] / (i + 1)
            else:
                param.data = param.data / (i + 1)

        for model_id, _ in model_dict.items():
            _, accg, _ = validate_multi(val_loader[model_id], glob, criterion, opt)
            acc2 += accg.item()
        acc2 /= len(val_loader)
        glob.cpu()
        print(acc2)
        print(last_glob_acc)
        global_contri.append(acc2 - last_glob_acc)
        del glob


    glob_model = My3Model(opt.num_class,None,None)
    for param in glob_model.parameters():
        param.data.zero_()

    index = []

    for idx, (l, g) in enumerate(zip(local_contri, global_contri)):
        if l + g > (last_glob_acc):
            index.append(idx)
    print(index)
    new_model_dict = dict()
    for idx, (key, val) in enumerate(model_dict.items()):
        if key in index:
            new_model_dict[idx] = val
    if len(new_model_dict) <= 3:
        new_model_dict = model_dict
    else:
        pass
    model_name, param_num = get_model_name(new_model_dict)
    print(param_num)
        # print(param)
    # print(model_name)
    num=0
    for model_id, model in new_model_dict.items():
        model.cpu()
        for param,param_model in zip(glob_model.encoder.encoder_1.parameters(),model.encoder.encoder_1.parameters()):
            param.data = (param.data+param_model.data)
        for param,param_model in zip(glob_model.encoder.encoder_2.parameters(),model.encoder.encoder_2.parameters()):
            param.data = (param.data + param_model.data)

        for param,param_model in zip(glob_model.encoder.encoder_3.parameters(),model.encoder.encoder_3.parameters()):
            param.data = (param.data + param_model.data)

        for param,param_model in zip(glob_model.classifier.parameters(),model.classifier.parameters()):
            param.data = (param.data + param_model.data)


                # para_c[:,496:] += model.classifier.weight
    print(param_num)
    for param in glob_model.encoder.encoder_1.parameters():
        param.data = param.data/len(param_num['e1'])
    for param in glob_model.encoder.encoder_2.parameters():
        param.data = param.data/len(param_num['e2'])
    for param in glob_model.encoder.encoder_3.parameters():
        param.data = param.data/len(param_num['e3'])
    for idx,param in enumerate(glob_model.classifier.parameters()):
        if idx==0:
            param.data[:, :240] = param.data[:, :240]/len(param_num['c1'])
            param.data[:,240:496] = param.data[:,240:496]/len(param_num['c2'])
            param.data[:, 496:] = param.data[:, 496:]/len(param_num['c3'])
        else:
            param.data=param.data/len(new_model_dict)


    return glob_model

def model_agg_org(opt,model_dict,val_loader,criterion,last_glob_acc):
    def get_model_name(model_dict):
        model_name = []
        para_num= {'e1':[],'e2':[],'e3':[],"c1":[],"c2":[],"c3":[]}
        a_num,d_num,r_num,c_num=0,0,0,0
        for model_id, model in model_dict.items():
            if model.modality=='all':
                model_name.append(3)
                para_num['e1'].append(0)
                para_num['e2'].append(0)
                para_num['e3'].append(0)
                para_num['c1'].append(0)
                para_num['c2'].append(0)
                para_num['c3'].append(0)
            elif model.modality=='audio' or 'depth' or 'radar':
                model_name.append(1)
                if model.modality=='audio':
                    para_num['e1'].append(0)
                    para_num['c1'].append(0)
                elif model.modality=='depth':
                    para_num['e2'].append(0)
                    para_num['c2'].append(0)
                else:
                    para_num['e3'].append(0)
                    para_num['c3'].append(0)
            else:
                model_name.append(2)

        return model_name,para_num

    glob_model = My3Model(opt.num_class,None,None)
    for param in glob_model.parameters():
        param.data.zero_()
        # print(param)
    model_name,param_num=get_model_name(model_dict)
    # print(model_name)
    num=0
    for model_id, model in model_dict.items():
        model.cpu()
        if model_name[model_id] == 3:
            for param,param_model in zip(glob_model.encoder.encoder_1.parameters(),model.encoder.encoder_1.parameters()):
                param.data = (param.data+param_model.data)
            for param,param_model in zip(glob_model.encoder.encoder_2.parameters(),model.encoder.encoder_2.parameters()):
                param.data = (param.data + param_model.data)

            for param,param_model in zip(glob_model.encoder.encoder_3.parameters(),model.encoder.encoder_3.parameters()):
                param.data = (param.data + param_model.data)

            for param,param_model in zip(glob_model.classifier.parameters(),model.classifier.parameters()):
                param.data = (param.data + param_model.data)


        elif model_name[model_id] == 2:
            if model.modality=='AD':
                for param, param_model in zip(glob_model.encoder.encoder_1.parameters(),
                                              model.encoder.encoder_1.parameters()):
                    param.data = (param.data+param_model.data)
                for param, param_model in zip(glob_model.encoder.encoder_2.parameters(),
                                              model.encoder.encoder_2.parameters()):
                    param.data = (param.data + param_model.data)
                for param, param_model in zip(glob_model.classifier.parameters(), model.classifier.parameters()):
                    param.data[:,:496] = param.data[:,:496] + param_model.data[:,:496]
                # para_c[:,:496] += model.classifier.weight
            if model.modality=='DR':
                for param, param_model in zip(glob_model.encoder.encoder_2.parameters(),
                                              model.encoder.encoder_2.parameters()):
                    param.data = (param.data+param_model.data)
                for param, param_model in zip(glob_model.encoder.encoder_3.parameters(),
                                              model.encoder.encoder_3.parameters()):
                    param.data = (param.data+param_model.data)
                for param, param_model in zip(glob_model.classifier.parameters(), model.classifier.parameters()):
                    param.data[:,240:] = param.data[:,240:] + param_model.data[:,240:]
                # para_c[:,240:] += model.classifier.weight
            if model.modality=='AR':
                for param, param_model in zip(glob_model.encoder.encoder_1.parameters(),
                                              model.encoder.encoder_1.parameters()):
                    param.data = (param.data+param_model.data)
                for param, param_model in zip(glob_model.encoder.encoder_3.parameters(),
                                              model.encoder.encoder_3.parameters()):
                    param.data = (param.data+param_model.data)
                for param, param_model in zip(glob_model.classifier.parameters(), model.classifier.parameters()):
                    param.data[:,:240] = param.data[:,:240] + param_model.data[:,:240]
                    param.data[:, 496:] = param.data[:, 496:] + param_model.data[:, 496:]

                # para_c[:,:240] += model.classifier.weight[:,:240]
                # para_c[:, 496:] += model.classifier.weight[:, 496:]
        else:
            if model.modality == 'audio':
                for param, param_model in zip(glob_model.encoder.encoder_1.parameters(),
                                              model.encoder.parameters()):
                    param.data = (param.data + param_model.data)
                for idx,(param, param_model) in enumerate(zip(glob_model.classifier.parameters(), model.classifier.parameters())):
                    if idx==0:
                        param.data[:, :240] = param.data[:, :240] + param_model.data
                    else:
                        param.data = (param.data + param_model.data)
            if model.modality == 'depth':
                for param, param_model in zip(glob_model.encoder.encoder_2.parameters(),
                                              model.encoder.parameters()):
                    param.data = (param.data + param_model.data)
                for idx, (param, param_model) in enumerate(
                        zip(glob_model.classifier.parameters(), model.classifier.parameters())):
                    if idx == 0:
                        param.data[:,240:496]= param.data[:,240:496] + param_model.data
                    else:
                        param.data = (param.data + param_model.data)

            if model.modality == 'radar':
                for param, param_model in zip(glob_model.encoder.encoder_3.parameters(),
                                              model.encoder.parameters()):
                    param.data = (param.data + param_model.data)
                for idx, (param, param_model) in enumerate(
                        zip(glob_model.classifier.parameters(), model.classifier.parameters())):
                    if idx == 0:
                        param.data[:, 496:] = param.data[:, 496:] + param_model.data
                    else:
                        param.data = (param.data + param_model.data)
                # para_c[:,496:] += model.classifier.weight
    print(param_num)
    for param in glob_model.encoder.encoder_1.parameters():
        param.data = param.data/len(param_num['e1'])
    for param in glob_model.encoder.encoder_2.parameters():
        param.data = param.data/len(param_num['e2'])
    for param in glob_model.encoder.encoder_3.parameters():
        param.data = param.data/len(param_num['e3'])
    for idx,param in enumerate(glob_model.classifier.parameters()):
        if idx==0:
            param.data[:, :240] = param.data[:, :240]/len(param_num['c1'])
            param.data[:,240:496] = param.data[:,240:496]/len(param_num['c2'])
            param.data[:, 496:] = param.data[:, 496:]/len(param_num['c3'])
        else:
            param.data=param.data/len(model_dict)





    return glob_model



def gen_agg(gen_dict):
    # print(gen_dict.keys())
    if isinstance(gen_dict, list):
        d={}
        for idx,i in enumerate(gen_dict):
            d[idx]=i
        gen_dict=d
    for key in gen_dict.keys():
        model_parameters = [para for para in gen_dict[key].parameters()]
        break
    mean_parameters = [torch.zeros_like(param.data) for param in model_parameters]
    for name, gen_ in gen_dict.items():
        for params, mean_params in zip(gen_.parameters(), mean_parameters):
            mean_params += params.data
    for param in mean_parameters:
        param /= len(gen_dict)

    return mean_parameters


def model_evaluation_func(val_loader, model,glob, criterion, opt):
    model.cuda(opt.cuda_device)
    model.eval()
    glob.cuda(opt.cuda_device)
    glob.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    confusion = np.zeros((opt.num_class, opt.num_class))

    with torch.no_grad():
        end = time.time()
        for idx, (input_data1, input_data2, input_data3,labels) in enumerate(val_loader):

            if torch.cuda.is_available():
                input_data1 = input_data1.float().cuda(opt.cuda_device)
                input_data2 = input_data2.float().cuda(opt.cuda_device)
                input_data3 = input_data2.float().cuda(opt.cuda_device)

                labels = labels.cuda(opt.cuda_device)
            bsz = labels.shape[0]

            # forward
            gen_out_t = model(labels)
            out_gen, _ = glob(gen_out_t['output'],None,None,latent=True)
            # output, _ = model(input_data1, input_data2)
            loss = criterion(out_gen, labels)

            # update metric
            acc1, acc5 = accuracy(out_gen, labels, topk=(1, 5))
            # print(acc1)
            # print(acc5)
            losses.update(loss.item(), bsz)
            top1.update(acc1[0], bsz)

            # calculate and store confusion matrix


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1))
    return top1.avg


def main():
    opt = parse_option()
    print(opt.miss_zero)
    print(opt.learning_rate)
    print(opt.proportion_to_zero)


    random.seed(42)
    np.random.seed(42)

    train_loader_dict=dict()
    val_loader_dict=dict()
    model_dict=dict()
    gen_dict=dict()
    criterion_dict=dict()
    optimizers_dict=dict()
    before_model_dict=dict()
    copy_model_dict=None
    usr_list=[0,1,2,3,4,5,6,7,8,9]
    modality_list=['all', 'all','all', 'all','all', 'all','all', 'all','all', 'all']
    # set_loader_NON_iid(opt, 0, 'all')
    test_list=[]
    # test_loader=set_test(opt)
    # build data loader
    for idx,usr in enumerate(usr_list):
        train_loader,val_loader= set_loader(opt,usr,modality_list[idx])

        train_loader_dict[usr]=train_loader
        val_loader_dict[usr]=val_loader
        del train_loader,val_loader

    # print(test_list)
    # build model and criterion
    for idx,usr in enumerate(usr_list):
        model, criterion = set_model(opt,modality_list[idx])
        gen=set_gen_model(opt,model)
        # print(gen)
        # print(model)
        # w_parameter_init = get_model_array(model)
        optimizer = set_optimizer(opt, model)
        gen_dict[usr]=gen
        model_dict[usr]=model
        criterion_dict[usr]=criterion
        optimizers_dict[usr]=optimizer
        del model,criterion,optimizer,gen

    # build optimizer
    # print(model_dict)


    best_acc = 0
    record_loss = np.zeros(opt.epochs)
    record_acc = np.zeros(opt.epochs)
    record_dict=dict()
    record_glob=[]
    for usr in usr_list:
        record_dict[usr]=[]
    ACC = 1 / 11 * 100
    for epoch in range(1, opt.epochs + 1):
        for idx,usr in enumerate(usr_list):
            adjust_learning_rate(opt, optimizers_dict[usr], epoch)
            print('start usr{} training-------------------------------'.format(usr))

            # train for one epoch
            time1 = time.time()
            if modality_list[idx] == "all":
                loss = train_multi(train_loader_dict[usr], model_dict[usr], criterion_dict[usr], optimizers_dict[usr], epoch, opt,gen_dict[usr])

            else:
                loss = train_single(train_loader_dict[usr], model_dict[usr], criterion_dict[usr], optimizers_dict[usr], epoch, opt)
            time2 = time.time()

            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
            record_loss[epoch - 1] = loss
        copy_model_dict=copy.deepcopy(model_dict)
        glob_model=model_agg_org(opt,model_dict,val_loader_dict, torch.nn.CrossEntropyLoss().cuda(opt.cuda_device),ACC)

        for idx,modality in enumerate(modality_list):
            before_model_dict[idx]=copy.deepcopy(model_dict[idx])
            model_dict[idx].cpu()
            if modality == "audio":
                for param,glob_param in zip(model_dict[idx].encoder.parameters(),glob_model.encoder.encoder_1.parameters()):
                    param.data = glob_param.data.clone()
                for idx,(param,glob_param) in enumerate(zip(model_dict[idx].classifier.parameters(),glob_model.classifier.parameters())):
                    if idx==0:
                        param.data = glob_param.data[:,:240].clone()
                    else:
                        param.data =glob_param.data.clone()
            elif modality == "depth":
                for param,glob_param in zip(model_dict[idx].encoder.parameters(),glob_model.encoder.encoder_2.parameters()):
                    param.data = glob_param.data.clone()
                for idx,(param,glob_param) in enumerate(zip(model_dict[idx].classifier.parameters(),glob_model.classifier.parameters())):
                    if idx==0:
                        param.data = glob_param.data[:,240:496].clone()
                    else:
                        param.data =glob_param.data.clone()
            elif modality == "radar":
                for param,glob_param in zip(model_dict[idx].encoder.parameters(),glob_model.encoder.encoder_3.parameters()):
                    param.data = glob_param.data.clone()
                for idx,(param,glob_param) in enumerate(zip(model_dict[idx].classifier.parameters(),glob_model.classifier.parameters())):
                    if idx==0:
                        param.data = glob_param.data[:,496:].clone()
                    else:
                        param.data =glob_param.data.clone()
            else:
                for param,glob_param in zip(model_dict[idx].parameters(),glob_model.parameters()):
                    param.data = glob_param.data.clone()
                # model_dict[idx]=copy.deepcopy(glob_model)
        acc_num=0
        # print(acc_num)
        if opt.distill:
            if epoch<50:

            # if epoch // 5 == 0:
                num_clusters = 4
                sys_label=torch.tensor(np.random.choice(11, size=16)).cuda(opt.cuda_device)
                g_list=[]
                gen_means_dict={}
                model_k_mean=copy.deepcopy(gen_dict[0])
                new_gen_means_dict={}
                for name, gen_ in gen_dict.items():
                    gen_out=gen_(sys_label)

                    g_list.append(gen_out['output'].detach().cpu().numpy())


                g_list=torch.tensor(g_list)
                g_list_reshape = g_list.reshape(g_list.shape[0], -1)
                cluster_ids_x, cluster_centers = kmeans(
                    X=torch.tensor(g_list_reshape), num_clusters=num_clusters, distance='euclidean', device=torch.device("cpu")
                )
                # print(cluster_ids_x)
                for i in range(num_clusters):
                    gen_means_dict[i]= []
                # print(cluster_centers)
                for idx,i in enumerate(cluster_ids_x):
                    gen_means_dict[i.item()].append(gen_dict[idx])
                # print(gen_means_dict)
                for idx,i in enumerate(range(num_clusters)):
                    model_para = gen_agg(gen_means_dict[i])
                    for new_params, old_params in zip(model_para, model_k_mean.parameters()):
                        old_params.data.copy_(new_params)
                    new_gen_means_dict[idx]=copy.deepcopy(model_k_mean)
                print(new_gen_means_dict)





                model_test = copy.deepcopy(new_gen_means_dict[0])
                all_perms = list(itertools.permutations(list(new_gen_means_dict.keys())))
                # parem_sample=random.choices(all_perms,k=10)
                # print(parem_sample)
                marginal_contributions = []
                history = {}
                for perm in all_perms:
                    perm_values = {}
                    local_models = {}

                    for client_id in perm:
                        model = copy.deepcopy(new_gen_means_dict[client_id])
                        local_models[client_id] = model

                        # get the current index eg: (A,B,C) on the 2nd iter, the index is (A,B)
                        if len(perm_values.keys()) == 0:
                            index = (client_id,)
                        else:
                            index = tuple(sorted(list(tuple(perm_values.keys()) + (client_id,))))

                        if index in history.keys():
                            current_value = history[index]
                        else:
                            current_value=0
                            model_para = gen_agg(local_models)
                            for new_params, old_params in zip(model_para, model_test.parameters()):
                                old_params.data.copy_(new_params)

                            # print(model)
                            for model_id, model in model_dict.items():
                                model_dict[model_id].cuda(opt.cuda_device)
                                _current_value = model_evaluation_func(val_loader_dict[model_id], model_test,model_dict[model_id],
                                                                       torch.nn.CrossEntropyLoss().cuda(opt.cuda_device),
                                                                       opt)
                                current_value+=_current_value.item()
                            history[index] = current_value/len(model_dict)

                        perm_values[client_id] = max(0, current_value - sum(perm_values.values()))

                    marginal_contributions.append(perm_values)

                sv = {client_id: 0 for client_id in new_gen_means_dict.keys()}

                # sum the marginal contributions
                for perm in marginal_contributions:
                    for key, value in perm.items():
                        sv[key] += value

                # compute the average marginal contribution
                sv = {key: value / len(marginal_contributions) for key, value in sv.items()}
                print(sv)
                sv=list(sv.values())
                sum_sv=sum(sv)
                sv=[k/sum_sv for k in sv]
                print(sv)
                model_parameters = [para for para in new_gen_means_dict[0].parameters()]
                mean_parameters = [torch.zeros_like(param.data) for param in model_parameters]
                for idx,(name, gen_) in enumerate(new_gen_means_dict.items()):
                    S=sv[idx]
                    for idx,(params, mean_params) in enumerate(zip(gen_.parameters(), mean_parameters)):
                        mean_params += S*params.data
                        # else:
                        #     mean_params += 0.1*params.data

                # for param in mean_parameters:
                #     param /= len(gen_dict)
                for name, gen_ in gen_dict.items():
                        for new_params, old_params in zip(mean_parameters, gen_.parameters()):
                            old_params.data.copy_(new_params)
            model_parameters = [para for para in gen_dict[0].parameters()]
            mean_parameters = [torch.zeros_like(param.data) for param in model_parameters]
            for name, gen_ in gen_dict.items():
                for params, mean_params in zip(gen_.parameters(), mean_parameters):
                    mean_params += params.data
            for param in mean_parameters:
                param /= len(gen_dict)
            for name, gen_ in gen_dict.items():
                for new_params, old_params in zip(mean_parameters, gen_.parameters()):
                    old_params.data.copy_(new_params)





        pa_list=[]
        for model_id, model in model_dict.items():
            model_dict[model_id].cuda(opt.cuda_device)
            loss, _acc, confusion = validate_multi(val_loader_dict[model_id], model_dict[model_id], torch.nn.CrossEntropyLoss().cuda(opt.cuda_device), opt)
            acc_num += _acc.item()
        ACC = acc_num / len(val_loader_dict)

        record_glob.append(acc_num / len(val_loader_dict))
        print('--------------glob-------------', acc_num / len(val_loader_dict))




    #





if __name__ == '__main__':
    main()
