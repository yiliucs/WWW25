import numpy as np
from numpy.random._examples.cffi.extending import rng
import torch
# from setting import args
from torch.distributions.dirichlet import Dirichlet


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    # label_distribution = Dirichlet(torch.full((n_clients,), alpha)).sample((n_classes,))
    label_distribution = Dirichlet(torch.full((n_clients,), alpha).float()).sample()
    # print(label_distribution)
    # 1. Get the index of each label
    class_idcs = []
    for value in range(7):
        indices = torch.nonzero(torch.eq(train_labels, value)).squeeze()
        class_idcs.append(indices)
    # print(class_idcs)
    # 2. According to the distribution, the label is assigned to each client
    client_idcs = [[] for _ in range((n_clients))]
    # print(len(class_idcs[0]))
    for c in class_idcs:
        total_size = len(c)
        # print(total_size)
        splits = (label_distribution * total_size).int()
        # print(c)
        # print(fracs)
        # print(splits)
        # print(splits[:-1])
        splits[-1] = total_size - splits[:-1].sum()
        # print(splits)
        idcs = torch.split(c, splits.tolist())
        for i, idx in enumerate(idcs):
            client_idcs[i] += [idcs[i]]

    client_idcs = [torch.cat(idcs) for idcs in client_idcs]
    return client_idcs

# torch.manual_seed(42)
label=torch.tensor(torch.load('DATA/label_train.pkl')).float()
client_idcs=dirichlet_split_noniid(label,100,100)
data=torch.load('DATA/Inerframes_train.pkl')
print(client_idcs)
print(data[client_idcs[0]])
torch.save(client_idcs,'./spilt_noniid.pkl')
