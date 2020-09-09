import os
import random

import networkx as nx
import re
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from random import choice


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)


    def __next__(self):
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator.
        '''
        while True:
            for data in dataloader:
                yield data

class networkDatasetIterator(object):
    """
    训练网络迭代器
    """
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)


    def __next__(self):

        data = next(self.iterator)

        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


class networkDataset(Dataset):
    """
    Define the dataset for network.
    """
    def __init__(self,simMatrix, args):
        self.args = args
        self.simMatrix = simMatrix
        self.batchSize = self.args.batch_size
        self.len = len(simMatrix)
        # print('in init')

    def __len__(self):
        return self.len

    # override __getitem__
    def __getitem__(self, idx):
        # print('in dataset:'+str(idx))
        return idx, self.simMatrix[idx]

    @staticmethod
    def collate_fn(data,batch_size):
        '''
        combine batch
        '''
        # print('network collate fn')
        # print(data)
        if len(data) < batch_size:
            lenDiff = batch_size - len(data)
            ids = [_[0] for _ in data]
            emb = [_[1] for _ in data]
            for counter in range(lenDiff):
                i = counter % len(data)
                ids.append(ids[i])
                emb.append(emb[i])
        else:
            ids = [_[0] for _ in data]
            emb = [_[1] for _ in data]
        return ids, emb




class treeDataset(Dataset):
    """
    Define the dataset for tree.
    """

    def __init__(self, omegaEmbeddings, args):
        self.args = args
        self.omegaEmbeddings = omegaEmbeddings
        self.batchSize = self.args.batch_size
        self.len = len(omegaEmbeddings)

    def __len__(self):
        return self.len

    # Override __getitem__
    def __getitem__(self, idx):
        return idx, self.omegaEmbeddings[idx]

    @staticmethod
    def collate_fn(data, batch_size):
        '''
        Combine batch
        '''
        # print('in collate start')
        # print(data)
        # print('in collate over')
        # if len(data) < batch_size:
        #     lenDiff = batch_size - len(data)
        #     ids = [_[0] for _ in data]
        #     emb = [_[1] for _ in data]
        #     for counter in range(lenDiff):
        #         i = counter % len(data)
        #         ids.append(ids[i])
        #         emb.append(emb[i])
        # else:
        #     ids = [_[0] for _ in data]
        #     emb = [_[1] for _ in data]
        # return ids, emb
        ids = [_[0] for _ in data]
        emb = [_[1] for _ in data]

        ids = torch.tensor(ids)
        emb = torch.tensor(emb)

        return ids, emb