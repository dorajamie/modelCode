import os
import random

import networkx as nx
import re
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from random import choice


class DatasetH(Dataset):
    def __init__(self, leaves, tree,graph, args, total_level):

        self.len = len(leaves)
        self.all_leaves = leaves
        self.negative_sample_size = args.negative_sample_size
        self.hidden_dim = args.hidden_dim
        self.tree = tree
        self.graph = graph
        self.total_level = total_level


    def __len__(self):
        return self.len

    # override __getitem__
    def __getitem__(self, idx):
        # pick positive sample
        positive_one = list(self.all_leaves)[idx]
        positive_parent_idx = self.tree[idx].path[-2]
        positive_bros = self.tree[positive_parent_idx].direct_children

        if(len(positive_bros) == 1):        # single node
            positive_another_idx = idx
        else:                               # multiple nodes
            positive_another_idx = choice(list(positive_bros - {idx}))

        positive_sample_bros = [positive_one, positive_another_idx]

        negative_choices_pool = self.all_leaves - positive_bros

        negative_sample_list = []
        negative_sample_size = 0
        # pick negative samples
        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.choice(list(negative_choices_pool), size=self.negative_sample_size * 2,
                                               replace=False)
            negative_sample_list.extend(negative_sample)
            negative_sample_size += len(negative_sample)
        negative_sample_list = negative_sample_list[:self.negative_sample_size]

        negative_sample_bros_list = []
        negative_sample_bros_list_path = []
        for each in negative_sample_list:
            negative_sample_bros_list.append([idx, each])
            negative_sample_bros_list_path.append(self.tree[each].path)

        # positive is in the format of [one,another]
        # negative is in the format of [bro,bro,bro,bro.......]
        positive_sample_bros = torch.FloatTensor(positive_sample_bros)
        negative_sample_bros_list = torch.FloatTensor(negative_sample_bros_list)
        positive_sample_bros_path = [self.tree[idx].path,self.tree[positive_another_idx].path]

        return torch.FloatTensor(positive_sample_bros),torch.FloatTensor(negative_sample_bros_list),positive_sample_bros_path,negative_sample_bros_list_path

    @staticmethod
    def collate_fn(data):

        '''
        combine batch
        '''
        # negative_sample_size = data[0][2]
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        positive_sample_path = [_[2] for _ in data]
        negative_sample_path = [_[3] for _ in data]

        return positive_sample, negative_sample,positive_sample_path, negative_sample_path

class DatasetV(Dataset):
    def __init__(self, leaves, tree, args, total_level):

        self.len = len(tree) - len(leaves)
        self.all_leaves = leaves
        self.negative_sample_size = args.negative_sample_size
        self.hidden_dim = args.hidden_dim
        self.tree = tree
        self.total_level = total_level
        self.level_set = [[] for i in range(total_level+1)]
        for eachNode in tree:
            l = len(eachNode.path)
            self.level_set[len(eachNode.path)].extend([eachNode.id])


    def __len__(self):
        return self.len

    # override __getitem__
    def __getitem__(self, idx):
        idx = idx + len(self.all_leaves)
        positive_sample_level = len(self.tree[idx].path)
        # other nodes set in the same none-leaf level
        diff_pool = set(self.level_set[positive_sample_level]) - set([idx])
        if len(diff_pool) == 0:
            negative_samples = np.array([-1 for i in range(self.negative_sample_size)])
        elif len(diff_pool) <= self.negative_sample_size:
            negative_samples = np.array(list(diff_pool))
        else:
            negative_samples = np.random.choice(list(diff_pool), size=self.negative_sample_size, replace=False)
        return positive_sample_level, idx, negative_samples


    @staticmethod
    def collate_fn(data):
        '''
        combine batch
        '''

        sample_level = [_[0] for _ in data]
        positive_sample = [_[1] for _ in data]
        negative_sample = [_[2] for _ in data]
        return sample_level, positive_sample, negative_sample



class BidirectionalOneShotIterator(object):
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
    网络训练的数据集生成类
    """
    def __init__(self,parent, tree, children, simMatrix, args):
        self.len = len(children)
        self.args = args
        self.children = children
        self.simMatrix = simMatrix
        self.batchSize = self.args.batch_size

    def __len__(self):
        return self.len

    # override __getitem__
    def __getitem__(self, idx):
        # ids = []
        # emb = []
        #
        # if self.len <= self.args.batch_size:
        #     n = 0
        #     for _ in range(self.args.batch_size):
        #         ids.append(n)
        #         emb.append(self.simMatrix[n])
        #         n = (n + 1) % self.len
        # else:
        #     ids.append(idx)
        #     emb.append(self.simMatrix[idx])
        #     for _ in range(self.args.batch_size - 1):
        #         tar = random.randint(0, self.len-1)
        #         ids.append(tar)
        #         emb.append(self.simMatrix[tar])
        # return ids, emb
        return idx, self.simMatrix[idx]


# class collateFn(data)

    @staticmethod
    def collate_fn(data,batch_size):
        '''
        combine batch
        '''
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

class networkLeavesDataset(Dataset):
    """
    网络训练的数据集生成类
    """
    def __init__(self,simMatrix, args):
        self.len = 1
        self.args = args

        self.simMatrix = simMatrix

    def __len__(self):
        return self.len

    # override __getitem__
    def __getitem__(self, idx):
        ids = []
        emb = []
        # print(idx)
        # print(self.len)
        # print(self.simMatrix)
        n = 0
        for _ in range(self.args.batch_size):
            ids.append(n)
            emb.append(self.simMatrix[n])
            n = (n + 1) % self.len

        return ids, emb



    @staticmethod
    def collate_fn(data):
        '''
        combine batch
        '''
        ids = [_[0] for _ in data]
        emb = [_[1] for _ in data]
        return ids, emb