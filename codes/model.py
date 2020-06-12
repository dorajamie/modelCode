#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import pprint

import numpy as np
import math
import torch
import torch.nn as nn


import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
from torch.utils.data import DataLoader


class ASModel(nn.Module):
    def __init__(self,tree, tree_height, all_leaves,args):
        super(ASModel, self).__init__()
        self.tree = tree
        self.tree_height = tree_height
        self.all_leaves = all_leaves

        self.hidden_dim = args.hidden_dim
        self.single_dim = args.hidden_dim // 2
        self.circle_margin = args.circle_margin
        self.negative_sample_size = args.negative_sample_size

        self.exceed_punishment_ratio = 0.25
        self.gap_punishment_ratio = 0.25
        self.overlap_punishment_ratio = 1 - self.exceed_punishment_ratio - self.gap_punishment_ratio

        # scale each circle by circle_x times
        self.single_circle_range = 2 * math.pi * 1

        subnode_embedding = torch.zeros(len(tree) - 1, self.hidden_dim)
        nn.init.uniform_(
            tensor=subnode_embedding,
            a=self.single_circle_range,
            b=self.single_circle_range * self.tree_height
        )
        # guarantee the lower is samller than the higher
        subnode_embedding_l, subnode_embedding_h = torch.split(subnode_embedding,self.single_dim,dim=1)
        subnode_embedding_l_n = torch.where(subnode_embedding_l < subnode_embedding_h, subnode_embedding_l, subnode_embedding_h)
        subnode_embedding_h_n = torch.where(subnode_embedding_l > subnode_embedding_h, subnode_embedding_l, subnode_embedding_h)
        subnode_embedding = torch.cat((subnode_embedding_l_n,subnode_embedding_h_n),dim=1)

        root_embedding_lower = torch.zeros(1,self.single_dim)
        root_embedding_upper = self.single_circle_range * torch.ones(1,self.single_dim)
        root_embedding = torch.cat((root_embedding_lower,root_embedding_upper), 1)
        self.rootnode_embedding = root_embedding[0]
        # combine the subnode and the root
        node_embedding = torch.cat((subnode_embedding, root_embedding),0)
        self.node_embedding = nn.Parameter(node_embedding,requires_grad=True)
        # self.node_embedding[len(tree) - 1].requires_grad_=False
        # self.node_embedding = torch.cat((self.subnode_embedding, self.rootnode_embedding),0)




    def forward(self, sample, mode):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'h':
            positive_batch_samples, negative_batch_samples,pos_path,neg_path = sample

            # compute the diff of two path
            def calc_diff(path1, path2):
                return self.tree_height - len(set(path1) & set(path2))

            # compute the distance of two nodes
            def calc_dist(path1, path2):    # path1 和 path2 出现了相等的情况?
                # nonzero element num
                diff = calc_diff(path1, path2)
                if diff < 1:
                    diff = 1
                path1 = torch.tensor(path1)
                path2 = torch.tensor(path2)

                path1_ = torch.index_select(
                    self.node_embedding,
                    dim=0,
                    index=path1
                )

                path2_ = torch.index_select(
                    self.node_embedding,
                    dim=0,
                    index=path2
                )

                diff_ = (diff - 1) * self.single_circle_range * torch.ones(self.single_dim)
                # 考虑出现负数的情况，如(第一维差为正，第二维差为负)
                dista_all =  (path1_ - path2_).chunk(2,dim=1)
                dist = 0.5 * (dista_all[0] + dista_all[1])
                dist = dist.sum(0) + diff_

                return dist
            punish = nn.Parameter(torch.zeros(1),requires_grad=True)


            zero_base = torch.zeros(self.single_dim)
            for index,each in enumerate(pos_path):
                pos_one = each[0]
                pos_ano = each[1]
                pos_dist = calc_dist(pos_one,pos_ano)

                for each_neg in neg_path[index]:
                    k = calc_diff(pos_one, each_neg) - 1
                    neg_dist = calc_dist(pos_one, each_neg)
                    tmp = k * self.circle_margin * self.single_circle_range * torch.ones(self.single_dim) + pos_dist - neg_dist
                    tmp_p = torch.where(tmp > 0, tmp, zero_base)
                    punish = punish + torch.norm(tmp_p)

            return punish


        elif mode == 'v':
            sample_level, positive_sample, negative_samples = sample

            positive_score = nn.Parameter(torch.zeros(1),requires_grad=True)

            # calc the punish of positive samples
            for k,v in enumerate(positive_sample):
                # pick up the parent's embedding
                parent_embedding = self.node_embedding[v]
                parent_embedding_l,parent_embedding_h = torch.chunk(parent_embedding,2,dim=0)
                # pick up the children's embedding
                children = list(self.tree[v].direct_children)
                children_embedding = torch.index_select(
                    self.node_embedding,
                    dim=0,
                    index=torch.tensor(children)
                )
                children_embedding_l,children_embedding_h = torch.split(children_embedding,self.single_dim,dim=1)

                zero_base_batch = torch.zeros(children_embedding_l.size())
                zero_base_line = torch.zeros(parent_embedding_l.size())
                exceed_p_1 = torch.add(parent_embedding_l, self.single_circle_range) - children_embedding_l
                excedd_p_2 = children_embedding_h - torch.add(parent_embedding_h, self.single_circle_range)
                exceed_p = torch.where(exceed_p_1 > 0,exceed_p_1,zero_base_batch).sum() \
                         +torch.where(excedd_p_2 > 0,excedd_p_2,zero_base_batch).sum()
                exceed_p = exceed_p * self.exceed_punishment_ratio
                positive_score = positive_score + exceed_p

                gap_p_1 = children_embedding_l-torch.add(parent_embedding_l, self.single_circle_range)
                gap_p_2 =torch.add(parent_embedding_h, self.single_circle_range)-children_embedding_h
                gap_p = torch.where(gap_p_1 > 0,gap_p_1, zero_base_batch).sum() \
                        +torch.where(gap_p_2 > 0, gap_p_2, zero_base_batch).sum()
                gap_p = gap_p * self.gap_punishment_ratio
                positive_score = positive_score + gap_p

                # overlap
                overlap_p = (children_embedding_l[0] - children_embedding_l[0]).sum()

                for i in range(len(children)):
                    for j in range(i,len(children)):
                        max_l = torch.where(children_embedding_l[i] > children_embedding_l[j], children_embedding_l[i], children_embedding_l[j])
                        min_h = torch.where(children_embedding_h[i] > children_embedding_h[j], children_embedding_h[j], children_embedding_h[i])
                        overlap = min_h - max_l
                        overlap_p += torch.where(overlap > 0, overlap, zero_base_line).sum()
                overlap_p = overlap_p * self.overlap_punishment_ratio
                positive_score += overlap_p
            return positive_score
        elif mode == 'v_1':
            positive_sample_path = [_[0] for _ in sample]
            positive_sample = []
            for path in [_[0:-1] for _ in positive_sample_path]:
                positive_sample.extend(path)

            punish = nn.Parameter(torch.zeros(1), requires_grad=True)

            # calc the punish of positive samples
            for k, v in enumerate(positive_sample):
                # pick up the parent's embedding
                if v==len(self.tree) - 1:
                    parent_embedding = self.rootnode_embedding
                else:
                    parent_embedding = self.node_embedding[v]

                parent_embedding_l, parent_embedding_h = torch.chunk(parent_embedding, 2, dim=0)
                # pick up the children's embedding
                children = list(self.tree[v].direct_children)
                children_embedding = torch.index_select(
                    self.node_embedding,
                    dim=0,
                    index=torch.tensor(children)
                )
                children_embedding_l, children_embedding_h = torch.split(children_embedding, self.single_dim, dim=1)

                zero_base_batch = torch.zeros(children_embedding_l.size())
                zero_base_line = torch.zeros(parent_embedding_l.size())
                exceed_p_1 = torch.add(parent_embedding_l, self.single_circle_range) - children_embedding_l
                exceed_p_2 = children_embedding_h - torch.add(parent_embedding_h, self.single_circle_range)
                exceed_p = torch.where(exceed_p_1 > 0, exceed_p_1, zero_base_batch).sum() \
                           + torch.where(exceed_p_2 > 0, exceed_p_2, zero_base_batch).sum()
                exceed_p = exceed_p * self.exceed_punishment_ratio
                pprint.pprint("exceed part punishment:%f" % exceed_p)
                punish = punish + exceed_p

                child_range = (children_embedding_h - children_embedding_l).sum(0)
                parent_range = parent_embedding_h - parent_embedding_l

                gap_p = (parent_range - child_range).sum()

                # gap_p_1 = children_embedding_l - torch.add(parent_embedding_l, self.single_circle_range)
                # gap_p_2 = torch.add(parent_embedding_h, self.single_circle_range) - children_embedding_h
                # gap_p = torch.where(gap_p_1 > 0, gap_p_1, zero_base_batch).sum() \
                #         + torch.where(gap_p_2 > 0, gap_p_2, zero_base_batch).sum()
                # gap_p = gap_p * self.gap_punishment_ratio
                pprint.pprint("gap part punishment:%f" % gap_p)
                if gap_p > 0:
                    punish = punish + gap_p

                # overlap
                overlap_p = (children_embedding_l[0] - children_embedding_l[0]).sum()


                for i in range(len(children)):
                    for j in range(i, len(children)):
                        max_l = torch.where(children_embedding_l[i] > children_embedding_l[j], children_embedding_l[i],
                                            children_embedding_l[j])
                        min_h = torch.where(children_embedding_h[i] > children_embedding_h[j], children_embedding_h[j],
                                            children_embedding_h[i])
                        overlap = min_h - max_l
                        overlap_p += torch.where(overlap > 0, overlap, zero_base_line).sum()



                overlap_p = overlap_p * self.overlap_punishment_ratio

                pprint.pprint("overlap part punishment:%f" % overlap_p)

                punish += overlap_p
            return punish
        else:
            raise ValueError('mode %s not supported' % mode)

    @staticmethod
    def train_step(model, optimizer, train_iterator_v, train_iterator_h):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        # h-direction(global)
        positive_sample_bros, negative_sample_bros_list, positive_sample_bros_path, negative_sample_bros_list_path = next(train_iterator_h)
        # print('positive sample path')
        # print(positive_sample_bros_path)
        # print('negative sample path')
        # print(negative_sample_bros_list_path)

        punish_h = model((positive_sample_bros, negative_sample_bros_list, positive_sample_bros_path,negative_sample_bros_list_path), mode='h')

        # v-direction(local)
        # sample_level, positive_sample, negative_samples = next(train_iterator_v)
        # punish_v = model((sample_level, positive_sample, negative_samples), mode='v')


        # debug: use the same samples as the h-mode
        punish_v = model(positive_sample_bros_path, mode='v_1')

        alpha = 0.4
        loss = alpha * punish_h + (1-alpha) * punish_v

        loss.backward()

        optimizer.step()


        log = {
            'v_loss': punish_v.item(),
            'h_loss': punish_h.item(),
            'loss': loss.item(),
            'embedding':model.node_embedding
        }

        return log

