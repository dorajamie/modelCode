#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import pprint
import time

import numpy as np
import math
import torch
import torch.nn as nn


import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
from torch.utils.data import DataLoader


class TreeModel(nn.Module):
    def __init__(self,omega,args,parent,children,res,leavesCnt,device):
        super(TreeModel, self).__init__()
        self.omega = omega
        self.args = args
        self.leavesCnt = torch.div(torch.Tensor(leavesCnt), sum(leavesCnt)).unsqueeze(1)
        # exceed_part
        self.exceed_punishment_ratio = 0.2
        # gap_part
        self.gap_punishment_ratio = 0.5
        # overlap_part
        self.overlap_punishment_ratio = 1 - self.exceed_punishment_ratio - self.gap_punishment_ratio
        # scale each circle by circle_x times
        self.pi = 2 * math.pi * 10
        # nums of children of current parent
        self.nodes_num = len(children)
        # children&parent
        self.children_nodes = children
        self.parent = parent
        self.single_circle_range = self.args.single_circle_range
        # res
        self.res = res
        # dimensions
        self.hidden_dim = self.args.hidden_dim
        self.single_dim = self.args.single_dim

        # init the nodes_num
        final_embedding = torch.zeros(self.nodes_num, self.hidden_dim)


        self.parent_embedding = res[parent]
        parent_embedding_l, parent_embedding_h = torch.chunk(self.parent_embedding, 2, dim=0)
        self.parent_embedding_l = parent_embedding_l
        self.parent_embedding_h = parent_embedding_h

        self.level_range =  max(parent_embedding_h) - min(parent_embedding_l)

        nn.init.uniform_(
            tensor=final_embedding,
            a=(self.single_circle_range + min(parent_embedding_l)),
            b=(self.single_circle_range + max(parent_embedding_h))
            # a=(self.single_circle_range * 1) ,
            # b=(self.single_circle_range * 5)
        )
        # guarantee the lower is samller than the higher
        node_embedding_l, node_embedding_h = torch.split(final_embedding,self.single_dim,dim=1)
        # nn.init.constant_(
        #     node_embedding_l,
        #     self.single_circle_range + min(parent_embedding_l)
        # )
        # nn.init.constant_(
        #     node_embedding_h,
        #     self.single_circle_range + max(parent_embedding_h)
        # )

        node_embedding_l = torch.where(node_embedding_l < node_embedding_h, node_embedding_l, node_embedding_h)
        node_embedding_h = torch.where(node_embedding_l > node_embedding_h, node_embedding_l, node_embedding_h)
        children_embedding = torch.cat((node_embedding_l,node_embedding_h),dim=1)
        # define children embedding
        self.children_embedding = nn.Parameter(children_embedding, requires_grad=True)
        # parent_embedding
        # if parent > 2320:
        #
        #     print('init')
        #     print(parent)
        #     print(self.parent_embedding)
        #     print(children_embedding)

        # root_embedding_lower = torch.zeros(1,self.single_dim)
        # root_embedding_upper = self.single_circle_range * torch.ones(1,self.single_dim)
        # root_embedding = torch.cat((root_embedding_lower,root_embedding_upper), 1)
        # self.rootnode_embedding = root_embedding[0]
        # combine the subnode and the root
        # node_embedding = torch.cat((subnode_embedding, root_embedding),0)
        # self.node_embedding = nn.Parameter(node_embedding,requires_grad=True)
        # self.node_embedding[len(tree) - 1].requires_grad_=False
        # self.node_embedding = torch.cat((self.subnode_embedding, self.rootnode_embedding),0)
        # 计算好的相似度
        omegaMatrix = self.omega
        # selfOmegaInnerDot = torch.norm(omegaMatrix, dim= 1, keepdim= True)

        selfOmegaInnerDot = np.square(np.linalg.norm(omegaMatrix, axis=1, keepdims=True))
        self.omega_sim_np = -2 * np.dot(omegaMatrix, np.transpose(omegaMatrix)) + selfOmegaInnerDot + np.transpose(selfOmegaInnerDot)
        # print('self.omega_sim_np')
        # print(np.around(self.omega_sim_np,decimals=2))

        self.omega_sim =  TreeModel.clip_by_min(torch.from_numpy(self.omega_sim_np)).to(device)
        self.omegaNormed = self.omega_sim / TreeModel.clip_by_min(torch.norm(self.omega_sim))
        # self.omega_sim =  TreeModel.clip_by_min(self.omega_sim)
        # print('omega_sim')
        # print(np.around(self.omega_sim.numpy(),decimals=2))
        # print(np.around(self.omegaNormed.numpy(),decimals=2))

    @staticmethod
    def clip_by_min(x, m=1e-10):
        return torch.clamp(x, m, float('inf'))
    @staticmethod
    def clip_by_max(x, mi=-2,ma=1e5):
        return torch.clamp(x,mi,ma)

    def forward(self,steps):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''
        # if self.parent == 2331:
        #     print('parent')
        #     print(steps)
        #     print(self.children_embedding)
        children_embedding_l, children_embedding_h = torch.split(self.children_embedding, self.single_dim, dim=1)

        embDiff = children_embedding_h - children_embedding_l
        # print(children_embedding_l)
        # print(children_embedding_h)
        # print(embDiff)
        # diffNorm = torch.norm(embDiff, dim=1, keepdim= True)
        # selfEmbInnerDot = torch.mul(diffNorm,diffNorm)

        # self.emb_sim = -2 * embDiff.mm(embDiff.t()) + selfEmbInnerDot + selfEmbInnerDot.t()
        # embNormed = self.emb_sim / TreeModel.clip_by_min(torch.norm(self.emb_sim))
        # loss_sim = torch.norm(TreeModel.clip_by_min(self.omegaNormed - embNormed) + TreeModel.clip_by_min(embNormed - self.omegaNormed))
        # parent_embedding_l, parent_embedding_h = torch.chunk(self.parent_embedding, 2, dim=0)
        parent_embedding_l = self.parent_embedding_l
        parent_embedding_h = self.parent_embedding_h

        exceed_p_1 = torch.add(parent_embedding_l, self.single_circle_range) - children_embedding_l
        exceed_p_2 = children_embedding_h - torch.add(parent_embedding_h, self.single_circle_range)

        loss_exceed = torch.relu(exceed_p_1).sum() + torch.relu(exceed_p_2).sum()
        # if self.parent == 2331:
        #     print(steps)
        #     print('loss_exceed')
        #     print(loss_exceed)
        # for i in range(self.nodes_num):
        #     for j in range(i, self.nodes_num):
        #         if i == j:
        #             continue
        #         max_l = torch.where(children_embedding_l[i] > children_embedding_l[j], children_embedding_l[i],
        #                             children_embedding_l[j])
        #         min_h = torch.where(children_embedding_h[i] > children_embedding_h[j], children_embedding_h[j],
        #                             children_embedding_h[i])
        #         overlap = min_h - max_l
        #         # overlapPart = torch.relu(overlap).sum()
        #         # print(overlapPart)
        #         loss_overlap += torch.relu(overlap).sum()

        children_embedding_l_1 = torch.reshape(children_embedding_l.t(),(children_embedding_l.numel(),1))
        children_embedding_l_1 = children_embedding_l_1.repeat(1, self.nodes_num)

        children_embedding_h_1 = torch.reshape(children_embedding_h.t(),(children_embedding_h.numel(),1))
        children_embedding_h_1 = children_embedding_h_1.repeat(1, self.nodes_num)

        children_embedding_l_2 = torch.repeat_interleave(children_embedding_l.t(), repeats=self.nodes_num, dim=0)
        children_embedding_h_2 = torch.repeat_interleave(children_embedding_h.t(), repeats=self.nodes_num, dim=0)

        max_l = torch.where(children_embedding_l_1 > children_embedding_l_2, children_embedding_l_1,
                            children_embedding_l_2)
        min_h = torch.where(children_embedding_h_1 < children_embedding_h_2, children_embedding_h_1,
                            children_embedding_h_2)
        overlap_pre = min_h - max_l

        filter = torch.ones((self.nodes_num, self.nodes_num)) - torch.eye((self.nodes_num))
        filter = filter.repeat(self.single_dim, 1)
        overlap_ = torch.mul(overlap_pre, filter)
        # loss_overlap = torch.where(overlap_ > 0, overlap_, torch.Tensor([0])).sum()
        part1 = torch.where(overlap_ > 0, overlap_, torch.Tensor([0]))
        # print('part1')
        # print(part1)
        gapDiff = embDiff.t().reshape(children_embedding_l.numel(), 1)
        gapDiff = TreeModel.clip_by_min(gapDiff)

        # gapDiff = 1 / gapDiff
        # print('gapDiff')
        # print(gapDiff)
        part2 = torch.div(part1,gapDiff)
        # print('part2')
        # print(self.children_embedding)
        # print(part2)
        # print(torch.clamp(torch.tan(3.1415 * part2 / 2), min=1e-10, max=1e5))
        # exit(1)
        # loss_overlap = torch.clamp(torch.tan(3.1415 * part2 / 2), min=1e-10, max=1e5).sum()
        loss_overlap = part2.sum()

        ####### new design start ########
        dim_mixed_diff = torch.abs(children_embedding_l_1 - children_embedding_l_2)

        lower_dist = torch.unsqueeze(dim_mixed_diff, 0).reshape(self.single_dim, self.nodes_num, self.nodes_num)
        lower_dist_normed = torch.norm(lower_dist, dim=(1, 2)).unsqueeze(dim=1).unsqueeze(dim=1)

        self.emb_sim = torch.div(lower_dist,TreeModel.clip_by_min(lower_dist_normed))
        self.emb_sim = torch.sum(self.emb_sim,dim=0)
        loss_sim = torch.norm(self.omegaNormed - self.emb_sim)
        # if self.parent == 2331:
        #     print(steps)
        #     print('loss_sim')
        #     print(loss_sim)

        ####### new design end   ########


        # loss_extra = (1000 * self.leavesCnt *  TreeModel.clip_by_min(torch.exp(-1 * (embDiff)))).sum()
        # loss_extra = (1000 * torch.exp(-1 * torch.div(embDiff, self.leavesCnt) )).sum()

        numerator = TreeModel.clip_by_min(torch.div(embDiff,self.level_range))
        denominator = self.leavesCnt
        div = torch.div(numerator, denominator)
        loss_extra = torch.abs(torch.log(div))
        loss_extra = TreeModel.clip_by_max(loss_extra).sum()
        # cmp1 = torch.div(numerator, denominator)
        # cmp2 = torch.div(denominator, numerator)
        #
        # loss_extra = torch.norm(cmp1 - 1) + torch.norm(cmp2-1)

        # print('abs')
        # print(torch.abs(torch.div(numerator, self.leavesCnt) - 1))
        # loss_extra = torch.abs(torch.exp(TreeModel.clip_by_max(torch.div(numerator, denominator) - 1,ma=10) ) - 1).sum()
        # loss_extra = (1 - torch.div(numerator, denominator))
        # loss_extra = TreeModel.clip_by_min().sum()

        loss_positive = TreeModel.clip_by_min(torch.exp(-1 * (embDiff))).sum()
        # if self.parent == 2331:
        #     print(steps)
        #     print('loss_positive')
        #     print(loss_positive)
        # print('-----------------------------loss_extra:%f' % loss_extra)
        # print(embDiff)
        # print(numerator)
        # print(denominator)
        # print(cmp1-1)
        # print(cmp2-1)
        # print(torch.norm(cmp1 - 1))
        # print(torch.norm(cmp2 - 1))

        # if self.parent == 2331:
        #     print('********************')
        #     print('loss_sim:%f' % loss_sim)
        #     print('loss_exceed:%f' % loss_exceed)
        #     print('loss_extra:%f' % loss_extra)
        #     print('loss_overlap:%f' % loss_overlap)
        #     print('loss_positive:%f' % loss_positive)

        if steps % 1000 ==0:
            pass
            print('********************')
            print('loss_sim:%f' % loss_sim)
            print('loss_exceed:%f' % loss_exceed)
            print('loss_extra:%f' % loss_extra)
            print('loss_overlap:%f' % loss_overlap)
            print('loss_positive:%f' % loss_positive)
            # print(numerator)
            # print(denominator)
            # print(cmp1)
            # print(cmp2)
            # print(cmp1-cmp2)
            # print('self.leavesCnt')
            # print(self.leavesCnt)
            # print('embDiff')
            # print(embDiff)
            # print('self.level_range')
            # print(self.level_range)
            # print('numerator')
            # print(numerator)
            # print('denominator')
            # print(denominator)
            # print('res')
            # print(torch.div(numerator, self.leavesCnt) - 1)


        loss = \
            1 * loss_sim + \
            5 * loss_extra +  \
            2 * loss_exceed + \
            9 * loss_overlap + \
            6 * loss_positive


        return loss



    @staticmethod
    def train_step(model, optimizer,step):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        model.train()
        optimizer.zero_grad()
        loss = model(step)
        loss.backward()
        optimizer.step()


        return loss.item(), model.children_embedding

