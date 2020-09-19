#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint

import numpy as np
import math

import torch
import torch.nn as nn
from operator import itemgetter



class HierarchyModel(nn.Module):
    def __init__(self,pnode,omega,args,childrenList,res,device,parentDict, tree):
        super(HierarchyModel, self).__init__()
        self.parent = pnode
        self.args = args
        self.tree = tree
        self.childrenList = childrenList
        self.childrenNodesNum = len(childrenList)
        self.circleRange = args.single_circle_range
        self.hiddenDim = args.hidden_dim_t
        self.singleDim = args.single_dim_t
        self.device = device

        self.simOmegaMatrix = omega
        self.parentDict = parentDict
        self.res = res.to(device)
        # The number of children of each node in this layer.
        # self.childrenNumOfEachParent = []

        # 每个孩子节点所能触达的叶子节点个数
        self.eachNodeLeavesNumCounter = []
        for child in childrenList:
            self.eachNodeLeavesNumCounter.append(len(tree[child].leaves))

        # correspondingAllParentsIds = list(itemgetter(*childrenList)(parentDict))
        # eachNodeParentLeavesNum = []
        # for parent in correspondingAllParentsIds:
        #     eachNodeParentLeavesNum.append(len(tree[parent].leaves))
        # 一个list，内部元素为每个孩子节点所触达的叶子节点的个数所占比例
        self.eachNodeLeavesNumRatio = torch.div(torch.Tensor(self.eachNodeLeavesNumCounter), sum(self.eachNodeLeavesNumCounter)).to(device)

        # self.eachNodeLeavesNum = torch.div(torch.Tensor(self.eachNodeLeavesNumCounter), torch.tensor(eachNodeParentLeavesNum)).to(device)

        resEmbLower, resEmbHigher = torch.split(self.res, self.singleDim, dim=1)
        self.parentEmbedding = res[pnode]
        correspondingParentsEmbL,correspondingParentsEmbH = torch.chunk(self.parentEmbedding, 2, dim=0)

        correspondingParentsEmbL = correspondingParentsEmbL.to(device)
        correspondingParentsEmbH = correspondingParentsEmbH.to(device)
        self.correspondingParentsEmbL_ = torch.add(correspondingParentsEmbL, self.circleRange).to(device)
        self.correspondingParentsEmbH_ = torch.add(correspondingParentsEmbH, self.circleRange).to(device)

        self.parentRange = correspondingParentsEmbH - correspondingParentsEmbL
        # print(self.eachNodeLeavesNumRatio)
        # print(self.parentRange)
        initRangeForChildren = torch.mul(self.eachNodeLeavesNumRatio.unsqueeze(1), self.parentRange).to(device)

        # Initialize the embedding of the next layer.
        self.parent_embedding = res[pnode]
        parent_embedding_l, parent_embedding_h = torch.chunk(self.parent_embedding, 2, dim=0)
        for dim in range(self.singleDim):
            layerLowerEmbeddingE = torch.zeros(self.childrenNodesNum, 1).to(device)
            nn.init.uniform_(
                tensor=layerLowerEmbeddingE,
                a=min(parent_embedding_l)+self.circleRange,
                b=max(parent_embedding_h)+self.circleRange
            )

            if dim == 0:
                layerLowerEmbedding = layerLowerEmbeddingE
            else:
                layerLowerEmbedding = torch.cat((layerLowerEmbedding, layerLowerEmbeddingE), 1)

        layerHigherEmbedding = layerLowerEmbedding + initRangeForChildren
        # print(layerLowerEmbedding)
        # print(initRangeForChildren)
        # print(layerHigherEmbedding)
        self.childrenLowerEmbedding = nn.Parameter(layerLowerEmbedding, requires_grad=True)
        self.childrenHigherEmbedding = nn.Parameter(layerHigherEmbedding, requires_grad=True)

        # Initialize the layer-based-distance dict for previous layer.
        # self.layerBasedRes = self.calcLayerBasedDist(layerBasedRes,self.res)


    def calcLayerBasedDist(self,layerBasedRes,res):
        """
        Add nodes' distance at this layer based on the lower bound.
        :param layerBasedRes:
        :param res:
        :return:
        """
        layerContains = self.parentsList
        layerContainsInRes = torch.index_select(
            res,
            dim=0,
            index=torch.tensor(layerContains).to(self.device)
        )
        layerContainsLower, layerContainsHigher = torch.split(layerContainsInRes, self.singleDim, dim=1)

        curLayerDist = self.calcLowerBoundDist(layerContainsLower)

        # Calculate the number of children of each node in this layer.
        for parent in self.parentsList:
            self.childrenNumOfEachParent.append(len(self.tree[parent].direct_children))

        # Accumulate the upper layers' distance.
        if self.curLayer > 0:
            accumulatedLayerDist = curLayerDist + layerBasedRes[self.curLayer - 1]
            accumulatedLayerExpand = torch.repeat_interleave(accumulatedLayerDist, repeats=torch.tensor(self.childrenNumOfEachParent).to(self.device),dim=0)
            accumulatedLayerExpand = torch.repeat_interleave(accumulatedLayerExpand, repeats=torch.tensor(self.childrenNumOfEachParent).to(self.device),dim=1)
            layerBasedRes[self.curLayer] = accumulatedLayerExpand
        else:
            curLayerDistExpand = torch.repeat_interleave(curLayerDist, repeats=torch.tensor(self.childrenNumOfEachParent).to(self.device), dim=0)
            curLayerDistExpand = torch.repeat_interleave(curLayerDistExpand, repeats=torch.tensor(self.childrenNumOfEachParent).to(self.device), dim=1)
            layerBasedRes[self.curLayer] = curLayerDistExpand

        return layerBasedRes


    def calcLowerBoundDist(self, lowerBoundEmbedding, needSum = True):
        """
        Calculate the distance based on the lower bound among embeddings.
        :param lowerBoundEmbedding:
        :return:
        """
        nodesNum = len(lowerBoundEmbedding)
        emb1 = torch.reshape(lowerBoundEmbedding.t(), (lowerBoundEmbedding.numel(), 1))
        emb1 = emb1.repeat(1, nodesNum)
        emb2 = torch.repeat_interleave(lowerBoundEmbedding.t(), repeats=nodesNum, dim=0)

        dimMixedDiff = torch.abs(emb1 - emb2)
        # 每个维度为一个n*n的矩阵，纵向排列
        lowerDist = torch.unsqueeze(dimMixedDiff, 0).reshape(self.singleDim, nodesNum, nodesNum)
        if needSum:
            # 所有维度相加
            lowerDist = torch.sum(lowerDist, dim=0)

        return lowerDist



    def forward(self,idIndexes,omegaEmb,epoch):

        # if epoch % 2 == 0:
        #     self.childrenLowerEmbedding.requires_grad_(True)
        #     self.childrenHigherEmbedding.requires_grad_(False)
        # else:
        #     self.childrenLowerEmbedding.requires_grad_(False)
        #     self.childrenHigherEmbedding.requires_grad_(True)

        if epoch % 4 < 2:
            self.childrenLowerEmbedding.requires_grad_(True)
            self.childrenHigherEmbedding.requires_grad_(False)
        else:
            self.childrenLowerEmbedding.requires_grad_(False)
            self.childrenHigherEmbedding.requires_grad_(True)

        self.childrenEmbedding = torch.cat((self.childrenLowerEmbedding, self.childrenHigherEmbedding), 1)

        # idIndexes:自然顺序，dataloader加载而来；ids：childrenList顺序
        ids = [self.childrenList[i] for i in idIndexes]
        # 本批次样本数
        nodesNum = len(ids)

        omegaEmb4ids = omegaEmb
        finalEmb4ids = torch.index_select(
            self.childrenEmbedding,
            dim=0,
            index=idIndexes
        )

        # Calculate the penalty for the exceed part.
        # print(finalEmb4ids)
        childrenEmbeddingLower, childrenEmbeddingHigher = torch.split(finalEmb4ids, self.singleDim, dim=1)

        # if len(ids) > 1:
        #     correspondingParentsIds = list(itemgetter(*ids)(self.parentDict))
        # else:
        #     correspondingParentsIds = [self.parentDict[ids[0]]]

        # correspondingParentsEmbL = torch.index_select(
        #     resEmbLower,
        #     dim=0,
        #     index=torch.tensor(correspondingParentsIds).to(self.device)
        # )
        # correspondingParentsEmbH = torch.index_select(
        #     resEmbHigher,
        #     dim=0,
        #     index=torch.tensor(correspondingParentsIds).to(self.device)
        # )
        # correspondingParentsEmbL_ = torch.add(correspondingParentsEmbL, self.circleRange)
        # correspondingParentsEmbH_ = torch.add(correspondingParentsEmbH, self.circleRange)

        exceedPart1_1 = self.correspondingParentsEmbL_ - childrenEmbeddingLower
        exceedPart1_2 = self.correspondingParentsEmbL_ - childrenEmbeddingHigher

        exceedPart2_1 = childrenEmbeddingHigher - self.correspondingParentsEmbH_
        exceedPart2_2 = childrenEmbeddingLower - self.correspondingParentsEmbH_

        lossExceed = torch.relu(exceedPart1_1).sum() + torch.relu(exceedPart2_1).sum() + torch.relu(exceedPart1_2).sum() + torch.relu(exceedPart2_2).sum()
        # Calculate the penalty for the overlap part.
        childrenEmbDiff = childrenEmbeddingHigher - childrenEmbeddingLower
        childrenEmbeddingLowerTran1 = torch.reshape(childrenEmbeddingLower.t(),(childrenEmbeddingLower.numel(),1))
        childrenEmbeddingLowerTran1 = childrenEmbeddingLowerTran1.repeat(1, nodesNum)
        childrenEmbeddingHigherTran1 = torch.reshape(childrenEmbeddingHigher.t(),(childrenEmbeddingHigher.numel(),1))
        childrenEmbeddingHigherTran1 = childrenEmbeddingHigherTran1.repeat(1, nodesNum)
        childrenEmbeddingLowerTran2 = torch.repeat_interleave(childrenEmbeddingLower.t(), repeats=nodesNum, dim=0)
        childrenEmbeddingHigherTran2 = torch.repeat_interleave(childrenEmbeddingHigher.t(), repeats=nodesNum, dim=0)

        maxLower = torch.where(childrenEmbeddingLowerTran1 > childrenEmbeddingLowerTran2, childrenEmbeddingLowerTran1,
                            childrenEmbeddingLowerTran2)
        minHigher = torch.where(childrenEmbeddingHigherTran1 < childrenEmbeddingHigherTran2, childrenEmbeddingHigherTran1,
                            childrenEmbeddingHigherTran2)

        overlapPre = minHigher - maxLower
        overlapFilter = torch.ones((nodesNum, nodesNum)) - torch.eye((nodesNum))
        overlapFilter = overlapFilter.repeat(self.singleDim, 1).to(self.device)
        overlap_ = torch.mul(overlapPre, overlapFilter)

        overlapNumerator = torch.where(overlap_ > 0, overlap_, torch.Tensor([0]).to(self.device))

        gapDiff = childrenEmbDiff.t().reshape(childrenEmbeddingLower.numel(), 1)
        gapDiff = HierarchyModel.clip_by_min(gapDiff)
        # overlap = torch.div(overlapNumerator, gapDiff)
        overlap = overlapNumerator
        lossOverlap = overlap.sum()

        # Calculate the penalty for the shape-like part.
        numeratorShapeLike = HierarchyModel.clip_by_min(torch.div(childrenEmbDiff, self.parentRange))

        denominatorShapeLike = torch.index_select(
            self.eachNodeLeavesNumRatio,
            dim=0,
            index=idIndexes
        )


        denominatorShapeLike = denominatorShapeLike.unsqueeze(1)

        shapeLikeDiv = torch.div(numeratorShapeLike, denominatorShapeLike)

        shapeLikeDiv = HierarchyModel.clip_by_max(shapeLikeDiv, ma=1.99, mi=0.01)


        lossShapeLike = torch.abs(torch.tan(torch.mul(torch.add(shapeLikeDiv, -1),math.pi / 2))).sum()

        currentLayerDistance = self.calcLowerBoundDist(childrenEmbeddingLower, needSum=True)

        # correspondingAccumulatedDist = torch.index_select(
        #     self.layerBasedRes[self.curLayer],
        #     dim=0,
        #     index=idIndexes
        # )
        #
        # correspondingAccumulatedDist = torch.index_select(
        #     correspondingAccumulatedDist,
        #     dim=1,
        #     index=idIndexes
        # )

        realDistance = currentLayerDistance

        realDistanceNormed = torch.norm(realDistance)

        distNormed = torch.div(realDistance, HierarchyModel.clip_by_min(realDistanceNormed))
        distNormed = torch.sum(distNormed, dim=0)


        omegaInnerProduct = torch.norm(omegaEmb4ids, dim=1, keepdim=True)
        omegaInnerProduct = torch.mul(omegaInnerProduct, omegaInnerProduct)

        omegaDist = -2 * torch.mm(omegaEmb4ids, omegaEmb4ids.t()) + omegaInnerProduct + omegaInnerProduct.t()
        omegaDist = HierarchyModel.clip_by_min(omegaDist).to(self.device)
        omegaDistNormed = omegaDist / HierarchyModel.clip_by_min(torch.norm(omegaDist))

        lossDistance = torch.norm(omegaDistNormed - distNormed)

        lossGap = torch.relu(self.parentRange * 1 - torch.sum(childrenEmbDiff, dim=0))

        # lossPositive = HierarchyModel.clip_by_min(torch.exp(-1 * (childrenEmbDiff))).sum()
        lossPositive = lossGap.sum()

        # print(self.parentRange)
        # print(childrenEmbDiff)
        # print(torch.sum(childrenEmbDiff,dim=0))
        # exit(1)


        # gap_p_1 = childrenEmbeddingLower - correspondingParentsEmbL_
        # gap_p_2 = correspondingParentsEmbH_ - childrenEmbeddingHigher
        # lossPositive = torch.relu(gap_p_1).sum() + torch.relu(gap_p_2).sum()
        # parentShowCntDict = {}
        # for key in correspondingParentsIds:
        #     parentShowCntDict[key] = parentShowCntDict.get(key, 0) + 1
        # gapSingle = (correspondingParentsEmbDiff - childrenEmbDiff).sum(dim=0)
        # minus = torch.zeros_like(gapSingle).to(self.device)
        # for (k,v) in parentShowCntDict.items():
        #     tmp = torch.index_select(
        #         resEmbDiff,
        #         dim=0,
        #         index=torch.tensor([k]).to(self.device)
        #     )
        #     tmp = torch.mul(tmp, v-1)
        #     minus = torch.add(minus, tmp)
        # lossPositive = torch.relu(gapSingle - minus).sum()

        if epoch % 1000 ==0:
            for k in self.childrenList:
                pprint.pprint(str(k) + '    ' + str(len(self.tree[k].leaves)))
            pppp = self.childrenEmbedding
            pprint.pprint(pppp.detach().numpy())
            l, h = torch.split(pppp, self.args.single_dim_t, dim=1)
            pprint.pprint(np.around((h - l).detach().numpy(), decimals=6))

        if epoch % 100 == 0:
            print('*****************************')
            print('epoch:'+str(epoch))
            print('loss:'+str(self.parent))
            print('lossDistance:%f' % lossDistance)
            print('lossShapeLike:%f' % lossShapeLike)
            print('lossExceed:%f' % lossExceed)
            print('lossOverlap:%f' % lossOverlap)
            print('lossPositive:%f' % lossPositive)

        # mark = epoch % 3
        # if mark == 0 :
        #     loss = self.args.loss_distance * lossDistance + self.args.loss_overlap * lossOverlap
        # elif mark == 1:
        #     loss = self.args.loss_shape * lossShapeLike + self.args.loss_positive * lossPositive
        # else:
        #     loss = self.args.loss_exceed * lossExceed

        mark = epoch % 2
        if mark == 0:
            loss = self.args.loss_distance * lossDistance + self.args.loss_overlap * lossOverlap + self.args.loss_exceed * lossExceed + self.args.loss_shape * lossShapeLike
        # elif mark == 1:
        #     loss = self.args.loss_shape * lossShapeLike + self.args.loss_positive * lossPositive
        else:
            loss = self.args.loss_positive * lossPositive

        # loss = \
        #     self.args.loss_distance * lossDistance + \
        #     self.args.loss_shape * lossShapeLike +  \
        #     self.args.loss_exceed * lossExceed + \
        #     self.args.loss_overlap * lossOverlap + \
        #     self.args.loss_positive * lossPositive

        return loss



    @staticmethod
    def trainStep(model, optimizer,treeIterator, step):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        optimizer.zero_grad()
        data = next(treeIterator)
        loss = model(data,step)
        loss.backward()
        optimizer.step()
        return loss.item(), model.childrenEmbedding

    @staticmethod
    def clip_by_min(x, m=1e-10):
        return torch.clamp(x, m, float('inf'))

    @staticmethod
    def clip_by_max(x, mi=-2, ma=1e5):
        return torch.clamp(x, mi, ma)