#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
from operator import itemgetter



class HierarchyModel(nn.Module):
    def __init__(self,layer,omega,args,childrenList,parentsList,res,device,parentDict,layerBasedRes, tree):
        super(HierarchyModel, self).__init__()

        self.args = args
        self.tree = tree
        self.childrenList = childrenList
        self.parentsList = parentsList
        self.childrenNodesNum = len(childrenList)
        self.parentsNodesNum = len(parentsList)
        self.circleRange = args.single_circle_range
        self.hiddenDim = args.hidden_dim_t
        self.singleDim = args.single_dim_t
        self.device = device
        self.curLayer = layer
        self.omega = omega
        self.parentDict = parentDict
        self.res = res.to(device)
        # The number of children of each node in this layer.
        self.childrenNumOfEachParent = []

        # Calculate the leaf children number of each node in this layer.
        eachNodeLeavesNum = []
        for child in childrenList:
            eachNodeLeavesNum.append(len(tree[child].leaves))
        self.eachNodeLeavesNum = torch.div(torch.Tensor(eachNodeLeavesNum), sum(eachNodeLeavesNum)).to(device)



        # Initialize the embedding of the next layer.
        layerEmbedding = torch.zeros(self.childrenNodesNum, self.hiddenDim).to(device)
        nn.init.uniform_(
            tensor=layerEmbedding,
            a=(self.circleRange * (layer + 1) ),
            b=(self.circleRange * (layer + 2) )
        )
        nodeEmbeddingL, nodeEmbeddingH = torch.split(layerEmbedding, self.singleDim, dim=1)
        # Make the higher is larger than the lower
        nodeEmbeddingL = torch.where(nodeEmbeddingL < nodeEmbeddingH, nodeEmbeddingL, nodeEmbeddingH)
        nodeEmbeddingH = torch.where(nodeEmbeddingL > nodeEmbeddingH, nodeEmbeddingL, nodeEmbeddingH)
        layerEmbedding = torch.cat((nodeEmbeddingL, nodeEmbeddingH), dim=1)
        self.childrenEmbedding = nn.Parameter(layerEmbedding, requires_grad=True)

        # Initialize the layer-based-distance dict for previous layer.
        self.layerBasedRes = self.calcLayerBasedDist(layerBasedRes,self.res)

        # print('omega for all:')
        # print(self.omega)

    def calcLayerBasedDist(self,layerBasedRes,res):
        """
        Add nodes' distance at this layer based on the lower bound.
        :param layerBasedRes:
        :param res:
        :return:
        """
        layerContains = self.parentDict[self.curLayer]

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
        # The index order.
        # The original order.
        ids = [self.childrenList[i] for i in idIndexes]
        # print('idIndexes:')
        # print(idIndexes)
        nodesNum = len(ids)

        omegaEmb4ids = omegaEmb
        finalEmb4ids = torch.index_select(
            self.childrenEmbedding,
            dim=0,
            index=idIndexes
        )
        # parentsEmbLower, parentsEmbHigher = torch.split(self.parentsEmbedding, self.singleDim, dim=1)
        resEmbLower, resEmbHigher = torch.split(self.res, self.singleDim, dim=1)

        # Calculate the penalty for the exceed part.
        childrenEmbeddingLower, childrenEmbeddingHigher = torch.split(finalEmb4ids, self.singleDim, dim=1)

        if len(ids) > 1:
            correspondingParentsIds = list(itemgetter(*ids)(self.parentDict))
        else:
            correspondingParentsIds = [self.parentDict[ids[0]]]

        correspondingParentsEmbL = torch.index_select(
            resEmbLower,
            dim=0,
            index=torch.tensor(correspondingParentsIds).to(self.device)
        )
        correspondingParentsEmbH = torch.index_select(
            resEmbHigher,
            dim=0,
            index=torch.tensor(correspondingParentsIds).to(self.device)
        )
        correspondingParentsEmbL_ = torch.add(correspondingParentsEmbL, self.circleRange)
        correspondingParentsEmbH_ = torch.add(correspondingParentsEmbH, self.circleRange)
        exceedPart1 = correspondingParentsEmbL_ - childrenEmbeddingLower
        exceedPart2 = childrenEmbeddingHigher - correspondingParentsEmbH_
        # print('childrenEmbedding in this step:')
        # print(self.childrenEmbedding)
        # print('selected childrenEmbedding in this step:')
        # print(finalEmb4ids)
        # print('correspondingParentsEmbL')
        # print(correspondingParentsEmbL)
        # print('correspondingParentsEmbH')
        # print(correspondingParentsEmbH)
        # print('lossExceed:')
        # print(exceedPart1)
        # print(exceedPart2)
        lossExceed = torch.relu(exceedPart1).sum() + torch.relu(exceedPart2).sum()
        # print(lossExceed)
        # Calculate the penalty for the overlap part.
        # print('childrenEmbDiff')
        childrenEmbDiff = childrenEmbeddingHigher - childrenEmbeddingLower
        # print(childrenEmbDiff)
        childrenEmbeddingLowerTran1 = torch.reshape(childrenEmbeddingLower.t(),(childrenEmbeddingLower.numel(),1))
        childrenEmbeddingLowerTran1 = childrenEmbeddingLowerTran1.repeat(1, nodesNum)
        childrenEmbeddingHigherTran1 = torch.reshape(childrenEmbeddingHigher.t(),(childrenEmbeddingHigher.numel(),1))
        childrenEmbeddingHigherTran1 = childrenEmbeddingHigherTran1.repeat(1, nodesNum)
        # print('childrenEmbeddingLowerTran1')
        # print(childrenEmbeddingLowerTran1)
        # print('childrenEmbeddingHigherTran1')
        # print(childrenEmbeddingHigherTran1)
        childrenEmbeddingLowerTran2 = torch.repeat_interleave(childrenEmbeddingLower.t(), repeats=nodesNum, dim=0)
        childrenEmbeddingHigherTran2 = torch.repeat_interleave(childrenEmbeddingHigher.t(), repeats=nodesNum, dim=0)
        # print('childrenEmbeddingLowerTran2')
        # print(childrenEmbeddingLowerTran2)
        # print('childrenEmbeddingHigherTran2')
        # print(childrenEmbeddingHigherTran2)
        maxLower = torch.where(childrenEmbeddingLowerTran1 > childrenEmbeddingLowerTran2, childrenEmbeddingLowerTran1,
                            childrenEmbeddingLowerTran2)
        minHigher = torch.where(childrenEmbeddingHigherTran1 < childrenEmbeddingHigherTran2, childrenEmbeddingHigherTran1,
                            childrenEmbeddingHigherTran2)
        # print('maxLower')
        # print(maxLower)
        # print('minHigher')
        # print(minHigher)
        overlapPre = minHigher - maxLower
        # print('overlapPre')
        # print(overlapPre)
        overlapFilter = torch.ones((nodesNum, nodesNum)) - torch.eye((nodesNum))
        overlapFilter = overlapFilter.repeat(self.singleDim, 1).to(self.device)
        overlap_ = torch.mul(overlapPre, overlapFilter)
        # print('overlap_')
        # print(overlap_)
        overlapNumerator = torch.where(overlap_ > 0, overlap_, torch.Tensor([0]).to(self.device))
        # print('overlapNumerator')
        # print(overlapNumerator)
        gapDiff = childrenEmbDiff.t().reshape(childrenEmbeddingLower.numel(), 1)
        gapDiff = HierarchyModel.clip_by_min(gapDiff)
        # print('gapDiff')
        # print(gapDiff)
        overlap = torch.div(overlapNumerator, gapDiff)
        # print('overlap')
        # print(overlap)
        # print('lossOverlap:')
        lossOverlap = overlap.sum()
        # print(lossOverlap)
        # Calculate the penalty for the shape-like part.
        resEmbDiff = resEmbHigher - resEmbLower
        # Expand the parentDiff
        correspondingParentsEmbDiff = torch.index_select(
            resEmbDiff,
            dim=0,
            index=torch.tensor(correspondingParentsIds).to(self.device)
        )
        # print('shapeLike:')
        # print('correspondingParentsEmbDiff')
        # print(correspondingParentsEmbDiff)

        numeratorShapeLike = HierarchyModel.clip_by_min(torch.div(childrenEmbDiff, correspondingParentsEmbDiff))
        # print(numeratorShapeLike)
        denominatorShapeLike = torch.index_select(
            self.eachNodeLeavesNum,
            dim=0,
            index=idIndexes
        )
        # print('denominatorShapeLike')
        denominatorShapeLike = denominatorShapeLike.unsqueeze(1)
        # print(denominatorShapeLike)

        shapeLikeDiv = torch.div(numeratorShapeLike, denominatorShapeLike)
        # print('shapeLikeDiv')
        # print(shapeLikeDiv)

        # lossShapeLike = 100 * torch.add(shapeLikeDiv,-1)**2
        # lossShapeLike = torch.abs(torch.log(shapeLikeDiv))
        shapeLikeDiv = HierarchyModel.clip_by_max(shapeLikeDiv, ma=math.pi * 0.49999)
        lossShapeLike = torch.abs(torch.tan(torch.mul(torch.add(shapeLikeDiv, -1),math.pi / 2)))

        # print('lossShapeLike')
        # print(lossShapeLike)
        lossShapeLike = HierarchyModel.clip_by_max(lossShapeLike).sum()
        # print('lossShapeLike')
        # print(lossShapeLike)

        # Calculate the distance similarity
        # print('distance')
        # print('currentLayerDistance')

        currentLayerDistance = self.calcLowerBoundDist(childrenEmbeddingLower, needSum=True)
        # print(currentLayerDistance)
        #
        # print('accumulated layer dist')
        # print(self.layerBasedRes)

        correspondingAccumulatedDist = torch.index_select(
            self.layerBasedRes[self.curLayer],
            dim=0,
            index=idIndexes
        )

        correspondingAccumulatedDist = torch.index_select(
            correspondingAccumulatedDist,
            dim=1,
            index=idIndexes
        )

        realDistance = correspondingAccumulatedDist + currentLayerDistance
        # print(realDistance)
        realDistanceNormed = torch.norm(realDistance)
        # print('realDistanceNormed')
        # print(realDistanceNormed)
        distNormed = torch.div(realDistance, HierarchyModel.clip_by_min(realDistanceNormed))
        distNormed = torch.sum(distNormed, dim=0)


        omegaInnerProduct = torch.norm(omegaEmb4ids, dim=1, keepdim=True)
        omegaInnerProduct = torch.mul(omegaInnerProduct, omegaInnerProduct)

        omegaDist = -2 * torch.mm(omegaEmb4ids, omegaEmb4ids.t()) + omegaInnerProduct + omegaInnerProduct.t()
        omegaDist = HierarchyModel.clip_by_min(omegaDist).to(self.device)
        omegaDistNormed = omegaDist / HierarchyModel.clip_by_min(torch.norm(omegaDist))

        lossDistance = torch.norm(omegaDistNormed - distNormed)

        lossPositive = HierarchyModel.clip_by_min(torch.exp(-1 * (childrenEmbDiff))).sum()

        if epoch % 100 == 0:
            print('*****************************')
            print('epoch:'+str(epoch))
            print('loss:'+str(self.curLayer))
            print('lossDistance:%f' % lossDistance)
            print('lossShapeLike:%f' % lossShapeLike)
            print('lossExceed:%f' % lossExceed)
            print('lossOverlap:%f' % lossOverlap)
            print('lossPositive:%f' % lossPositive)



        loss = \
            self.args.loss_distance * lossDistance + \
            self.args.loss_shape * lossShapeLike +  \
            self.args.loss_exceed * lossExceed + \
            self.args.loss_overlap * lossOverlap + \
            self.args.loss_positive * lossPositive


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