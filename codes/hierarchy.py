#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint

import numpy as np
import math
from collections import Counter
import torch
import torch.nn as nn
from operator import itemgetter
np.set_printoptions(threshold=100000000)


class HierarchyModel(nn.Module):
    def __init__(self,layer,omega,args,childrenList,parentsList,res,device,parentDict,layerBasedRes, tree):
        super(HierarchyModel, self).__init__()

        self.debug_layer = 0
        self.debug_epoch = 5400000

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
        self.eachNodeLeavesNumCounter = []
        for child in childrenList:
            self.eachNodeLeavesNumCounter.append(len(tree[child].leaves))

        correspondingAllParentsIds = list(itemgetter(*childrenList)(parentDict))
        eachNodeParentLeavesNum = []
        for parent in correspondingAllParentsIds:
            eachNodeParentLeavesNum.append(len(tree[parent].leaves))

        self.eachNodeLeavesNumRatio = torch.div(torch.Tensor(self.eachNodeLeavesNumCounter), sum(self.eachNodeLeavesNumCounter)).to(device)

        self.eachNodeLeavesNum = torch.div(torch.Tensor(self.eachNodeLeavesNumCounter), torch.tensor(eachNodeParentLeavesNum)).to(device)

        initRangeForChildren = torch.mul(self.eachNodeLeavesNumRatio, self.circleRange).unsqueeze(1).to(device)

        # Initialize the embedding of the next layer.
        for dim in range(self.singleDim):
            layerLowerEmbeddingE = torch.zeros(self.childrenNodesNum, 1).to(device)
            nn.init.uniform_(
                tensor=layerLowerEmbeddingE,
                a=(self.circleRange * (layer + 1)),
                b=(self.circleRange * (layer + 2))
            )
            # nn.init.constant_(
            #     layerLowerEmbeddingE,self.circleRange * (layer + 1)
            # )
            if dim == 0:
                layerLowerEmbedding = layerLowerEmbeddingE
            else:
                layerLowerEmbedding = torch.cat((layerLowerEmbedding, layerLowerEmbeddingE), 1)

        # print(layerLowerEmbedding)
        # print(initRangeForChildren)

        layerHigherEmbedding = layerLowerEmbedding + initRangeForChildren

        self.childrenLowerEmbedding = nn.Parameter(layerLowerEmbedding, requires_grad=True)
        self.childrenHigherEmbedding = nn.Parameter(layerHigherEmbedding, requires_grad=True)

        layerEmbedding = torch.cat((layerLowerEmbedding,layerHigherEmbedding),1)
        # print(layerEmbedding)
        # Initialize the embedding of the next layer.
        # layerEmbedding = torch.zeros(self.childrenNodesNum, self.hiddenDim).to(device)
        # nn.init.uniform_(
        #     tensor=layerEmbedding,
        #     a=(self.circleRange * (layer + 1) ),
        #     b=(self.circleRange * (layer + 2) )
        # )
        #
        # nodeEmbeddingL, nodeEmbeddingH = torch.split(layerEmbedding, self.singleDim, dim=1)
        # # Make the higher is larger than the lower
        # nodeEmbeddingL = torch.where(nodeEmbeddingL < nodeEmbeddingH, nodeEmbeddingL, nodeEmbeddingH)
        # nodeEmbeddingH = torch.where(nodeEmbeddingL > nodeEmbeddingH, nodeEmbeddingL, nodeEmbeddingH)
        # layerEmbedding = torch.cat((nodeEmbeddingL, nodeEmbeddingH), dim=1)
        # self.childrenEmbedding = nn.Parameter(layerEmbedding, requires_grad=True)

        self.childrenEmbedding = torch.cat((self.childrenLowerEmbedding,self.childrenHigherEmbedding),1)


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

        if epoch % 2 == 0:
            pass




        # The index order.
        # The original order.
        ids = [self.childrenList[i] for i in idIndexes]
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('idIndexes:')
            print(idIndexes)
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

        # correspondingParentsDistinctIds = list(set(correspondingParentsIds))
        # parentShowCntDict = Counter(correspondingParentsIds)


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

        exceedPart1_1 = correspondingParentsEmbL_ - childrenEmbeddingLower
        exceedPart1_2 = correspondingParentsEmbL_ - childrenEmbeddingHigher

        exceedPart2_1 = childrenEmbeddingHigher - correspondingParentsEmbH_
        exceedPart2_2 = childrenEmbeddingLower - correspondingParentsEmbH_

        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('res')
            print(self.res)
            print('childrenEmbedding in this step:')
            print(self.childrenEmbedding)
            print('selected childrenEmbedding in this step:')
            print(finalEmb4ids)
            print('correspondingParentsEmbL')
            print(correspondingParentsEmbL)
            print('correspondingParentsEmbH')
            print(correspondingParentsEmbH)
            print('lossExceed:')
            print(torch.relu(exceedPart1_1))
            print(torch.relu(exceedPart1_2))
            print(torch.relu(exceedPart2_1))
            print(torch.relu(exceedPart2_2))
        lossExceed = torch.relu(exceedPart1_1).sum() + torch.relu(exceedPart2_1).sum() + torch.relu(exceedPart1_2).sum() + torch.relu(exceedPart2_2).sum()
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print(lossExceed)
        # Calculate the penalty for the overlap part.
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('childrenEmbDiff')
        childrenEmbDiff = childrenEmbeddingHigher - childrenEmbeddingLower
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print(childrenEmbDiff)
        childrenEmbeddingLowerTran1 = torch.reshape(childrenEmbeddingLower.t(),(childrenEmbeddingLower.numel(),1))
        childrenEmbeddingLowerTran1 = childrenEmbeddingLowerTran1.repeat(1, nodesNum)
        childrenEmbeddingHigherTran1 = torch.reshape(childrenEmbeddingHigher.t(),(childrenEmbeddingHigher.numel(),1))
        childrenEmbeddingHigherTran1 = childrenEmbeddingHigherTran1.repeat(1, nodesNum)
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('childrenEmbeddingLowerTran1')
            print(childrenEmbeddingLowerTran1)
            print('childrenEmbeddingHigherTran1')
            print(childrenEmbeddingHigherTran1)
        childrenEmbeddingLowerTran2 = torch.repeat_interleave(childrenEmbeddingLower.t(), repeats=nodesNum, dim=0)
        childrenEmbeddingHigherTran2 = torch.repeat_interleave(childrenEmbeddingHigher.t(), repeats=nodesNum, dim=0)
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('childrenEmbeddingLowerTran2')
            print(childrenEmbeddingLowerTran2)
            print('childrenEmbeddingHigherTran2')
            print(childrenEmbeddingHigherTran2)
        maxLower = torch.where(childrenEmbeddingLowerTran1 > childrenEmbeddingLowerTran2, childrenEmbeddingLowerTran1,
                            childrenEmbeddingLowerTran2)
        minHigher = torch.where(childrenEmbeddingHigherTran1 < childrenEmbeddingHigherTran2, childrenEmbeddingHigherTran1,
                            childrenEmbeddingHigherTran2)
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('maxLower')
            print(maxLower)
            print('minHigher')
            print(minHigher)
        overlapPre = minHigher - maxLower
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('overlapPre')
            print(overlapPre)
        overlapFilter = torch.ones((nodesNum, nodesNum)) - torch.eye((nodesNum))
        overlapFilter = overlapFilter.repeat(self.singleDim, 1).to(self.device)
        overlap_ = torch.mul(overlapPre, overlapFilter)
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('overlap_')
            print(overlap_)
        overlapNumerator = torch.where(overlap_ > 0, overlap_, torch.Tensor([0]).to(self.device))
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('overlapNumerator')
            print(overlapNumerator)
        gapDiff = childrenEmbDiff.t().reshape(childrenEmbeddingLower.numel(), 1)
        gapDiff = HierarchyModel.clip_by_min(gapDiff)
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('gapDiff')
            print(gapDiff)
        overlap = torch.div(overlapNumerator, gapDiff)
        # overlap = overlapNumerator
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('overlap')
            print(overlap)
            print('lossOverlap:')
        # overlap = torch.abs(torch.tan(torch.mul(overlap, math.pi / 2.01)))
        # overlap = HierarchyModel.clip_by_max(overlap,ma=5)
        # print(childrenEmbDiff)
        # print(overlap)
        # if epoch % 100 == 0:
        #     a = overlap
        #     print(a.detach().numpy())
        # overlapDragLower =

        lowerA2HigherB = torch.abs(childrenEmbeddingLowerTran1 - childrenEmbeddingHigherTran2)
        higherA2LowerB = torch.abs(childrenEmbeddingHigherTran1 - childrenEmbeddingLowerTran2)
        overlapDrag = torch.where(lowerA2HigherB < higherA2LowerB, lowerA2HigherB, higherA2LowerB)
        # overlapDrag = lowerA2HigherB
        findSubset = torch.where(overlap == 1, overlap, torch.Tensor([0]).to(self.device))
        findSubsetT = torch.cat(findSubset.t().chunk(self.singleDim, 1), 0)

        tmp = torch.add(findSubsetT, findSubset)
        selectedSubset = torch.mul(tmp, overlapDrag)
        finalOverlap = torch.where(selectedSubset > 0, selectedSubset, overlap)
        lossOverlap = overlap.sum()
        # lossOverlap  = finalOverlap.sum()
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('lowerA2HigherB')
            print(lowerA2HigherB)
            print('overlapDrag')
            print(overlapDrag)
            print('findSubset')
            print(findSubset)
            print('findSubsetT')
            print(findSubsetT)
            print('tmp')
            print(tmp)
            print('selectedSubset')
            print(selectedSubset)
            print('finalOverlap')
            print(finalOverlap)
            print('lossOverlap')
            print(lossOverlap)
        # Calculate the penalty for the shape-like part.
        resEmbDiff = resEmbHigher - resEmbLower
        # Expand the parentDiff
        correspondingParentsEmbDiff = torch.index_select(
            resEmbDiff,
            dim=0,
            index=torch.tensor(correspondingParentsIds).to(self.device)
        )
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('shapeLike:')
            print('correspondingParentsEmbDiff')
            print(correspondingParentsEmbDiff)



        numeratorShapeLike = HierarchyModel.clip_by_min(torch.div(childrenEmbDiff, correspondingParentsEmbDiff))

        denominatorShapeLike = torch.index_select(
            self.eachNodeLeavesNum,
            dim=0,
            index=idIndexes
        )
        # shapeA = torch.div(childrenEmbDiff, correspondingParentsEmbDiff)


        denominatorShapeLike = denominatorShapeLike.unsqueeze(1)

        shapeLikeDiv = torch.div(numeratorShapeLike, denominatorShapeLike)
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print(numeratorShapeLike)
            print(denominatorShapeLike)
            print(shapeLikeDiv)

        # print(shapeLikeDiv)
        shapeLikeDiv = HierarchyModel.clip_by_max(shapeLikeDiv, ma=1.99, mi=0.01)
        # print(shapeLikeDiv)
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('after clamp:')
            print(shapeLikeDiv)

        lossShapeLike = torch.abs(torch.tan(torch.mul(torch.add(shapeLikeDiv, -1),math.pi / 2))).sum()


        # lossShapeLike = HierarchyModel.clip_by_max(lossShapeLike).sum()
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('lossShapeLike')
            print(lossShapeLike)

        # Calculate the distance similarity
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('distance')
            print('currentLayerDistance')

        currentLayerDistance = self.calcLowerBoundDist(childrenEmbeddingLower, needSum=True)
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print(currentLayerDistance)

            print('accumulated layer dist')
            print(self.layerBasedRes)

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
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print(realDistance)
        realDistanceNormed = torch.norm(realDistance)
        if self.curLayer == self.debug_layer and epoch > self.debug_epoch:
            print('realDistanceNormed')
            print(realDistanceNormed)
            if epoch == self.debug_epoch +2:
                exit(1)
        distNormed = torch.div(realDistance, HierarchyModel.clip_by_min(realDistanceNormed))
        distNormed = torch.sum(distNormed, dim=0)


        omegaInnerProduct = torch.norm(omegaEmb4ids, dim=1, keepdim=True)
        omegaInnerProduct = torch.mul(omegaInnerProduct, omegaInnerProduct)

        omegaDist = -2 * torch.mm(omegaEmb4ids, omegaEmb4ids.t()) + omegaInnerProduct + omegaInnerProduct.t()
        omegaDist = HierarchyModel.clip_by_min(omegaDist).to(self.device)
        omegaDistNormed = omegaDist / HierarchyModel.clip_by_min(torch.norm(omegaDist))

        lossDistance = torch.norm(omegaDistNormed - distNormed)

        lossPositive = HierarchyModel.clip_by_min(torch.exp(-1 * (childrenEmbDiff))).sum()
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
            print('loss:'+str(self.curLayer))
            print('lossDistance:%f' % lossDistance)
            print('lossShapeLike:%f' % lossShapeLike)
            print('lossExceed:%f' % lossExceed)
            print('lossOverlap:%f' % lossOverlap)
            print('lossPositive:%f' % lossPositive)
            if self.debug_layer == self.curLayer and epoch > self.debug_epoch:
                exit(1)


        loss = \
            self.args.loss_distance * lossDistance + \
            self.args.loss_shape * lossShapeLike +  \
            self.args.loss_exceed * lossExceed + \
            self.args.loss_overlap * lossOverlap + \
            self.args.loss_positive * lossPositive

        # loss = self.args.loss_overlap * lossOverlap

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