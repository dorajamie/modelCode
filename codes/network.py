#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn




class NetworkModel(nn.Module):
    def __init__(self,children, args):
        super(NetworkModel, self).__init__()
        self.args = args
        self.childrenList = children
        embedding = torch.zeros( len(children) , self.args.hidden_dim_n)
        nn.init.uniform_(
            tensor=embedding,
            a=-1,
            b=1
        )

        w = torch.zeros(self.args.hidden_dim_n, len(self.childrenList))
        nn.init.uniform_(
            tensor=w,
            a=-1,
            b=1
        )

        b = torch.zeros(len(self.childrenList))
        nn.init.uniform_(
            tensor=b,
            a=-1,
            b=1
        )
        self.w = nn.Parameter(w, requires_grad=True)
        self.b = nn.Parameter(b, requires_grad=True)
        self.node_embedding = nn.Parameter(embedding, requires_grad=True)


    def forward(self, data):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        ids = data[0]
        emb = data[1]

        emb_selected = torch.index_select(
            self.node_embedding,
            dim=0,
            index=torch.tensor(ids)
        )
        self.sof = torch.softmax(torch.mm(emb_selected, self.w) + self.b,  dim=0,dtype=torch.float)
        self.sof = torch.clamp(self.sof, 1e-10, float('inf'))
        self.loss = torch.mean(-torch.sum(torch.tensor(emb) * torch.log(self.sof), dim=1))

        return self.loss


    @staticmethod
    def train_step(model, optimizer, train_iterator):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        model.train()
        optimizer.zero_grad()
        data = next(train_iterator)

        loss = model(data)
        loss.backward()
        optimizer.step()
        return loss,model.node_embedding

