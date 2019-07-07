import math
import random
import torch
import torch.nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
import numpy as np

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    X[X != X] = 0 # remove nan
    return X

def cosine_sim(im, s):
    """Cosine similarity between all the protein pairs
    """
    return im.mm(s.t())

class ContrastiveLoss(torch.nn.Module):
    """
    Taken from: https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0, device=None):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.device = device

    def sim(self, o1, o2):
        return torch.pow(torch.norm(o1 - o2, dim=1), 2)

    # Cosine similarity hard negative mining
    def cosine_forward(self, o1, o2):
        # normalize:
        o1, o2 = l2norm(o1), l2norm(o2)

        # compute image-sentence score matrix
        scores = cosine_sim(o1, o2)
        diagonal = scores.diag().view(o1.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask).to(self.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

    # euclidean distance
    # harder negative mining thanks to jimmy yan
    def forward(self, output1, output2, quant=100):
        # compute the positive loss of all the data
        pos_loss = torch.pow(torch.norm(output2.sub(output1), dim=1), 2)
        neg_loss = torch.empty_like(pos_loss)

        quant = min(quant, output1.size(0)-1)
        # find the hardest negative example
        for i, o_i in enumerate(output1):
            values, indices = torch.topk(torch.norm(output2.sub(o_i), dim=1), quant, largest=False)

            rn = int(random.random()*quant)
            while indices[rn] == i:
                rn = int(random.random()*quant)

            neg_loss[i] = self.margin - values[rn]

        neg_loss = torch.clamp(neg_loss, 0.0)

        return torch.mean(pos_loss) + torch.mean(neg_loss)
        
    # og contrastive loss
    def contrastive_forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
