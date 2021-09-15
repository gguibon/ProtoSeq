import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
from code.classifier.base import BASE
import numpy as np


class PROTO(BASE):
    '''
        PROTOTYPICAL NETWORK FOR FEW SHOT LEARNING
    '''
    def __init__(self, ebd_dim, args):
        super(PROTO, self).__init__(args)
        self.ebd_dim = ebd_dim

        self.args = args

        self.mlp = self._init_mlp( self.ebd_dim, self.args.proto_hidden, self.args.dropout )

    def _compute_prototype(self, XS, YS):
        '''
            Compute the prototype for each class by averaging over the ebd.

            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size

            @return prototype: way x ebd_dim
        '''
        # sort YS to make sure classes of the same labels are clustered together
        sorted_YS, indices = torch.sort(YS)
        sorted_XS = XS[indices]

        prototype = []
        for i in range(self.args.way):
            prototype.append(torch.mean(
                sorted_XS[i*self.args.shot:(i+1)*self.args.shot], dim=0,
                keepdim=True))

        prototype = torch.cat(prototype, dim=0)

        return prototype

    def forward(self, XS, YS, XQ, YQ):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''
        if self.mlp is not None:
            XS = self.mlp(XS.float())
            XQ = self.mlp(XQ.float())

        try: YS, YQ = self.reidx_y(YS, YQ)
        except: 
            unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
            unique2, inv_Q = torch.unique(YQ, sorted=True, return_inverse=True)
            # print(colored('HERE (proto.py)', 'red'), unique1, unique2 )

        prototype = self._compute_prototype(XS, YS)

        pred = -self._compute_l2(prototype, XQ)

        loss = F.cross_entropy(pred, YQ)

        acc = BASE.compute_acc(pred, YQ)

        f1 = BASE.compute_f1(pred, YQ)

        mcc = BASE.compute_mcc(pred, YQ)    

        micro_f1 = BASE.compute_f1_micro_noneutral(pred, YQ, labels=None)

        return acc, loss, f1, mcc, micro_f1
