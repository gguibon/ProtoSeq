import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
from code.classifier.base import BASE
from functools import reduce

from torchcrf import CRF


class PROTOseq(BASE):
    '''
        Sequential Prototypical Networks : classifier
        Get as input the output of prior CNN-BiLSTM encoder to complete the ProtoSeq
        Inherits the BASE model
    '''
    def __init__(self, ebd_dim, args):
        super(PROTOseq, self).__init__(args)
        self.ebd_dim = ebd_dim

        self.mlp = self._init_mlp(
                self.ebd_dim, self.args.proto_hidden, self.args.dropout)

        if self.args.crf: 
            self.crf = CRF(self.args.way, batch_first=True)

        # Init the output file to append additional info (if args.out)
        self._init_outputfile(args.suffix_output)
    
    def _compute_prototype_seq(self, XS, YS):
        '''
            Compute the prototype for each class by averaging over the ebd.

            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size

            @return prototype: way x ebd_dim
        '''

        size0 = reduce( lambda x, y: x*y, list(XS.size())[:-1])
        XS = XS.view(size0, XS.size(-1))
        YS = YS.view(-1)

        # Sort YS to make sure classes of the same labels are clustered together
        sorted_YS, indices = torch.sort(YS)
        sorted_XS = XS[indices]
        
        output, inverse_indices = torch.unique(sorted_YS, sorted=True, return_inverse=True)

        prototype = []
        for y in output:
            indices = (sorted_YS == y).nonzero(as_tuple=True)[0]
            prototype.append( torch.mean(sorted_XS[indices], dim=0, keepdim=True) )

        prototype = torch.cat(prototype, dim=0)
        assert prototype.size(0) == self.args.way
        return prototype
    
    def _compute_l2_seq(self, XS, XQ):
        '''
            Compute the pairwise l2 distance for the sequence
            @param XS (support x): support_size x ebd_dim
            @param XQ (support x): query_size x ebd_dim

            @return dist: query_size x support_size
        '''
        
        batch_elements = []

        for batch in XQ:
            diff = XS.unsqueeze(0) - batch.unsqueeze(1)
            dist = torch.norm(diff, dim=2)
            batch_elements.append( dist )

        dist = torch.stack(batch_elements, dim=0)

        return dist

    def forward(self, XS, YS, XQ, YQ, out=False, XS_ids=None, XQ_ids=None):
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

        device = XS.device

        prototype = self._compute_prototype_seq(XS, YS)
        
        pred = -self._compute_l2_seq(prototype, XQ)
        
        if self.args.crf:
            ## the CRF way to compute message labels by maximizing the labels predicted by class prototypes
            # emission scores are relative l2 distances
            loss = -self.crf(pred, YQ)

            pred = torch.tensor(self.crf.decode(pred)).view(-1)
            YQ = YQ.view(-1)

            acc = BASE.compute_acc(pred.to(device), YQ, nomax=True)

            # Weighted F1 and micro f1 ignore neutral class, following related work on DailyDialog
            f1 = BASE.compute_f1(pred, YQ, nomax=True, labels=self.args['labels'])

            micro_f1_noneutral = BASE.compute_f1_micro_noneutral(pred, YQ, labels=self.args['labels'], nomax=True)

            mcc = BASE.compute_mcc(pred, YQ, nomax=True)

            # To save some inside information (only during test)
            if out: 
                self._append_output(YQ, pred, XQ_ids, prototype, XQ) 
        
        else:
            ## The easy way: compute scores on messages' labels predicted by class prototypes
            pred, YQ = pred.view(-1, pred.size(-1)), YQ.view(-1) 

            # Apply weights in order to minimize majority classe from the loss (i.e. the neutral one -- ~80% of labels --)
            if self.args.targetdata == 'dailydialog': 
                weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                class_weights = torch.FloatTensor(weights).to(device)
            else: class_weights = None

            loss = F.cross_entropy(pred, YQ, weight=class_weights)

            acc = BASE.compute_acc(pred, YQ)

            f1 = BASE.compute_f1(pred, YQ, labels=self.args['labels'])

            micro_f1_noneutral = BASE.compute_f1_micro_noneutral(pred, YQ, labels=self.args['labels'])

            mcc = BASE.compute_mcc(pred, YQ)   

            # To save some inside information (only during test)
            if out: 
                self._append_output(YQ, pred, XQ_ids, prototype, XQ)

        return acc, loss, f1, mcc, micro_f1_noneutral
