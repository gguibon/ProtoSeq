import torch, os
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, classification_report, confusion_matrix, roc_curve, auc
import numpy as np


class BASE(nn.Module):
    '''
        BASE model
    '''
    def __init__(self, args):
        super(BASE, self).__init__()
        self.args = args

        # cached tensor for speed
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)

    def _compute_l2(self, XS, XQ):
        '''
            Compute the pairwise l2 distance
            @param XS (support x): support_size x ebd_dim
            @param XQ (support x): query_size x ebd_dim

            @return dist: query_size x support_size

        '''
        diff = XS.unsqueeze(0) - XQ.unsqueeze(1)
        dist = torch.norm(diff, dim=2)

        return dist

    def _compute_cos(self, XS, XQ):
        '''
            Compute the pairwise cos distance
            @param XS (support x): support_size x ebd_dim
            @param XQ (support x): query_size x ebd_dim

            @return dist: query_size support_size

        '''
        dot = torch.matmul(
                XS.unsqueeze(0).unsqueeze(-2),
                XQ.unsqueeze(1).unsqueeze(-1)
                )
        dot = dot.squeeze(-1).squeeze(-1)

        scale = (torch.norm(XS, dim=1).unsqueeze(0) *
                 torch.norm(XQ, dim=1).unsqueeze(1))

        scale = torch.max(scale,
                          torch.ones_like(scale) * 1e-8)

        dist = 1 - dot/scale

        return dist

    def reidx_y(self, YS, YQ):
        '''
            Map the labels into 0,..., way
            @param YS: batch_size
            @param YQ: batch_size

            @return YS_new: batch_size
            @return YQ_new: batch_size
        '''
        unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
        unique2, inv_Q = torch.unique(YQ, sorted=True, return_inverse=True)

        if len(unique1) != len(unique2):
            raise ValueError(
                'Support set classes are different from the query set')

        if len(unique1) != self.args.way:
            raise ValueError(
                'Support set classes are different from the number of ways')

        if int(torch.sum(unique1 - unique2).item()) != 0:
            raise ValueError(
                'Support set classes are different from the query set classes')

        Y_new = torch.arange(start=0, end=self.args.way, dtype=unique1.dtype,
                device=unique1.device)

        return Y_new[inv_S], Y_new[inv_Q]

    def _init_mlp(self, in_d, hidden_ds, drop_rate):
        modules = []

        for d in hidden_ds[:-1]:
            modules.extend([
                nn.Dropout(drop_rate),
                nn.Linear(in_d, d),
                nn.ReLU()])
            in_d = d

        modules.extend([
            nn.Dropout(drop_rate),
            nn.Linear(in_d, hidden_ds[-1])])

        return nn.Sequential(*modules)

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size

            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    def _init_outputfile(self, suffix):
        '''
            initialize output file to track outputs
        '''
        # creates the dumps directory if it does not exists
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "dumps"))
        if not os.path.exists(out_dir): os.makedirs(out_dir)

        with open('dumps/TRUE{}.tsv'.format(suffix), 'wb') as f: f.write(b'')
        with open('dumps/PRED{}.tsv'.format(suffix), 'wb') as f: f.write(b'')
        with open('dumps/ids{}.tsv'.format(suffix), 'wb') as f: f.write(b'')
        with open('dumps/protos{}.tsv'.format(suffix), 'wb') as f: f.write(b'')
        with open('dumps/query_vectors{}.tsv'.format(suffix), 'wb') as f: f.write(b'')
        self.suffix = suffix

    def _append_output(self, true, pred, ids, prototypes, XQ):
        '''
            append result into output files
            @param true: numpy array of y_true
            @param pred: numpy array of y_pred
            @param ids: numpy array of ids
            @param prototypes: prototypes tensors from Prototypical Networks
            @param queries: query tensors from encoder
        '''
        with open('dumps/TRUE{}.tsv'.format(self.suffix), 'ab') as f:
            # np.savetxt(f, true.cpu().detach().numpy())
            np.save(f, true.cpu().detach().numpy())
        
        with open('dumps/PRED{}.tsv'.format(self.suffix), 'ab') as f:
            # np.savetxt(f, pred.cpu().detach().numpy())
            np.save(f, pred.cpu().detach().numpy())
 
        with open('dumps/ids{}.tsv'.format(self.suffix), 'ab') as f:
            # np.savetxt(f, ids.cpu().detach().numpy())
            np.save(f, ids.cpu().detach().numpy())

        with open('dumps/protos{}.tsv'.format(self.suffix), 'ab') as f:
            # np.savetxt(f, prototypes.cpu().detach().numpy())
            np.save(f, prototypes.cpu().detach().numpy())

        with open('dumps/query_vectors{}.tsv'.format(self.suffix), 'ab') as f:
            # np.savetxt(f, XQ.cpu().detach().numpy())   
            np.save(f, XQ.cpu().detach().numpy())      

    @staticmethod
    def compute_acc(pred, true, dim=1, nomax=False):
        '''
            Compute the accuracy.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        if nomax: return torch.mean((pred == true).float()).item()
        else: return torch.mean((torch.argmax(pred, dim=dim) == true).float()).item()
        
    @staticmethod
    def compute_f1(y_pred, true, dim=1, nomax=False,  labels=None, average='weighted'):
        '''
            Compute the weighted f1 score.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        if not nomax: _, y_pred = torch.max(y_pred, dim)

        f1 = f1_score(true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average=average, labels=labels )

        return f1

    @staticmethod
    def compute_mcc(y_pred, true, dim=1, nomax=False):
        '''
            Compute the matthews correlation coeficient.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        if not nomax: _, y_pred = torch.max(y_pred, dim)

        mcc = matthews_corrcoef(true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

        return mcc

    @staticmethod
    def compute_f1_micro_noneutral(y_pred, true, dim=1, nomax=False, labels=None):
        '''
            Compute the weighted f1 score.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        if not nomax: _, y_pred = torch.max(y_pred, dim)

        f1 = f1_score(true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average='micro', labels=labels)

        return f1