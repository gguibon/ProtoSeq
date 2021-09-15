from unicodedata import bidirectional
import numpy as np
from termcolor import colored

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class CNNseq(nn.Module):
    '''
        An aggregation method that encodes every document through different
        convolution filters (followed by max-over-time pooling).
    '''
    def __init__(self, ebd, args):
        super(CNNseq, self).__init__()
        self.args = args

        self.ebd = ebd

        self.input_dim = self.ebd.embedding_dim

        # Convolution
        self.convs = nn.ModuleList([nn.Conv1d(
                    in_channels=self.input_dim,
                    out_channels=args.cnn_num_filters,
                    kernel_size=K) for K in args.cnn_filter_sizes])

        self.ebd_dim = args.cnn_num_filters * len(args.cnn_filter_sizes)


    def _conv_max_pool(self, x, conv_filter=None, weights=None):
        '''
        Compute sentence level convolution
        Input:
            x:      batch_size, max_doc_len, embedding_dim
        Output:     batch_size, num_filters_total
        '''
        assert(len(x.size()) == 3)

        x = x.permute(0, 2, 1)  # batch_size, embedding_dim, doc_len
        x = x.contiguous()

        # Apply the 1d conv. Resulting dimension is
        # [batch_size, num_filters, doc_len-filter_size+1] * len(filter_size)
        assert(not ((conv_filter is None) and (weights is None)))
        if conv_filter is not None:
            x = [conv(x) for conv in conv_filter]

        elif weights is not None:
            x = [F.conv1d(x, weight=weights['convs.{}.weight'.format(i)],
                        bias=weights['convs.{}.bias'.format(i)])
                for i in range(len(self.args.cnn_filter_sizes))]

        # max pool over time. Resulting dimension is
        # [batch_size, num_filters] * len(filter_size)
        x = [F.max_pool1d(sub_x, sub_x.size(2)).squeeze(2) for sub_x in x]
        

        # concatenate along all filters. Resulting dimension is
        # [batch_size, num_filters_total]
        x = torch.cat(x, 1)
        x = F.relu(x)
        return x

    def forward(self, data, weights=None):
        '''
            @param data dictionary
                @key text: batch_size * max_text_len
            @param weights placeholder used for maml

            @return output: batch_size * embedding_dim
        '''

        device = data['text'].device
        
        # Apply the word embedding, result:  batch_size, doc_len, embedding_dim
        ebd = self.ebd(data, weights)

        # apply 1d conv + max pool, result:  batch_size, num_filters_total        
        ref = tuple(data['text'].size())
        shape = (ref[0], ref[1], ( len(self.args.cnn_filter_sizes) * self.args.cnn_num_filters))
        output = torch.randn(shape).to(device)
        
        if weights is None:
            for i in range(ebd.size(0)):
                out = self._conv_max_pool(ebd[i], conv_filter=self.convs)
                output[i] = out
        else:
            for i in range(ebd.size(0)):
                out = self._conv_max_pool(ebd[i], weights=weights)
                output[i] = out
        
        return output