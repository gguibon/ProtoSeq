import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNseq(nn.Module):

    def __init__(self, ebd, args):
        super(RNNseq, self).__init__()
        self.args = args

        self.ebd = ebd

        self.input_dim = self.ebd.embedding_dim
        
        if args.warmproto:
            self.hidden_size = args.warmprotoconfig["hidden_size"]
            self.num_layers = args.warmprotoconfig["num_layers"]
            self.dropout = args.warmprotoconfig["dropout"]
            self.bidirectional = args.warmprotoconfig["bidirectional"]
        else: 
            self.hidden_size = 128
            self.num_layers = 1
            self.dropout = 0
            self.bidirectional = True

        self.rnn = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers, batch_first=True,
                bidirectional=self.bidirectional, dropout=self.dropout)

        self.ebd_dim = self.hidden_size * 2


    def forward(self, data):
        """
            @param data dictionary
                @key text: batch_size * max_text_len
            @param weights placeholder used for maml

            @return output: batch_size * embedding_dim
        """

        device = data['text'].device

        # Apply the word embedding, result:  batch_size, doc_len, embedding_dim
        ebd = self.ebd(data)

        ref = tuple(data['text'].size())
        shape = (ref[0], ref[1], ref[2], self.ebd.embedding_dim)
        output = torch.randn(shape).to(device)
        
        for i in range(data['text'].size(0)):
            # Apply rnn without pad sequence object (it is already padded from loader.py)
            out, _ = self.rnn(ebd[i])
            output[i] = out

        output = torch.sum(output, dim=2)

        return output
