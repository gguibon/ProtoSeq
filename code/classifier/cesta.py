import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm
from torchcrf import CRF
from code.classifier.base import BASE

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CESTa(BASE):
    def __init__(self, ebd_dim, args):
        super(CESTa, self).__init__(args)
        # Hard coded params from the CESTa paper
        self.args = args
        self.ebd_dim = ebd_dim
        self.utt_out_dim = 100

        self.linear1 = nn.Linear(self.ebd_dim, self.utt_out_dim)

        # Transformer Encoder
        self.src_mask = None
        self.ninp = self.utt_out_dim # output dim from linear mapping cnn -> 100
        self.nhead = 4
        self.nhid = 100
        self.nlayers = 12
        self.transfo_pe_dropout = 0.1
        self.transfo_dropout = 0.1
        self.pos_encoder = PositionalEncoding(self.ninp, dropout=self.transfo_pe_dropout)
        self.transformer = nn.Transformer(d_model=self.ninp, nhead=self.nhead, num_encoder_layers=self.nlayers, dim_feedforward=2048, dropout=0.5)

        # BiLSTM
        self.lstm_units = 30
        self.hidden_bilstm = self.init_hidden_lstm(bidirectional=True, hidden_dim=self.lstm_units)
        self.bilstm = nn.LSTM(self.utt_out_dim, self.lstm_units//2, num_layers=1, bidirectional=True)
        
        # LSTM
        self.hidden_lstm = self.init_hidden_lstm(hidden_dim=self.lstm_units)
        self.lstm = nn.LSTM(self.utt_out_dim, self.lstm_units, num_layers=1, bidirectional=False)

        # Fully Connected Layer
        self.linear = nn.Linear(self.lstm_units, self.args.way)

        # CRF layer
        self.softmax = nn.Softmax(dim=2)
        self.crf = CRF(self.args.way, batch_first=True)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

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

    def init_hidden_lstm(self, bidirectional=False, hidden_dim=300):
        if bidirectional: directions = 2
        else: directions = 1
        if self.args.cuda >= 0:
            return (torch.randn(directions, self.args.context_size, hidden_dim // directions).to("cuda"),
                    torch.randn(directions, self.args.context_size, hidden_dim // directions).to("cuda"))
        else:
            return (torch.randn(directions, self.args.context_size, hidden_dim // directions),
                    torch.randn(directions, self.args.context_size, hidden_dim // directions))

    def forward(self, utt_feats, YS=None, XQ=None, YQ=None, weights=None, return_preds=False, authors=None):
        ## utt_feats is the utterance features from the CNN encoder (cnn.py)
        device = utt_feats.device

        # Add the additional linear layer described in CESTa
        utt_feats = F.relu( self.linear1(utt_feats) ) # map to 100 dimensions


        ### Global Context Encoder (Transformer + BiLSTM)
        if self.src_mask is None or self.src_mask.size(0) != utt_feats.size(0):
            mask = self._generate_square_subsequent_mask(utt_feats.size(0)).to(device)
            self.src_mask = mask
        # Apply Transformer block
        global_x = self.pos_encoder(utt_feats)
        # global_x = self.transformer_encoder(utt_feats, self.src_mask)
        global_x = self.transformer(utt_feats, utt_feats)

        # Apply BiLSTM
        bilstm_out, bilstm_hidden = self.bilstm(global_x, self.hidden_bilstm)

        ### Individual context Encoder (LSTM)
        # individual features retrieval by speaker (wehave two speakers, 1 and 2)
        utt_feats.retain_grad()
        feats_1 = utt_feats.clone()
        feats_2 = utt_feats.clone()
        for i, a in enumerate(authors):
            for j, num in enumerate(a):
                if num == 1: feats_2[i][j] = torch.zeros(utt_feats.size(-1), device=device)
                elif num == 2: feats_1[i][j] = torch.zeros(utt_feats.size(-1), device=device)
        # get indiv representations for each speaker
        indiv_1_out, (indiv_1_hidden, indiv_1_cell) = self.lstm(feats_1, self.hidden_lstm)
        indiv_2_out, (indiv_2_hidden, indiv_2_cell) = self.lstm(feats_2, self.hidden_lstm)

        # # unify modified representations
        indiv_1_out.retain_grad()
        indiv_out = indiv_1_out.clone()
        for i, (a, b) in enumerate(zip(indiv_1_out, indiv_2_out)):
            if torch.all(torch.eq(a, b)): indiv_out[i] = a
            elif torch.all(torch.eq(a, torch.zeros(a.size(-1), device=device))): indiv_out[i] = b
            else: indiv_out[i] = a

        # ### Fusion of both encoders
        X = bilstm_out.add(indiv_out)
        X = F.relu(self.linear(X))

        ### Apply CRFs
        pred = self.softmax(X)
        loss = -self.crf(pred, YS)
        pred = torch.tensor(self.crf.decode(pred)).view(-1)
        YS = YS.view(-1)

        acc = BASE.compute_acc(pred.to(device), YS, nomax=True)
        f1 = BASE.compute_f1(pred, YS, nomax=True, average="macro")
        micro_f1_noneutral = BASE.compute_f1_micro_noneutral(pred, YS, labels=self.args['labels'], nomax=True)
        mcc = BASE.compute_mcc(pred, YS, nomax=True)

        if return_preds:
            return acc, loss, f1, mcc, micro_f1_noneutral, pred, YS

        return acc, loss, f1, mcc, micro_f1_noneutral


