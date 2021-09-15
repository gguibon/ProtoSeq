import math
from sys import getsizeof
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm
import code.train.utils as utils

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

class TransformerSeqModel(nn.Module):
	"""Uses a Transformer encoder to encode the input sequence of texts"""

	def __init__(self, ebd, args, dropout=0.2):
		super(TransformerSeqModel, self).__init__()

		self.args = args
		self.ebd = ebd
		self.ebd_dim = self.ebd.embedding_dim
		self.input_dim = self.ebd.embedding_dim

		# ntokens = len(corpus.dictionary) ## no need for ntokens (i.e. embedding size) (we are using pretrained ones)
		self.ntokens = ebd.vocab_size
		self.ninp = args.transfo_emsize # == emsize # ninp number of inputs
		self.nhead = args.transfo_nhead 
		self.nhid = args.transfo_nhid
		self.nlayers = args.transfo_nlayers

		self.model_type = 'Transformer'
		self.src_mask = None
		self.pos_encoder = PositionalEncoding(self.ninp, dropout=args.transfo_pe_dropout)

		encoder_layers = TransformerEncoderLayer(self.ninp, self.nhead, self.nhid, args.transfo_dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)

		self.encoder = self.ebd

	def _generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	def forward(self, src):

		device = src['text'].device

		encoded = self.encoder(src) * math.sqrt(self.ninp)

		ref = tuple(src['text'].size())
		shape = (ref[0], ref[1], self.args.transfo_emsize)
		output = torch.randn(shape).to(device)
		
		for i in range(src['text'].size(0)):
			if self.src_mask is None or self.src_mask.size(0) != src['text'][i].size(0):
				mask = self._generate_square_subsequent_mask(src['text'][i].size(0)).to(device)
				self.src_mask = mask
			
			pos_encoded = self.pos_encoder(encoded[i])
			output_i = self.transformer_encoder(pos_encoded, self.src_mask)
			
			output[i] = torch.mean(output_i, 1)
			
		return output