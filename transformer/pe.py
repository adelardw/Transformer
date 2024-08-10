import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, emb_size):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.embed_size = emb_size

        pe = torch.zeros(self.max_seq_len, self.embed_size)
        pos = torch.arange(self.max_seq_len)
        ind = torch.arange(self.embed_size)[None, ::2]

        for pos in pos:
            pe[pos, ::2] = torch.sin(pos / (10000**(2*ind / self.embed_size)))
            pe[pos, 1::2] = torch.cos(pos / (10000**(2*ind / self.embed_size)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, embedding):

        seq_len = embedding.size(1)

        return embedding + self.pe[:, :seq_len]     
        

