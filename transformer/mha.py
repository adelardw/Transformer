import torch
from torch import nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, in_size, num_heads,head_size, out_size, query_in_size=None):
        super().__init__()

        self.in_size = in_size
        self.num_heads = num_heads
        self.out_size = out_size
        self.head_size = head_size
        if query_in_size is not None:
            self.query_in_size = query_in_size
        else:
            self.query_in_size = in_size

        self.q_weights = nn.Linear(self.query_in_size, self.num_heads * self.head_size, bias=False)
        self.k_weights = nn.Linear(self.in_size, self.num_heads * self.head_size, bias=False)
        self.v_weights = nn.Linear(self.in_size, self.num_heads * self.head_size, bias=False)

        self.o_weights = nn.Linear(self.num_heads * self.head_size, self.out_size)
    
    def forward(self, query, key, value, mask=None):
        # input
        # (batch size, query seq len or seq len, in size or query in size)

        batch_size = query.size()[0]
        query_seq_len = query.size()[1]
        seq_len = key.size()[1]

        q = self.q_weights(query).view(batch_size, query_seq_len, self.num_heads, self.head_size)
        k = self.k_weights(key).view(batch_size, seq_len, self.num_heads, self.head_size)
        v = self.v_weights(value).view(batch_size, seq_len, self.num_heads, self.head_size)

        # (batch size, num heads, query seq len or seq len, head size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # (b, n, q, h) @ (b, n, l, h).T -> (b, n, q, h) @ (b, n, h, l) -> (b,n, q, l)

        relevance = q @ k.transpose(2, 3) / math.sqrt(self.head_size)

        if mask is not None:
            relevance = relevance.masked_fill(mask, -1e18)
        
        relevance = F.softmax(relevance, dim=-1)

        # (b, n, q, l) @ (b, n, l, h) -> (b, n, q, h)

        head_i = relevance @ v

        out = head_i.transpose(1, 2).reshape(batch_size, query_seq_len, self.num_heads * self.head_size)

        return self.o_weights(out)
    


