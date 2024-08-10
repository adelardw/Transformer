import torch
import torch.nn as nn
from mha import MultiHeadAttention
from collections import OrderedDict

class TransformerEncoderBlock(nn.Module):
    def __init__(self, in_size, 
                       head_size,
                       num_heads,
                       out_size,
                       dropout_p,
                       fc_hidden_size,
                       query_in_size=None):
        super().__init__()

        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.out_size = out_size
        self.dropout_p = dropout_p
        self.fc_hidden_size = fc_hidden_size
        if query_in_size is not None:
            self.query_in_size = query_in_size
        else:
            self.query_in_size = in_size


        self.attention = MultiHeadAttention(in_size=self.in_size,
                                            num_heads=self.num_heads,
                                            head_size=self.head_size,
                                            out_size = self.out_size,
                                            query_in_size=self.query_in_size)
        
        if self.in_size != self.out_size:
            self.adapt_residual = nn.Linear(self.query_in_size, self.out_size, bias=False)
        else:
            self.adapt_residual = nn.Identity()
        
        self.norm_1 = nn.LayerNorm(self.out_size)
        self.dropout_1 = nn.Dropout(self.dropout_p)

        self.norm_1 = nn.LayerNorm(self.out_size)
        self.dropout_1 = nn.Dropout1d(dropout_p)
        
        self.feed_forward = nn.Sequential(OrderedDict([
            ("lin_1", nn.Linear(in_features=self.out_size,
                                out_features=self.fc_hidden_size)),
            ("act", nn.ReLU()),
            ("lin_2", nn.Linear(in_features=self.fc_hidden_size,
                                out_features=self.out_size)),
        ]))

        self.norm_2 = nn.LayerNorm(self.out_size)
        self.dropout_2 = nn.Dropout1d(dropout_p)

    def forward(self, query, key, value):
    
        # Получаем на вход 3 тензора batch_size x seq_len x in_size
        attention_out = self.attention(key=key, query= query, value=value)
        attention_residual_out = attention_out + self.adapt_residual(query)
        norm_1_out = self.dropout_1(self.norm_1(attention_residual_out))

        ff_out = self.feed_forward(norm_1_out)
        ff_residual_out = ff_out + norm_1_out
        norm_2_out = self.dropout_2(self.norm_2(ff_residual_out))
        return norm_2_out



