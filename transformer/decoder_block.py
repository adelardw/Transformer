import torch
import torch.nn as nn
from encoder_block import TransformerEncoderBlock
from mha import MultiHeadAttention
from mask import trill_mask


class TransformerDecoderBlock(nn.Module):
    def __init__(self, in_size, 
                       out_size,
                       head_size,
                       num_heads,
                       dropout_p,
                       encoder_out_size,
                       fc_hidden_size,
                       query_in_size = None):
        
        super().__init__()

        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.out_size = out_size
        self.dropout_p = dropout_p
        self.encoder_out_size = encoder_out_size
        self.fc_hidden_size = fc_hidden_size

        if query_in_size is not None:
            self.query_in_size = query_in_size
        else:
            self.query_in_size = in_size

        self.masked_attention = MultiHeadAttention(in_size=self.in_size,
                                                   num_heads=self.num_heads,
                                                   head_size=self.head_size,
                                                   out_size=self.out_size)

        self.enoder_block = TransformerEncoderBlock(in_size=self.encoder_out_size,
                                                    query_in_size=self.out_size,
                                                    head_size=self.head_size,
                                                    num_heads=self.num_heads,
                                                    out_size=self.out_size,
                                                    dropout_p=self.dropout_p,
                                                    fc_hidden_size=self.fc_hidden_size)
                                                    
        
        if self.in_size != self.out_size:
            self.adapt_residual = nn.Linear(self.in_size, self.out_size, bias=False)
        else:
            self.adapt_residual = nn.Identity()

        self.norm = nn.LayerNorm(self.out_size)
        self.dropout = nn.Dropout(self.dropout_p)
    
    def forward(self, decoder_input, encoder_output):
        
        mask = trill_mask(decoder_input)
        masked = self.masked_attention(decoder_input, decoder_input, decoder_input, mask)
        residual = masked + self.adapt_residual(decoder_input)
        norm = self.dropout(self.norm(residual))

        return self.enoder_block(query=norm, key=encoder_output, value=encoder_output)


