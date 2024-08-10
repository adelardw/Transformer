import torch
import torch.nn as nn
from pe import PositionalEncoding
from encoder_block import TransformerEncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(self, in_size,
                       vocab_size, 
                       out_size,
                       head_size,
                       num_heads,
                       num_layers,
                       dropout_p,
                       fc_hidden_size,
                       max_seq_len):
        
        super().__init__()

        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.out_size = out_size
        self.dropout_p = dropout_p
        self.fc_hidden_size = fc_hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(self.vocab_size, self.in_size)    
        self.pe = PositionalEncoding(self.max_seq_len, self.in_size)

        self.encoder_blocks = nn.ModuleDict({
            f"encoder_block_{i}": TransformerEncoderBlock(
                in_size=self.in_size if i==0 else self.out_size,
                head_size=self.head_size,
                num_heads=self.num_heads,
                out_size=self.out_size,
                fc_hidden_size=self.fc_hidden_size,
                dropout_p=self.dropout_p
            ) for i in range(self.num_layers)
        })

    def forward(self, encoder_input):

        # Получаем на вход batch_size x seq_len
        encoder_emb = self.embeddings(encoder_input)  # (batch_size, seq_len, emb_size)
        out = self.pe(encoder_emb)
        for block in self.encoder_blocks.values():
            out = block(out, out, out)  # (batch_size, seq_len, out_size)

        return out

