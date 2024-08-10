import torch.nn as nn
import torch
from decoder_block import TransformerDecoderBlock
from pe import PositionalEncoding

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size,
                       max_seq_len,
                       in_size,
                       out_size,
                       head_size,
                       num_heads,
                       num_layers,
                       dropout_p,
                       fc_hidden_size,
                       encoder_out_size = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.in_size = in_size
        self.out_size = out_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.fc_hidden_size = fc_hidden_size

        if encoder_out_size is not None:
            self.encoder_out_size = encoder_out_size
        else:
            self.encoder_out_size = in_size
        

        self.embeddings = nn.Embedding(self.vocab_size, self.in_size)
        self.pe = PositionalEncoding(self.max_seq_len, self.in_size)

        self.dropout = nn.Dropout(self.dropout_p)

        self.decoder_blocks = nn.ModuleDict({
            f"decoder_block_{i}": TransformerDecoderBlock(in_size= self.in_size if i == 0 else self.out_size,
                                                         num_heads=self.num_heads,
                                                         head_size=self.head_size,
                                                         out_size=self.out_size,
                                                         encoder_out_size=self.encoder_out_size,
                                                         dropout_p=self.dropout_p,
                                                         fc_hidden_size=self.fc_hidden_size)
            for i in range(self.num_layers)
        })
        
        self.fc = nn.Linear(self.out_size, self.vocab_size)

    
    def forward(self, decoder_input, encoder_output):
        embedded = self.embeddings(decoder_input)
        embedded_pe = self.pe(embedded)

        out = self.dropout(embedded_pe)

        for block in self.decoder_blocks.values():
            out = block(out, encoder_output)
        
        return self.fc(out)





