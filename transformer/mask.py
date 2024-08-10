import torch

def trill_mask(decoder_input):
    batch_size, seq_len, _ = decoder_input.shape

    mask = torch.tril(torch.ones((seq_len, seq_len))).expand(batch_size, 1, seq_len, seq_len).bool()

    return mask