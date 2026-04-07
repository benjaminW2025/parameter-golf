import torch
import torch.nn as nn
import torch.functional as F
from decoder import Decoder


class UniversalTransformer(nn.Module):
    def __init__(self,
                 emb_size,
                 num_attn_heads_encoder,
                 num_attn_heads_decoder,
                 vocab_size,
                 dropout = 0.1,
                 transition = "conv" # or "fft"
                 )
        super().__init__()

        self.encoder = # ben's encoder
        self.decoder = Decoder(
                            emb_size, 
                            num_attn_heads_decoder, 
                            vocab_size, 
                            dropout, 
                            transition)
    
    # TODO: figure out tokenization and where we're gonna put the embedding layers
        
    def encode(self, x):
        # wrap encoder forward call with early stopping mechanism
        return
        
    def decode(self, x, H_enc):
        # wrap decoder forward call with early stopping mechanism
        return

    def forward(self, x): # x is our input STRING (L, B, D)
        # do tokenization
        # add start token?

        # right-shift (before or after encoder?)
        x_shift_right = x[:-1:] # no end token
        x_shift_left = x[1::] # no start token

        H_enc = self.encode(x)
        return self.decode(x, H_enc)
    






        
