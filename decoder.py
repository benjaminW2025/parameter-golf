import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self,
                 embedding_size,
                 num_attn_heads,
                 vocab_size,
                 dropout = 0.1,
                 use_conv = False,
                 ):
        super().__init__()

        # Params
        self.embedding_size = embedding_size
        self.num_attn_heads = num_attn_heads

        # Attention layers
        self.multihead_attn = nn.modules.activation.MultiheadAttention(
            embed_dim=embedding_size, 
            num_heads=num_attn_heads, 
            batch_first=True
        )
        self.multihead_self_attn = nn.modules.activation.MultiheadAttention(
            embed_dim=embedding_size, 
            num_heads=num_attn_heads, 
            batch_first=True
        )

        # Dropout
        self.dropout = nn.Dropout(p=dropout) # play with this?

        # Layer Normalization
        self.layer_norm_1 = nn.LayerNorm(embedding_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_size)
        self.layer_norm_3 = nn.LayerNorm(embedding_size)

        # Transition Function
        self.transition_func = FFNNHead()

        # Final Linear Layer
        self.to_vocab_logits = nn.Linear(embedding_size, vocab_size)


    def get_time_and_positional_embedding(seq_size, embedding_size, t):
        # position and dimension indices
        i = torch.arange(seq_size).unsqueeze(1)
        j = torch.arange(embedding_size // 2)

        # denominator term (depends only on j)
        denom = 10000 ** ((2.0 * j) / embedding_size)
        denom = denom.unsqueeze(0)

        # compute position + time terms
        pos_term = i / denom            
        time_term = t / denom            
        angles = pos_term + time_term 

        # allocate output
        P = torch.zeros(seq_size, embedding_size)

        # fill even and odd indices
        P[:, 0::2] = torch.sin(angles)
        P[:, 1::2] = torch.cos(angles)

        return P
    

    def call_block(self, H_prev, H_enc, t): # H_prev is (batch x seq_size x embd_size) embedding tensor
        B, L, D = H_prev.shape
        """
        Make sure the batch_first is consistent with the rest of the model (i.e., Ben's encoder)
        """

        # Get Positional and Time Embeddings
        P_t = self.get_time_and_positional_embedding(self.embedding_size, t)
        H_p_and_t = H_prev + P_t

        # Self-Attention Block (left attention only)
        attn_out_self, _ = self.multihead_self_attn(
            H_p_and_t, H_p_and_t, H_p_and_t, is_causal=True
        )
        A_t = self.layer_norm_1(H_p_and_t + self.dropout(attn_out_self)) 
        
        # Cross-Attention Block
        attn_out_cross, _ = self.multihead_attn(
            A_t, H_enc, H_enc
        )
        B_t = self.layer_norm_2(A_t + self.dropout(attn_out_cross))

        # Transition Function
        ff_out = self.transition_func(B_t)
        C_t = self.layer_norm_3(B_t + self.dropout(ff_out))

        return C_t

    def forward(self, H_0, H_enc, T): # input: batch, sequence, embedding
        x = H_0

        for t in range(T): # replace this for early-stopping mechanism
            x = self.call_block(x, H_enc, t)

        logits = self.to_vocab_logits(x)

        return logits
    


class ConvHead(nn.Module):
    """
    To my understanding, this transition function is first a 1D 'depthwise' convolution over each position (this means d kernels) and then
    a 'pointwise' convolution accross embeddings (apparently this is just a square linear layer). Dimension is preserved, but information is
    'mixed' accross position and channel
    """
    def __init__(self, emb_size, kernel_size):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=emb_size,
            out_channels=emb_size,
            kernel_size=kernel_size, # play with this: how much mixing accross positions?
            padding=kernel_size // 2, # overhang when on first/last position is k//2
            groups=emb_size # gives each channel its own kernel
        )

        self.pointwise_conv = nn.Conv1d(
            in_channels=emb_size,
            out_channels=emb_size,
            kernel_size=1 ## apparently this makes it pointwise?
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Rework dimensions (make L, B, D)?
        """
        return


    

class FFNNHead(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.relu = nn.ReLU()
        self.up_project = nn.Linear(embedding_size, hidden_size)
        self.down_project = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        """
        Depending on the task, we use one of two different transition functions: either a separable
        convolution (Chollet, 2016) or a fully-connected neural network that consists of a single rectified-linear
        activation function between two affine transformations, applied position-wise, i.e. individually to
        each row of At.
        """

        x = self.up_project(x)

        x = self.relu(x)

        x = self.down_project(x)

        return self.relu(x)
