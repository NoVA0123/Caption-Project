import torch
from torch import nn
from torch.nn import functional as F
import math


class cmha(nn.Module):
    def __init__(self, config):
        assert config.nEmbd % config.nHead == 0
        super(cmha, self).__init__()
        # Query, key and value in a batch
#        self.qkvLayer = nn.Linear(config.nEmbd,
                                  #3 * config.nEmbd)
        self.qkLayer = nn.Linear(config.nEmbd,
                                  2 * config.nEmbd)
        self.vLayer = nn.Linear(config.nEmbd,
                                  config.nEmbd)
        # Output projection
        self.proj = nn.Linear(config.nEmbd,
                              config.nEmbd)
        self.proj.TRANSFORMER_SCALE_INIT = 1
        # Regularization
        self.nHead = config.nHead
        self.nEmbd = config.nEmbd
        # Casual mask
        self.register_buffer('bias', torch.tril(
            torch.ones(config.blockSize, config.blockSize)
            ).view(1, 1, config.blockSize, config.blockSize))


    def forward(self, x, CnnImg):
        BatchSize, SeqLen, DModel = x.size()
        # Creating query, key and value matrix
        qk = self.qkLayer(CnnImg)
        #qkv = self.qkvLayer(x)
        v = self.vLayer(x)
        # Splitting the projected matrix
        qk = qk.repeat(1, SeqLen, 1)
        q, k = qk.split(self.nEmbd, dim=2)

        # Changing the dimensions of the matrix for multi head attention
        q = q.view(BatchSize,
                   SeqLen,
                   DModel // self.nHead,
                   self.nHead).transpose(1, 2)
        k = k.view(BatchSize,
                   SeqLen,
                   DModel // self.nHead,
                   self.nHead).transpose(1, 2)
        v = v.view(BatchSize,
                   SeqLen,
                   DModel // self.nHead,
                   self.nHead).transpose(1, 2)

        # Applying attention
        '''
        We need to find the square root of the size of last dimension of key
        vector. This is done, in order to reduce the variance between each
        scalar values. After that we multiply Query and key vecotr and divide
        it by the square root value, we found before and then we fill the mask
        and at last we will find the softmax value.

        This is the attention used in 'Attention is all you need paper', but we
        are going to use optimized variant of it called flash attention, it is
        an inbuilt function in pytorch.
        
        Att = (q @ k.transpose(-2, -1)) * (1. / math.sqrt(k.size(-1)))
        Att = Att.masked_fill(self.bias[:, :, :SeqLen, :SeqLen] == 0,
                              float('-inf'))
        Att = F.softmax(Att, dim=-1)
        # Matrix Multiplication with Value vector
        y = Att @ v'''
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Re - assemble the matrix to its original shape
        x = x.transpose(1, 2).contiguous().view(BatchSize, SeqLen, DModel)
        # Output projection
        x = self.proj(x)
        return x
