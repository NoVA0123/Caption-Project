from torch import nn
import torch


class ffn(nn.Module):
    def __init__(self, config):
        super(ffn, self).__init__()
        '''
        In Attention is all you need(https://arxiv.org/abs/1706.03762) paper,
        size of the last dimension(Dmodel) is quadrupled using a linear
        projection and in GPT-2 GELU is used as an activation function, and
        lastly on more linear projection is applied to return the output to its
        original size.
        '''
        self.linear1 = nn.Linear(config.nEmbd,
                                 4 * config.nEmbd)
        self.gelu = nn.GELU(approximate='tanh')
        self.proj = nn.Linear(config.nEmbd * 4,
                              config.nEmbd)
        self.proj.TRANSFORMER_SCALE_INIT = 1

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.proj(x)
        return x
