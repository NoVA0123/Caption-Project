from torch import nn
from base_files.transformer_files.feed_forward import ffn
from base_files.transformer_files.multi_head_attention import cmha


# Implementing Decoder block

class block(nn.Module):
    def __init__(self, config):
        super(block, self).__init__()
        self.layerNorm1 = nn.LayerNorm(config.nEmbd) # layer normalization
        self.attn = cmha(config) # Not masked self attention
        self.layerNorm2 = nn.LayerNorm(config.nEmbd)
        self.fFN = ffn(config) # feed forward network


    def forward(self, x, CnnImg):
        '''
        Step-1: Input -> LayerNorm -> Casual Attention = Modified input
        Step-2: Input + Modified input = Input
        Step-3: Input -> LayerNorm -> Feed forward network or Multi layer
                                      Perceptron = Modified Input 
        Step-4: Input + Modified input = Decoder output
        '''
        # x = x + self.attn(self.layerNorm1(x))
        x = x + self.attn(self.layerNorm1(x), CnnImg)
        x = x + self.fFN(self.layerNorm2(x))
        return x
