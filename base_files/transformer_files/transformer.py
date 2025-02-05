import inspect
import torch
from torch import nn
from base_files.transformer_files.decoder import block
from torch.nn import functional as F


class transformer(nn.Module):

    def __init__(self, config, CnnModel):
        super(transformer, self).__init__()
        self.config = config # Transformer config
        '''
        We need to insert modules to the transformer. To do it ModuleDict will
        be used to insert the modules
        '''
        self.transformer = nn.ModuleDict(dict(
            # Token Embeddings at input stage
            tokEmbd = nn.Embedding(config.vocabSize,
                                   config.nEmbd),
            # Positional Embeddings
            posEmbd = nn.Embedding(config.blockSize,
                                   config.nEmbd),
            # Hidden layers or Decoder Blocks
            hid = nn.ModuleList([block(config) for _ in range(config.nLayers)]),
            # Layer normalization is applied at the end of each Decoder output
            layerNorm = nn.LayerNorm(config.nEmbd),
            ))
        self.head = nn.Linear(config.nEmbd,
                              config.vocabSize,
                              bias=False)

        # Cnn Model
        self.cnnModel = CnnModel
        self.cnnLayer = nn.Linear(1000, config.nEmbd)

        # Pointing final Linear projection weights to token embedding weights
        self.transformer.tokEmbd.weight = self.head.weight

        # Initializing weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        '''
        This function is to initialize weights
        '''
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'TRANSFORMER_SCALE_INIT'):
                std *= (2 * self.config.nLayers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0., std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0., std=0.02)

    def configure_optimizers(self,
                             WeightDecay:float,
                             LearningRate:float,
                             device):
        # Find all the parameters that requires gradients
        Params = {NumParams: p for NumParams, p in self.named_parameters()}
        Params = {NumParams: p for NumParams, p in Params.items() if p.requires_grad}

        '''
        Create optimizer group of weight decay and non decay. Tensors with
        higher dimensions require weight decay to reach optimum value and tensors
        with lower dimensions like bias do not require weight decay.
        '''
        DecayParams = [p for _, p in Params.items() if p.dim() >= 2]
        NonDecayParams = [p for _, p in Params.items() if p.dim() < 2]
        OptimGroups = [
                {'params': DecayParams, 'weight_decay': WeightDecay},
                {'params': NonDecayParams, 'weight_decay': 0.0}
                ]

        # To find number of decay and non decay parameters
        NumDecayParams = sum(p.numel() for p in DecayParams)
        NumNonDecayParams = sum(p.numel() for p in NonDecayParams)
        print(f"Number of decaying parameter tensors: {len(DecayParams)}, with {NumDecayParams} parameters")
        print(f"Number of non decaying parameter tensors: {len(NonDecayParams)}, with {NumNonDecayParams} parameters")

        # Check fused is available or not
        if 'cuda' in device:
            FusedAvailable = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            UseFused = FusedAvailable 
        else:
            UseFused = False
        print(f'Using fused AdamW: {UseFused}')
        # Configuring optimizer
        Optimizer = torch.optim.AdamW(OptimGroups,
                                      lr=LearningRate,
                                      betas=(0.9, 0.95),
                                      eps=1e-8,
                                      fused=UseFused)
        return Optimizer

    def forward(self, Input, Img, Label=None):
        # Input is of shape (BatchSize, SeqLen)
        BatchSize, SeqLen = Input.size()
        assert SeqLen <= self.config.blockSize, f"Cannot pass the sequence to the model, Error: length {SeqLen} is greater than the block size parameter for the model"

        # Applying embeddings and tokenization
        # Passing image through Cnn Model
        Img = self.cnnModel(Img)
        Img = self.cnnLayer(Img)
        Img = torch.reshape(Img,
                            (BatchSize, 1, self.config.nEmbd)) 

        Pos = torch.arange(0, SeqLen, dtype=torch.int, device=Input.device)
        PosEmbd = self.transformer.posEmbd(Pos)
        Input = self.transformer.tokEmbd(Input)

        # Adding both the embeddings and CNN output
        Input = PosEmbd + Input #+ Img

        # applying decoder block
        for block in self.transformer.hid:
            Input = block(Input, Img)

        # forward the final layernorm
        Input = self.transformer.layerNorm(Input)


        # Adding Image and input
        Input = Input + Img

        # Classifying
        logits = self.head(Input)
        if Label is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   Label.view(-1), ignore_index=-1)
            return logits, loss

        return logits
