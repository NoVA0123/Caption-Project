import torch
from torch import nn
from torchvision import models
import os


def get_cnn_model(ExistingPath=None,
                  SpecificDownloadPath=None):

    # If model needs to be downloaded on specifice path
    if SpecificDownloadPath is not None:
        os.environ['TORCH_HOME'] = SpecificDownloadPath

    # Loading the model
    effnetb0 = models.efficientnet_b0(pretrained=True)

    for param in effnetb0.parameters():
        param.requires_grad = False

    if ExistingPath is not None and os.path.exists(ExistingPath):
        weights = torch.load(ExistingPath)
        effnetb0.load_state_dict(weights)


    return effnetb0
