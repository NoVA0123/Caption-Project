from dataclasses import dataclass


# Creating a data class for the transformer config

@dataclass
class transformerconfig:
    blockSize: int
    vocabSize: int
    nLayers: int = 6
    nHead: int = 6
    nEmbd: int = 384
