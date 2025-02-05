import os
from tokenizers import (decoders,
                        models,
                        pre_tokenizers,
                        processors,
                        trainers,
                        Tokenizer)
from transformers import PreTrainedTokenizerFast
import pandas as pd
import torch
from tqdm.auto import tqdm


# Function to build tokenizer
def get_tokenizer(dataset:pd.DataFrame,
                  path:str='tokenizer.json') -> Tokenizer:
    if os.path.exists(path):
        tokenizer = Tokenizer.from_file(path)
        return tokenizer


    tokenizer = Tokenizer(models.BPE()) # To represent Unkown words
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # Words are seperated by spaces
    
    ''' Start and end of a sentence should be defined and also the minimum appearance
    of a word should be 2.
    '''

    trainer = trainers.BpeTrainer(special_tokens=["<|start_of_text|>",
                                                  "<|end_of_text|>",
                                                  "<|pad|>"])

    # Training the tokenizer
    tokenizer.train_from_iterator(get_all_sentences(dataset),
                                  trainer=trainer)

    # Changing the decoder
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Saving the tokenizer
    tokenizer.save(path)
    
    return tokenizer


def fast_tokenizer(tokenizer:Tokenizer,
                   MaxSeqLen:int) -> PreTrainedTokenizerFast:

    return PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                   pad_token='<|pad|>',
                                   bos_token='<|start_of_text|>', # Begining of sentence
                                   eos_token='<|end_of_text|>',
                                   padding_side='right',
                                   model_max_length=MaxSeqLen)


def get_all_sentences(dataset: pd.DataFrame):
    
    for item in dataset['caption'].tolist():
        yield item



# Convert text to id class
class texttoid:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerFast,
                 dataset: pd.DataFrame):

        self.dataset = dataset
        self.tokenizer = tokenizer # Tokenized Tensor
        self.padToken = tokenizer.convert_tokens_to_ids('<|pad|>')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> dict:

        row = "<|start_of_text|>" + self.dataset['caption'][index] + "<|end_of_text|>"
        DecoderInput = self.tokenizer(text=row,
                                      padding='max_length',
                                      return_tensors='pt') # Tokenized sentence
        DecoderInput = DecoderInput['input_ids'][0]

        # Label should 1 value ahead of input
        Label = torch.cat([
            DecoderInput[1:],
            torch.tensor([self.padToken])
            ])
        
        return{
                "decoder_input": DecoderInput,
                "label": Label
                }

