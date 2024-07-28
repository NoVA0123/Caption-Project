import torch
import time
import os
import pandas as pd
from torch.autograd import backward
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import json
from tqdm.auto import tqdm
from argparse import ArgumentParser
import warnings
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import torch.multiprocessing as mp
from base_files.transformer_files.dataclass import transformerconfig
from base_files.transformer_files.transformer import transformer
from base_files.cnn_model_files.cnn_model import get_cnn_model
from base_files.tokenizer_files.tokenizer import get_tokenizer, texttoid
from base_files.dataset_files.json_extracter import caption_extracter
from base_files.dataset_files.image_extracter import imgextracter



def setup(rank:int,
          world_size:int):

    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ['MASTER_PORT'] = '5674'

    init_process_group(backend='nccl',
                       rank=rank,
                       world_size=world_size)


# Setting seed for reproducability
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


# Creating a function to reduce learning rate using decay method
def get_decay_lr(it:int,
                 WarmupSteps:int,
                 MaxSteps:int,
                 MaxLr:float,
                 MinLr:float):
    # Linear decay for warmup steps
    if it < WarmupSteps:
        return MaxLr * (it + 1) / WarmupSteps
    
    # Constant learning rate
    if it > MaxSteps:
        return MinLr

    # In between we will apply cosine function
    DecayRatio = (it - WarmupSteps) / (MaxSteps - WarmupSteps)
    assert 0 <= DecayRatio <= 1
    Coeff = 0.5 * (1.0 + math.cos(math.pi * DecayRatio))
    return MinLr + Coeff * (MaxLr - MinLr)


# Function for distributed data
def parallel_data_sampler(rank,
                          WorldSize,
                          dataset,
                          batch_size:int):

    sampler = DistributedSampler(dataset,
                                 num_replicas=WorldSize,
                                 rank=rank,
                                 shuffle=False)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            shuffle=False)

    return dataloader

# Training the dataset
def train(rank:int,
          world_size:int,
          JsonPath:str,
          ):
    
    # Check for multiple GPU's
    if world_size > 1:
        setup(rank=rank,# Current GPU id
              world_size=world_size) # Total number of GPU's
        device = rank
        device_type = 'cuda'
        DistDataParallel = True

    else:
        device = 'cpu'

        # Use GPU if it is available
        if torch.cuda.is_available():
            device = 'cuda'

        # Use MPS if it is available(Apple devices only)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        
        device_type = device
        DistDataParallel = False

    null = None
    
    # Loading json
    with open (JsonPath, 'r') as f:
        data = json.load(f)

    FilePath = data['file_path']
    TrainJson = FilePath['json_path']['train_json']
    TrainImgPath = FilePath['image_path']['train_path']
    
    # Extracting caption and storing corresponding image path
    TrainData = caption_extracter(TrainJson, TrainImgPath)

    # Creating a tokenizer
    if data['tokenizer_config']['tokenizer_path'] is null:
        tokenizer = get_tokenizer(TrainData)
    else:
        tokenizer = get_tokenizer(TrainData,
                                  data['tokenizer_config']['tokenizer_path'])


    # Changing sample size
    TotalSamples = data['dataset_config']['max_sample']
    TrainData = TrainData.sample(TotalSamples,
                                 random_state=1337).reset_index(drop=True)
    
    '''Initializing Config parameters'''

    # Initializing transformer config 
    TrConf = data['transformer_config']
    MaxLen = TrConf['block_size']
    VocabSize = TrConf['vocab_size']
    NumLayers = TrConf['number_layers']
    NumHeads = TrConf['number_heads']
    DModel = TrConf['d_model']

    config = transformerconfig(blockSize=MaxLen,
                               vocabSize=VocabSize,
                               nLayers=NumLayers,
                               nHead=NumHeads,
                               nEmbd=DModel)

    # Initializing model hyper parameters
    ModelConfig = data['model_config']
    BatchSize = ModelConfig['batch_size']
    Epochs = ModelConfig['epochs']
    
    # Downloading the Cnn model
    CnnConf = data['cnn_model_config']
    ExistingPath = CnnConf['existing_path']
    SpecificDownloadPath = CnnConf['specific_download_path']
    if ExistingPath is not None and SpecificDownloadPath is not None:
        effnetv2s = get_cnn_model(MaxSeqLen=MaxLen,
                                  DModel=DModel,
                                  ExistingPath=ExistingPath,
                                  SpecificDownloadPath=SpecificDownloadPath)
    else:
        effnetv2s = get_cnn_model(MaxSeqLen=MaxLen,
                                  DModel=DModel)


    # Loading caption data into dataloader
    CaptionDataClass = texttoid(tokenizer=tokenizer,
                          MaxSeqLen=MaxLen,
                          dataframe=TrainData)

    CaptionData = parallel_data_sampler(rank=DDPRank,
                                        WorldSize=DDPWorldSize,
                                        dataset=CaptionDataClass,
                                        batch_size=BatchSize)

    # Loading Image data into dataloader
    ImgDataClass = imgextracter(dataframe=TrainData)

    ImgData = parallel_data_sampler(rank=DDPRank,
                                    WorldSize=DDPWorldSize,
                                    dataset=ImgDataClass,
                                    batch_size=BatchSize)

    # Initializing the transformer model
    model = transformer(config=config,
                        CnnModel=effnetv2s)
    model.to(device) 
    # To compile model and make model faster
    model = torch.compile(model)
    if DistDataParallel:
        '''
        DDP function is neccessary for Distributive computing because, during
        backward pass each gpu has different (due to different parts of
        dataset) gradient. This will create problem for optimizer, to fix this
        issue DDP averages gradient of every rank(GPU) and replaces rank's
        gradient with average. Easy way to understand: DDP synchronizes
        gradients of every GPU.
        '''
        model = DDP(model,
                    device_ids=[device])

    # We need to create raw model for our configure optimizer to work properly
    raw_model = model.module if DistDataParallel else model

    '''Initializing optimizer'''
    # Making a decay learning rate
    MaxLr = ModelConfig['learning_rate']['max_lr']
    MinLr = MaxLr * 0.1
    WarmupSteps = ModelConfig['learning_rate']['warmup_steps']
    MaxSteps = ModelConfig['learning_rate']['max_steps']

    '''optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=MaxLr,
                                  betas=(0.9, 0.95),
                                  eps=1e-8)'''
    optimizer = raw_model.configure_optimizers(WeightDecay=0.1,
                                           LearningRate=6e-4,
                                           device=device)

    # Creating gradient accumulation step to increase batch size
    TotalBatchSize = 2**19
    assert TotalBatchSize % (BatchSize * MaxLen * DDPWorldSize) == 0, "Make sure the total batch size is divisible by Batch * SeqLen"
    GradAccumSteps = TotalBatchSize // (BatchSize * MaxLen * DDPWorldSize)
    if master_process: # This will prevent displaying text multiple times
        print(f"Total batch size is: {TotalBatchSize} ")
        print(f"-> calculated gradient accumulation steps: {GradAccumSteps}")

    # Training
    GlobalSteps = 0
    for i in tqdm(range(Epochs)):
        IterImgData = iter(ImgData)
        IterCapData = iter(CaptionData)

        LocalSteps = 0
        for _ in range(len(ImgData)//GradAccumSteps):
            t0 = time.time() # Storing time of begining of the step
            # Storing values

            optimizer.zero_grad() # Setting optimizer to zero for every step

            # Initializing loss accumalation(details are present in loss calculating code)
            LossAccum = 0.
            # Accumulated gradient calculation
            for MicroSteps in range(GradAccumSteps):

                # Iterating the dataset
                caption = next(IterCapData)
                img = next(IterImgData)
                
                # Storing the values and converting them to device
                DecoderInput = caption['decoder_input'].to(device)
                Label = caption['label'].to(device)
                img = img.to(device)


                '''
                Autocasting to datatypes of model to bfloat16 as it is 4x
                faster than normal float32. It reduces the decimal value.
                '''
                if torch.cuda.is_bf16_supported():
                    with torch.autocast(device_type=device_type,
                                        dtype=torch.bfloat16):
                        _, loss = model(DecoderInput, img, Label)
                else:
                    _, loss = model(DecoderInput, img, Label)


                '''
                To calculate Gradient accumulation for larger batches, we need
                to add loss for each micro batch size and scaled it down during
                each step.
                '''
                loss = loss / GradAccumSteps
                LossAccum += loss.detach() # Keeps on adding gradient
                '''
                Gradient syncing is stopped before last step because it
                increase training process and synchronize every inner loop will
                waste time. We will not synchronize gradients of ranks until
                last step, we will just add them up and reduce them alltogther.


                Reason why we do Gradient Accumulation:-
                Reason for doing gradient accumulation is because larger batch 
                have tendency to smoothen out the convergence while small 
                batches have tendency to converge faster. Large batches are 
                good on large dataset but are costlier to train. To fix this
                gradient accumulation is used on smaller batches to accumulate
                gradient. If gradients are accumulated we can average the loss
                and get the same result as that on larger batch. But not doing
                will not accumulate and will not smooth out the training process,
                i.e. model will not converge(minimum loss) smoothly and will shock
                the model.
                '''
                if DistDataParallel:
                    model.require_backward_grad_sync = (MicroSteps == GradAccumSteps - 1)
                loss.backward()
            
            # Reduce gradients alltogther
            if DistDataParallel:
                dist.all_reduce(LossAccum,
                                op=dist.ReduceOp.AVG)
            # Applying norm on gradients to reduce shock of the model
            norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
            
            # Decay in learning rate
            lr = get_decay_lr(GlobalSteps,
                              WarmupSteps=WarmupSteps,
                              MaxSteps=MaxSteps,
                              MaxLr=MaxLr,
                              MinLr=MinLr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step() # Applying a backpropogation step
            # Synchronizing GPU and CPU runtime
            torch.cuda.synchronize()
            # Storing output time
            t1 = time.time()
            dt = t1 - t0 
            TokensProcessed = BatchSize * MaxLen * GradAccumSteps * DDPWorldSize
            TokensPerSec = TokensProcessed / dt

            GlobalSteps += 1
            LocalSteps += 1
            if master_process:
                print(f"Epoch: {i} | Steps: {LocalSteps} | loss: {LossAccum.item(): .2f} | lr: {lr: .5e} | norm: {norm: .2f} | Process time: {dt*1000:.2f}ms | tok/sec: {TokensPerSec:.2f}")

    # Destroy all parallel process
    if DistDataParallel:
        destroy_process_group()
'''
    ModelName = 'caption_model.pt'
    torch.save({
        'epoch': Epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': GlobalSteps,
        }, ModelName)'''

# Argument parser
def command_line_argument():
    parser = ArgumentParser()
    parser.add_argument('--path', dest='Path')
    return parser.parse_args()


# Running the model
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    JsonPath = command_line_argument()
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(train,
                 args=(world_size, JsonPath.Path),
                 nprocs=world_size)

    else:
        rank = 0
        world_size = 1
        train(rank,
              world_size,
              JsonPath.Path)