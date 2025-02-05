import torch
import time
import os
import pandas
from torch.cuda import is_bf16_supported
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
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
from base_files.tokenizer_files.tokenizer import get_tokenizer, texttoid, fast_tokenizer
from base_files.dataset_files.json_extracter import caption_extracter
from base_files.dataset_files.image_extracter import imgextracter
from validation import validation
from llama_architecture import mArgs, precompute_theta_pos_frequencies
from llama_architecture import transformer as llama_transformer


def is_bf16_supported():
    try:
        device = torch.device('cuda')
        x = torch.tensor([1, 2], dtype=torch.bfloat16, device=device)
        return True
    except Exception as e:
        print("bf16 is not supported")
        return False


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


# Function for distributed data for parallel processing
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

    # Ignore warnings
    warnings.filterwarnings('ignore')


    # Setting null to None(for Json)
    null = None
    
    # Loading json
    with open (JsonPath, 'r') as f:
        data = json.load(f)

    FilePath = data['file_path']
    TrainJson = FilePath['json_path']['train_json']
    TrainImgPath = FilePath['image_path']['train_path']
    TestImgPath = FilePath['image_path']['test_image_path']
    
    # Extracting caption and storing corresponding image path
    TrainData = caption_extracter(TrainJson, TrainImgPath)


    '''Initializing Config parameters'''

    # Initializing transformer config 
    TrConf = data['transformer_config']
    TrainModelName = TrConf['model_name']
    MaxLen = TrConf['block_size']
    VocabSize = TrConf['vocab_size']
    NumLayers = TrConf['number_layers']
    NumHeads = TrConf['number_heads']
    DModel = TrConf['d_model']
    ContinueTheWork = TrConf['continue']

    # Sample Size
    TotalSamples = data['dataset_config']['max_sample']

    # Initializing model hyper parameters
    ModelConfig = data['model_config']
    BatchSize = ModelConfig['batch_size']
    Epochs = ModelConfig['epochs']
    ModelDtype = ModelConfig['dtype']
    ModelPath = ModelConfig['existing_path']

    bf16 = False
    fp16 = False
    
    if ModelDtype == "bf16":
        bf16 = True
    elif ModelDtype == "fp16":
        fp16 = True

    # Cnn Model parameters
    CnnConf = data['cnn_model_config']
    ExistingPath = CnnConf['existing_path']
    SpecificDownloadPath = CnnConf['specific_download_path']


    # Creating a tokenizer
    if ContinueTheWork == 0:
        if data['tokenizer_config']['tokenizer_save_path'] is null:
            tokenizer = get_tokenizer(TrainData)
        else:
            tokenizer = get_tokenizer(TrainData,
                                      data['tokenizer_config']['tokenizer_path'])
    else:
        tokenizer = Tokenizer.from_file(
                data['tokenizer_config']['tokenizer_load_path'])


    # Creating fast tokenizer
    WrappedTokenizer = fast_tokenizer(tokenizer=tokenizer,
                                       MaxSeqLen=MaxLen)


    # Changing sample size
    TrainData = TrainData.sample(TotalSamples,
                                 random_state=1337).reset_index(drop=True)


    # Creating config
    if TrainModelName == 'gpt-2':
        config = transformerconfig(blockSize=MaxLen,
                                   vocabSize=VocabSize,
                                   nLayers=NumLayers,
                                   nHead=NumHeads,
                                   nEmbd=DModel)
    elif TrainModelName == 'llama-2':
        config = mArgs(dim=DModel,
                       nLayers=NumLayers,
                       nHeads=NumHeads,
                       MaxSeqLen=MaxLen,
                       VocabSize=VocabSize)
    

    # Downloading the Cnn model
    if ExistingPath is not None and SpecificDownloadPath is not None:
        efficientb0 = get_cnn_model(ExistingPath=ExistingPath,
                                    SpecificDownloadPath=SpecificDownloadPath)

    else:
        efficientb0 = get_cnn_model()


    # Loading caption data into dataloader
    CaptionDataClass = texttoid(WrappedTokenizer,
                                TrainData)


    if DistDataParallel:
        CaptionData = parallel_data_sampler(rank=rank,
                                            WorldSize=world_size,
                                            dataset=CaptionDataClass,
                                            batch_size=BatchSize)

    else:
        CaptionData = DataLoader(CaptionDataClass,
                                 batch_size=BatchSize)


    # Loading Image data into dataloader
    ImgDataClass = imgextracter(dataframe=TrainData)


    if DistDataParallel:
        ImgData = parallel_data_sampler(rank=rank,
                                        WorldSize=world_size,
                                        dataset=ImgDataClass,
                                        batch_size=BatchSize)

    else:
        ImgData = DataLoader(ImgDataClass,
                             batch_size=BatchSize)


    # Initializing the transformer model
    if bf16:
        torch.set_float32_matmul_precision('high')
    if TrainModelName == 'gpt-2':
        model = transformer(config=config,
                            CnnModel=efficientb0)
    elif TrainModelName == 'llama-2':
        model = llama_transformer(config,
                                  CnnModel=efficientb0,
                                  device=device)
    if ContinueTheWork:
        checkpoint = torch.load(ModelPath)
        state_dict = checkpoint['model_state_dict']
        for key in list(state_dict.keys()):
            state_dict[key.replace("module._orig_mod.", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict)

    model.to(device) 

    # To compile model and make model faster
    if model == 'gpt-2' or DModel == 384:
        model = torch.compile(model)


    # Adding grad scaler for mixed precision
    if device_type == 'cuda' and fp16:
        Scaler = torch.cuda.amp.GradScaler(enabled=True)
        if ContinueTheWork:
            Scaler.load_state_dict(checkpoint['scaler'])
        UseScaler = True
    else:
        UseScaler = False


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
    test = ModelConfig['learning_rate']['test']

    if test:
        Epochs = 1

    '''optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=MaxLr,
                                  betas=(0.9, 0.95),
                                  eps=1e-8)'''
    optimizer = raw_model.configure_optimizers(WeightDecay=0.1,
                                           LearningRate=6e-4,
                                           device=device_type)
    if ContinueTheWork:
        # Loading checkpoint
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    # Creating gradient accumulation step to increase batch size
    TotalBatchSize = 2**16
    assert TotalBatchSize % (BatchSize * MaxLen * world_size) == 0, "Make sure the total batch size is divisible by Batch * SeqLen"
    GradAccumSteps = TotalBatchSize // (BatchSize * MaxLen * world_size)
    if rank == 0: # This will prevent displaying text multiple times
        print(f"Total batch size is: {TotalBatchSize} ")
        print(f"-> calculated gradient accumulation steps: {GradAccumSteps}")

    # Tensorboard
    writer = SummaryWriter()

    # Training
    TimeTaken = 0
    if ContinueTheWork:
        GlobalSteps = checkpoint['global_step']
        StartEpochs = checkpoint['epoch']
        EndEpochs = StartEpochs + Epochs
    else:
        GlobalSteps = 0
        StartEpochs = 0
        EndEpochs = StartEpochs + Epochs

    for i in tqdm(range(StartEpochs, EndEpochs)):
        IterImgData = iter(ImgData)
        IterCapData = iter(CaptionData)

        LocalSteps = 0

        TrainRange = len(ImgData)//GradAccumSteps
        if test:
            TrainRange = 4
        for _ in range(TrainRange):
            t0 = time.time() # Storing time of begining of the step

            # Accumulated gradient calculation
            for MicroStep in range(GradAccumSteps):

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
                if bf16:
                    with torch.autocast(device_type=device_type,
                                        dtype=torch.bfloat16):
                        _ , loss = model(DecoderInput, img, Label)
                if fp16:
                    with torch.autocast(device_type=device_type,
                                        dtype=torch.float16):
                        _ , loss = model(DecoderInput, img, Label)
                else:
                    _ , loss = model(DecoderInput, img, Label)


                '''
                To calculate Gradient accumulation for larger batches, we need
                to add loss for each micro batch size and scaled it down during
                each step.
                '''
                loss = loss / GradAccumSteps

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
                    model.require_backward_grad_sync = (MicroStep == GradAccumSteps - 1)

            if UseScaler:
                Scaler.scale(loss).backward()
                Scaler.unscale_(optimizer)

            else:
                loss.backward()

            # Applying norm on gradients to reduce shock of the model
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Decay in learning rate
            lr = get_decay_lr(GlobalSteps,
                              WarmupSteps=WarmupSteps,
                              MaxSteps=MaxSteps,
                              MaxLr=MaxLr,
                              MinLr=MinLr)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if not UseScaler:
                optimizer.step() # Applying a backpropogation step

            else:
                Scaler.step(optimizer)
                Scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Synchronizing GPU and CPU runtime
            torch.cuda.synchronize()

            # Storing output time
            t1 = time.time()
            dt = t1 - t0 

            # Calculating Tokens processed per second
            Lossf = loss.item() * GradAccumSteps
            TokensProcessed = BatchSize * MaxLen * GradAccumSteps * world_size
            TokensPerSec = TokensProcessed / dt

            GlobalSteps += 1
            LocalSteps += 1

            '''if rank == 0 and not UseScaler:
                print(f"Epoch: {i} | Steps: {LocalSteps} | loss: {LossAccum.item(): .2f} | lr: {lr: .5e} |{norm: .2f} | Process time: {dt*1000:.2f}ms | tok/sec: {TokensPerSec:.2f}")

            elif rank == 0:
                print(f"Epoch: {i} | Steps: {LocalSteps} | loss: {LossAccum.item(): .2f} | lr: {lr: .5e} |Process time: {dt*1000:.2f}ms | tok/sec: {TokensPerSec:.2f}")'''
            if rank == 0:
                print(f"Epoch: {i+1} | Steps: {LocalSteps+1} | loss: {Lossf: .2f} | lr: {lr: .5e} | Process time: {dt*1000:.2f}ms | tok/sec: {TokensPerSec:.2f}")

            writer.add_scalar('Training Loss', Lossf, global_step=GlobalSteps)
            writer.add_scalar('Training Time Per Step', dt * 1000, global_step=GlobalSteps)
            TimeTaken += dt*1000
            writer.add_scalar("Training Time", TimeTaken, global_step=GlobalSteps)


            if rank == 0 and (GlobalSteps % 500 == 0 or GlobalSteps == 1):
                with torch.no_grad():
                    cap_text = validation(TrainModelName,
                                          TestImgPath,
                                          WrappedTokenizer,
                                          model,
                                          MaxLen)
    writer.close()
    

    if DistDataParallel and rank == 0 and UseScaler:

        ModelName = 'caption_model.pt'
        torch.save({
            'epoch': Epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': GlobalSteps,
            'scaler': Scaler.state_dict()
            }, ModelName)

    elif UseScaler:

        ModelName = 'caption_model.pt'
        torch.save({
            'epoch': Epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': GlobalSteps,
            'scaler': Scaler.state_dict()
            }, ModelName)

    elif DistDataParallel and rank == 0:

        ModelName = 'caption_model.pt'
        torch.save({
            'epoch': Epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': GlobalSteps
            }, ModelName)
    else: 

        ModelName = 'caption_model.pt'
        torch.save({
            'epoch': Epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': GlobalSteps
            }, ModelName)

    # Destroy all parallel process
    if DistDataParallel:
        destroy_process_group()


# Argument parser
def command_line_argument():
    parser = ArgumentParser()
    parser.add_argument('--path', dest='Path')
    return parser.parse_args()


# Running the model
if __name__ == "__main__":

    JsonPath = command_line_argument()
    world_size = torch.cuda.device_count()

    if world_size > 1 and torch.cuda.is_available():
        mp.spawn(train,
                 args=(world_size, JsonPath.Path),
                 nprocs=world_size)

    else:
        rank = 0
        world_size = 1

        train(rank,
              world_size,
              JsonPath.Path)
