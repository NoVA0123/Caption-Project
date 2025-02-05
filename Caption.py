import torch
import pandas as pd
import json
from argparse import ArgumentParser
import warnings
from tokenizers import Tokenizer
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.io import read_image
from base_files.transformer_files.dataclass import transformerconfig
from base_files.transformer_files.transformer import transformer
from llama_architecture import transformer as llama_transformer
from llama_architecture import mArgs
from base_files.cnn_model_files.cnn_model import get_cnn_model


@torch.no_grad()
def CaptionGenerator(JsonPath:str,
                     ImgPath: str,
                     TokenSize: str,
                     Temprature: str = '1.0',
                     Topk: str = '100',
                     SpecialPath = None):

    TokenSize = int(TokenSize)
    Topk = int(Topk)
    Temprature = float(Temprature)
    device = 'cpu'

    # Use GPU if it is available
    if torch.cuda.is_available():
        device = 'cuda'

    # Use MPS if it is available(Apple devices only)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        

    # Filtering the warnings
    warnings.filterwarnings('ignore')

    null = None

    # Importing json file
    with open (JsonPath, 'r') as f:
        data = json.load(f)

    if SpecialPath is None:
        ModelPath = data['model_config']['existing_path']
    else:
        ModelPath = SpecialPath
    ModelName = data['transformer_config']['model_name']

    # Importing tokenizer
    TokenizerPath = data["tokenizer_config"]['tokenizer_load_path']
    tokenizer = Tokenizer.from_file(TokenizerPath)
    

    # Creating a transform image object
    transform = v2.Compose([
        v2.Resize(size=[489,456], antialias=True),
	    v2.Resize(size=[256,224], antialias=True),
        v2.ToDtype(torch.float, scale=True),
        v2.RandomRotation(degrees=(0,180)),
        v2.CenterCrop(224),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Reading the image and transforming the image
    img = transform(read_image(ImgPath))


    # Initializing transformer config 
    TrConf = data['transformer_config']
    MaxLen = TrConf['block_size']
    VocabSize = TrConf['vocab_size']
    NumLayers = TrConf['number_layers']
    NumHeads = TrConf['number_heads']
    DModel = TrConf['d_model']

    if ModelName == 'gpt-2':
        config = transformerconfig(blockSize=MaxLen,
                                   vocabSize=VocabSize,
                                   nLayers=NumLayers,
                                   nHead=NumHeads,
                                   nEmbd=DModel)
    elif ModelName == 'llama-2':
        config = mArgs(dim=DModel,
                       nLayers=NumLayers,
                       nHeads=NumHeads,
                       MaxSeqLen=MaxLen,
                       VocabSize=VocabSize)


    # Downloading the Cnn model
    CnnConf = data['cnn_model_config']
    ExistingPath = CnnConf['existing_path']
    SpecificDownloadPath = CnnConf['specific_download_path']
    if ExistingPath is not None and SpecificDownloadPath is not None:
        effnetb0 = get_cnn_model(ExistingPath=ExistingPath,
                                  SpecificDownloadPath=SpecificDownloadPath)
    else:
        effnetb0 = get_cnn_model()


    # Initializing the transformer model
    if ModelName == 'llama-2':
        model = llama_transformer(config,
                                  CnnModel=effnetb0,
                                  device=device)
    else:
        model = transformer(config=config,
                            CnnModel=effnetb0)

    # Loading checkpoint
    checkpoint = torch.load(ModelPath)
    state_dict = checkpoint['model_state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace("module._orig_mod.", "")] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)


    '''Creating caption for Image'''
    model.eval()
    # NumReturnSequences = 4
    CurrentTok = tokenizer.token_to_id('<|start_of_text|>')
    XGen = torch.tensor([CurrentTok], dtype=torch.long)
    XGen = XGen.unsqueeze(0)
    XGen = XGen.to(device)

    img = img.unsqueeze(0)#.repeat(NumReturnSequences, 1, 1)
    img = img.to(device)
    SampleRng = torch.Generator(device=device)
    SampleRng.manual_seed(1337)
    if ModelName == 'llama-2':
        values = XGen
    for x in range(TokenSize):

        # forwarding the model
        if ModelName == 'llama-2':
            logits = model(XGen, img, StartPos=x)
        else:
            logits = model(XGen, img)
        # Take the logits at last position
        logits = logits[:, -1, :] / Temprature
        # Topk
        v, _ = torch.topk(logits, min(Topk, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        # Get the probablities
        probs = F.softmax(logits, dim=-1)
        # TopK sampling
        ix = torch.multinomial(probs, num_samples=1, generator=SampleRng) # (B, 1)

        # gather the corresponding indices
        if ModelName == 'llama-2':
            XGen = ix
            values = torch.cat((values, ix), dim=1)
        else:
            XGen = torch.cat((XGen, ix), dim=1)

        if ix[0] == 1:
            break
    if ModelName == 'llama-2':
        XGen = values

    # Print the text which has been generated
    '''DecodedValues = []
    for i in range(NumReturnSequences):

        tokens = XGen[i, :MaxLen].tolist()
        decoded = tokenizer.decode(tokens)
        print(decoded)
        DecodedValues.append(decoded)

    return DecodedValues'''
    XGen = XGen[0].tolist()
    Decoded = tokenizer.decode(XGen)
    print(f"Caption: {Decoded} \n {XGen}")
    return Decoded


# Argument parser
def command_line_argument():
    parser = ArgumentParser()
    parser.add_argument('--jpath', dest='JsonPath', help='Inserts json path inside program')
    parser.add_argument('--ipath', dest='ImgPath', help='Inserts image Path inside program')
    parser.add_argument('--size', dest='Size', help='Manual token size for the model')
    parser.add_argument('--temp', dest='Temprature', help='Adjust the temprature of the model')
    parser.add_argument('--topk', dest='TopK', help='Random tokens will picked from top K tokens')
    parser.add_argument('--mpath', dest='ModelPath', help='Inserts model path inside program')
    return parser.parse_args()

if __name__ == '__main__':
    Args = command_line_argument()
    jpath = Args.JsonPath
    ipath = Args.ImgPath
    mpath = Args.ModelPath
    size = Args.Size
    temp= Args.Temprature
    topk = Args.TopK
    decoded = CaptionGenerator(jpath,
                               ipath,
                               size,
                               temp,
                               topk,
                               mpath)
