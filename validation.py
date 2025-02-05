import torch
from torch.cuda import temperature
from torchvision.transforms import v2
from torchvision.io import read_image
import torch.nn.functional as F
import warnings


# Setting the seed
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


def validation(ModelName:str,
               ImgPath:str,
               tokenizer,
               model,
               TokenSize):

    Temprature = 0.64
    Topk = 100
    device = 'cpu'

    # Use GPU if it is available
    if torch.cuda.is_available():
        device = 'cuda'

    # Use MPS if it is available(Apple devices only)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        

    # Filtering the warnings
    warnings.filterwarnings('ignore')

    
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

    '''Creating caption for Image'''
    model.eval()
    # NumReturnSequences = 4
    CurrentTok = tokenizer.convert_tokens_to_ids('<|start_of_text|>')
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
    Decoded = tokenizer.decode(XGen[0], skip_special_tokens=True)
    print(f"Caption: {Decoded}\n")
    return Decoded
