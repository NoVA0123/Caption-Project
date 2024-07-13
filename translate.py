from config import get_config
from model import build_transformer
from tokenizers import Tokenizer
import torch
from base_trainer.vision_model.vision_model import vision_model
from argparse import ArgumentParser
from torchvision.transforms import v2
from torchvision.io import read_image
import warnings


def translate(ImgPath: str,
              TokenizerPath: str,
              ModelPath:str,
              CnnModelPath=None,
              SpecifiedPath=None):

    # Storing device identity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    # Initializing model
    print('\n\nInitializing the model :-')
    MaxLen = 75
    effnetv2s = vision_model(MaxLen, CnnModelPath, SpecifiedPath)
    tokenizer = Tokenizer.from_file(TokenizerPath)
    model = build_transformer(effnetv2s, tokenizer.get_vocab_size(), MaxLen).to(device)
    print('\nModel has been initialized.')

    # Loading the model
    print('\n\n Loading the Model :-')
    ModelFilename = ModelPath
    if torch.cuda.is_available():
        state = torch.load(ModelFilename)
    else:
        state = torch.load(ModelFilename, map_location=torch.device('cpu'))
    model.load_state_dict(state['model_state_dict'])
    print("\nModel has been loaded.")


    # translate
    print('\n\nTranslating...')
    model.eval()
    with torch.no_grad():

        # Read the image
        transform = v2.Compose([
            v2.ToDtype(torch.float, scale=True), # turns images into float16
            v2.Resize(size=[224, 224]), # Resizes the image
            v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1]), # Normalize values
            v2.ToDtype(torch.float)
            ])

        print('\n Reading the Image')
        Img = read_image(ImgPath)
        Img = transform(Img)
        Img = Img.unsqueeze(0)

        # Encoding the image
        EncoderOutput = model.encode(Img, MaxLen, 1).to(device)

        # Initializing the decoder
        DecoderInput = torch.tensor([[tokenizer.token_to_id('[SOS]')]],
                                    dtype=torch.long).to(device)

        # Generating Caption
        print('\n Generating captions')
        NewLen = MaxLen - 1
        while DecoderInput.size(1) < MaxLen:
            Output = model.decode(DecoderInput, EncoderOutput).to(device)

            # Project next toekn
            prob = model.projection(Output[:, -1])
            _, NextWord = torch.max(prob, dim=1)
            DecoderInput = torch.cat(
                    [
                        DecoderInput,
                        torch.tensor([[NextWord.item()]]).to(device)
                        ],
                    dim=1
                    ).to(device)

            # Translate the sentence
            print(f"{tokenizer.decode([NextWord.item()])}", end=' ')


            # Break if we predicted the sentece
            if NextWord == tokenizer.token_to_id("[EOS]"):
                break

            NewLen -= 1


    # Convert ids to tokens
    return tokenizer.decode(DecoderInput[0].tolist())


def get_command_line_arguments():
    parser = ArgumentParser()
    # Specifies required paths
    parser.add_argument("--paths",
                        dest='Paths',
                        required=True,
                        action='append',
                        help='Specify the paths')
    return parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = get_command_line_arguments()
    args = args.Paths
    print(translate(*args))