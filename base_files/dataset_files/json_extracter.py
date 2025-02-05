import json
import pandas as pd
from tqdm.auto import tqdm



def caption_extracter(JsonPath:str,
                      ImgPath:str) -> pd.DataFrame:

    # Loading json
    with open(JsonPath, 'r') as f:
        data = json.load(f)
        data = data['annotations']


    # Creating a list to store data
    ImgCap = []

    # Traversing the Json
    for sample in tqdm(data):
        '''
        Create a 12 digits string contains 0 as the character and add .jpg at
        the end. Start inserting name from the right side of 12 digit string.
        Combine the name and the path of the images and we got the image path to
        corresponding caption.
        '''
        ImgName = '%012d.jpg' % sample['image_id']
        ImgName = f'{ImgPath}{ImgName}'
        ImgCap.append([ImgName, sample['caption']])


    # Creating a DataFrame to store caption and corresponding image address
    captions = pd.DataFrame(ImgCap, columns=['image_path', 'caption'])

    return captions
