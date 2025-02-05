import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: %(message)s:')


# List of files
list_of_files = [
        'src/__init__.py',
        'base_trainer/Transformer/self_attention.py',
        'base_trainer/Transformer/multi_head_attention.py',
        'base_trainer/Transformer/decoder.py',
        'base_trainer/Transformer/embedding.py',
        'base_trainer/Transformer/encoder.py',
        'base_trainer/Transformer/feed_forward.py',
        'base_trainer/Transformer/layer_normalization.py',
        'base_trainer/Transformer/positional_encoding',
        'base_trainer/Transformer/transformer.py',
        'base_trainer/Tokenizer/tokenizer.py',
        'base_trainer/vision_model/vision_model.py'
        'requirements.txt',
        ]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating directory; {filedir} for the file {filename}')


    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)) == 0:
        with open(filepath, 'w') as f:
            pass
        logging.info(f'Creating empty file: {filepath}')

    else:
        logging.info(f'{filename} is already created')
