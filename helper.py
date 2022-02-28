import numpy as np
from PIL import Image


class UI:
    about_block = """

    ### About

    This is a pet breed classification app. Given an image of a dog or a cat, a trained AI model predicts what breed it belongs to.

    - [Repo](https://github.com/bashirkazimi/pet-breed-classification)
    - Dataset: [Oxford Cats and Dogs](https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset)
    - Read more about me [here](https://bashirkazimi.github.io/)
    - [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/bashir_kazimi.svg?style=social&label=Follow%20%40bashir_kazimi)](https://twitter.com/bashir_kazimi)
    """

    css = f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }}
    .reportview-container .main {{
        color: "#111";
        background-color: "#eee";
    }}
</style>
"""


headers = {"Content-Type": "application/json"}

classToBreed = {
    0: 'Egyptian_Mau',
    1: 'Persian',
    2: 'Ragdoll',
    3: 'Bombay',
    4: 'Maine_Coon',
    5: 'Siamese',
    6: 'Abyssinian',
    7: 'Sphynx',
    8: 'British_Shorthair',
    9: 'Bengal',
    10: 'Birman',
    11: 'Russian_Blue',
    12: 'great_pyrenees',
    13: 'havanese',
    14: 'wheaten_terrier',
    15: 'german_shorthaired',
    16: 'samoyed',
    17: 'boxer',
    18: 'leonberger',
    19: 'miniature_pinscher',
    20: 'shiba_inu',
    21: 'english_setter',
    22: 'japanese_chin',
    23: 'chihuahua',
    24: 'scottish_terrier',
    25: 'yorkshire_terrier',
    26: 'american_pit_bull_terrier',
    27: 'pug',
    28: 'keeshond',
    29: 'english_cocker_spaniel',
    30: 'staffordshire_bull_terrier',
    31: 'pomeranian',
    32: 'saint_bernard',
    33: 'basset_hound',
    34: 'newfoundland',
    35: 'beagle',
    36: 'american_bulldog'
}