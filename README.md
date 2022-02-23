# Pet Breed Classification

This is a pet breed classification project using PyTorch ResNet. 

The dataset Oxford [cats and dogs](https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset) dataset on Kaggle

I have used images of cats and dogs to be classified into 37 pet categories:

```
Breeds = {
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
```

Capitalized breed names are cats and the rest are dogs.

Command to train:

`python train.py`

Command to evaluate:

`python evaluate.py`


