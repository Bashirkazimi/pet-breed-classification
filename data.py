import numpy as np
import torch
from torch.utils import data
import pandas as pd
import os
from PIL import Image
from torchvision import transforms


class PetDataset(data.Dataset):
    def __init__(
            self,
            num_classes=37,
            annotation_file='data/myannotations.csv',
            data_root='data/images',
            split='train'
    ):
        self.num_classes = num_classes
        self.annotation_file = annotation_file
        self.data_root = data_root
        self.split = split
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.files = self.read_files()

    def __len__(self):
        return len(self.files)

    def read_files(self):
        files = []
        df = pd.read_csv(self.annotation_file)
        images = df['image'].tolist()
        breed_ids = df['breed_id'].tolist()
        species_list = df['species'].tolist()
        breeds = df['breed'].tolist()

        for image, breed_id, species, breed in zip(images, breed_ids, species_list, breeds):
            img_file = os.path.join(self.data_root, image+'.jpg')
            sample = {
                'img': img_file,
                'breed_id': breed_id,
                'species': species,
                'breed': breed
            }
            files.append(sample)

        return files

    def __getitem__(self, idx):
        item = self.files[idx]
        img_name = item['img']
        breed_id = item['breed_id']
        species = item['species']
        breed = item['breed']

        image = Image.open(img_name).convert('RGB')
        label = breed_id

        image = self.transform(image)

        sample = {'image': image, 'label': label, 'species': species, 'breed': breed}

        return sample