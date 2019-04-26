import PIL
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from progressbar import *
from densenet import densenet161, densenet121
from torchvision import transforms

import torch.nn.functional as F

model_map = {
    'densenet121':densenet121,
    'densenet161':densenet161
}



class ImageSet(Dataset):
    def __init__(self,df, transformer):
        self.df = df
        # self.id = id
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df.iloc[index]['image_path']
        image = self.transformer(Image.open(image_path))
        label_idx = self.df.iloc[index]['label_idx']
        sample = {
            # 'dataset_idx':item,
            'image':image,
            'label_idx':label_idx,
            'filename':os.path.basename(image_path)
        }

        return sample




class DefenseImageSet(Dataset):
    def __init__(self,df, transformer):
        self.df = df
        # self.id = id
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df.iloc[index]['image_path']
        image = self.transformer(Image.open(image_path))
        # label_idx = self.df.iloc[index]['label_idx']
        sample = {
            # 'dataset_idx':item,
            'image':image,
            # 'label_idx':label_idx,
            'filename':os.path.basename(image_path)
        }

        return sample



def load_image_data_for_cnn_training(dataset_dir, img_size, batch_size = 16):
    all_imgs = glob.glob('/home/zhuxudong/competition/ijcai2019/IJCAI_AAAC_2019_processed/*/*.jpg')
    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]

    train = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    #
    # train_data, val_data, train_id, val_id = train_test_split(train['image_path'], train['label_idx'],
    #                                                           stratify=train['label_idx'].values, train_size=0.9,
    #                                                           test_size=0.1)

    train_data, val_data = train_test_split(train,
                                            stratify=train['label_idx'].values, train_size=0.9, test_size=0.1)

    transformer_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size, (0.7, 1), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    transformer = transforms.Compose([
        transforms.Resize([img_size, img_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    data_train_transforms=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    data_val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    datasets = {
        'train_data': ImageSet(train_data, data_train_transforms),
        'val_data': ImageSet(val_data, data_val_transforms)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=True) for ds in datasets.keys()
    }
    return dataloaders





def load_data_for_defense(input_dir, img_size, batch_size=32):

    all_img_paths = glob.glob(os.path.join(input_dir, '*.png'))
    # all_labels = [-1 for i in range(len(all_img_paths))]
    dev_data = pd.DataFrame({'image_path':all_img_paths})

    transformer = transforms.Compose([
        transforms.Resize([img_size, img_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    datasets = {
        'dev_data': DefenseImageSet(dev_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=0,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders
