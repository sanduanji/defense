import PIL
from PIL import Image
import os
import glob
import pandas as pd
import torchvision
import torch
from torch.utils.data import DataLoader, Dataset
from progressbar import *
from PIL.Image import register_decoder



class ImageSet_preprocess(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        image = Image.open(image_path).convert('RGB')
        # save_img =
        save_image_path = image_path.replace('/IJCAI_2019_AAAC_train/','/IJCAI_AAAC_2019_processed/')
        _save_dir, _save_path = os.path.split(save_image_path)
        image.save(save_image_path)
        return save_image_path



def preprocess(batch_size=64):
    all_images = glob.glob('/home/zhuxudong/competition/ijcai2019/IJCAI_2019_AAAC_train/')
    train_data = pd.DataFrame(all_images)
    datasets = {
        'train_data':ImageSet_preprocess(train_data)
    }

    dataloaders = {
        ds:DataLoader(datasets[ds],
                      batch_size=batch_size,
                      num_workers=8,
                      shuffle=False) for ds in datasets.keys()
    }

    return dataloaders


# if __name__ == '__main__':
#     dataloader = preprocess()
#     widgets = ['jpeg:',Percentage(),Bar('#'), ' ', Timer(), ' ', ETA(), ' ', FileTransferSpeed()]
#     pbar = ProgressBar(widgets = widgets)
#     for batch_data in pbar(dataloader['train_data']):
#         preprocess()
#     pass
#
#






