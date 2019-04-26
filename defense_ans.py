from __future__ import print_function
import torch
import torchvision
from torch.autograd import Variable
from torch.nn import DataParallel
from torch import optim
import numpy as np
import time
import os
import glob
from progressbar import *
import argparse
import pandas as pd

from PIL import Image
from scipy.misc import imresize
from scipy.misc import imread


from dataset import load_image_data_for_cnn_training, load_data_for_defense, ImageSet
from densenet import densenet121, densenet161

from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

model_map = {
    'densenet121':densenet121,
    'densenet161':densenet161
}

def defense(model_name, input_dir, output_file, batch_size, weights_path):
    batchshape = [args.batch_size, args.image_height, args.image_width, 3]
    nb_class = 110
    Model = model_map[model_name]
    defense_model = Model(num_classes=110)
    print('loading data for model {}'.format(model_name))
    img_size = defense_model.input_size[0]
    defense_loaders = load_data_for_defense(input_dir, img_size, batch_size)


    # model  = torch.load_stai('./models/1.pth')
    # defense_model.load_state_dict(torch.load('/home/zhuxudong/competition/ijcai2019/pytorch_ijcai/ijcai_defense/tmp/my_ijcai_model.pth'))

    device = torch.device('cuda')
    gpu_ids = [0]
    # defense_model = torch.nn.DataParallel(defense_model, device_ids = gpu_ids)
    defense_model = torch.nn.DataParallel(defense_model)
    torch.backends.cudnn.benchmark = True
    # pth_file = glob.glob(os.path.join(weights_path, 'ep_*.pth'))[0]
    # pth_file = '/home/zhuxudong/competition/ijcai2019/pytorch_ijcai/ijcai_defense/tmp/my_ijcai_model.pth'
    pth_file = os.path.join(weights_path, 'my_ijcai_model.pth')
    print('loading model weights from {}'.format(pth_file))

    state_dict = torch.load(pth_file)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.'+ k  # remove `module.`
        new_state_dict[name] = v
    # load params
    defense_model.load_state_dict(new_state_dict)

    # check_point = torch.load(pth_file)


    # defense_model.load_state_dict(check_point['state_dict'])
    # defense_model.load_state_dict(check_point)
    result = {'filename':[],'predict':[]}
    defense_model.cuda()
    defense_model.eval()

    widgets = ['dev_data: ',Percentage(), ' ',Bar('*'), ' ', Timer(), ' ',ETA(), '' ,FileTransferSpeed()]
    pbar = ProgressBar(widgets)
    for batch_data in pbar(defense_loaders['dev_data']):
        print('defense')
        image = batch_data['image'].to(device)
        filename = batch_data['filename']
        with torch.no_grad():
            logits = defense_model(image)
        y_pred = logits.max(1)[1].detach().cpu().numpy().tolist()
        result['filename'].extend(filename)
        result['predict'].extend(y_pred)
    print('write result to: ', output_file)
    pd.DataFrame(result).to_csv(output_file, header=False, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='',
                        help='input defense data')
    parser.add_argument('--gpu_id', default=2, nargs='+',
                        help='gpu ids to use, e.g. 0 1 2 3', type=int)
    parser.add_argument('--batch_size', default=32,
                        help='batch size, e.g. 16, 32, 64...', type=int)
    parser.add_argument('--output_dir', default='',
                        help='output defense output answer')
    parser.add_argument('--weight_path', default='checkpoints/')
    parser.add_argument('--image_height', default=224,type=int,
                        help='set the imgae height')
    parser.add_argument('--image_width', default=224, type=int,
                        help='set the image width')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('defense begim')
    defense('densenet121', args.input_dir, os.path.join(args.output_dir), args.batch_size,  args.weight_path)
