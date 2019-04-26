# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

import sys,os
sys.path.insert(0, '')
import PIL
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.optim as optim
from progressbar import *
from torchvision import transforms
import torch.nn.functional as F

from dataset import load_image_data_for_cnn_training, load_data_for_defense, ImageSet
from densenet import densenet121, densenet161

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

model_map = {
    'densenet121':densenet121,
    'densenet161':densenet161
}


def accuracy_score(y_true, y_pred):
    y_pred = np.concatenate(tuple(y_pred))
    y_true = np.concatenate(tuple([[t for t in y] for y in y_true])).reshape(y_pred.shape)
    return (y_true == y_pred).sum() / float(len(y_true))


def start_train(model_name, model, train_loader, val_loader, device, lr=0.0001, n_ep=40, num_classes=110, save_path='/home/zhuxudong/competition/ijcai2019/pytorch_ijcai/ijcai_defense/tmp'):
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=2)
    best_acc = 0.0
    accuracy = 0.0
    running_accuracy = 0.0
    print('do training')
    # do training
    for i_ep in range(n_ep):
        print('Epoch {}/{}'.format(i_ep, n_ep-1))
        print('_'*50)
        model.train()
        train_losses = []
        widgets = ['train :',Percentage(), ' ', Bar('#'),' ', Timer(), ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets)
        for batch_data in pbar(train_loader):
            image = batch_data['image'].to(device)
            label = batch_data['label_idx'].to(device)
            optimizer.zero_grad()
            logits = model(image)
            loss = F.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            train_losses += [loss.detach().cpu().numpy().reshape(-1)]
        train_losses = np.concatenate(train_losses).reshape(-1).mean()

        model.eval()
        val_losses = []
        preds = []
        true_labels = []
        widgets = ['val:',Percentage(), ' ', Bar('#'),' ', Timer(),
           ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets)
        for batch_data in pbar(val_loader):
            image = batch_data['image'].to(device)
            label = batch_data['label_idx'].to(device)
            with torch.no_grad():
                logits = model(image)
            loss = F.cross_entropy(logits, label).detach().cpu().numpy().reshape(-1)
            val_losses += [loss]
            true_labels += [label.detach().cpu().numpy()]
            preds += [(logits.max(1)[1].detach().cpu().numpy())]
            _, pred = torch.max(logits, 1)

            running_accuracy += torch.sum(pred == label.data)

            # accuracy += torch.sum(preds == image).float()

        preds = np.concatenate(preds, 0).reshape(-1)
        true_labels = np.concatenate(true_labels, 0).reshape(-1)
        # print(preds.shape())
        # print(true_labels.shape())

        # acc = accuracy_score(true_labels, preds)
        epoch_acc = running_accuracy/len(val_loader.dataset)

        val_losses = np.concatenate(val_losses).reshape(-1).mean()
        scheduler.step(val_losses)
        # need python3.6
        print(f'Epoch : {i_ep}  val_acc : {epoch_acc:.5%} ||| train_loss : {train_losses:.5f}  val_loss : {val_losses:.5f}  |||')
        # if epoch_acc >= best_acc:
        #     best_acc = epoch_acc
        # files2remove = glob.glob(os.path.join(save_path,'ep_*'))
        # for _i in files2remove:
        #     os.remove(_i)
        # torch.save(model, os.path.join(save_path, f'/full_ep_{i_ep}_{model_name}_val_acc_{epoch_acc:.4f}.pth'))
        # torch.save(model.cpu().state_dict(), os.path.join(save_path, f'/ep_{i_ep}_{model_name}_val_acc_{epoch_acc:.4f}.pth'))
        save_state_dict_model_path = '/home/zhuxudong/competition/ijcai2019/pytorch_ijcai/ijcai_defense/tmp/state_dict.pth'
        save_model_path = '/home/zhuxudong/competition/ijcai2019/pytorch_ijcai/ijcai_defense/tmp/full_model.pth'
        # torch.save(model.cpu().state_dict(), os.path.join(save_path, '1.dpth'))
        torch.save(model.state_dict(), save_state_dict_model_path)
        torch.save(model, save_model_path)

        model.to(device)



def train_model(model_name, gpu_ids, batch_size):
    Model = model_map[model_name]
    model = Model(num_classes=110)
    print('loading data for training %s' %model_name)
    dataset_dir = '/home/zhuxudong/competition/ijcai2019/IJCAI_AAAC_2019_processed/'
    img_size = model.input_size[0]
    loaders = load_image_data_for_cnn_training(dataset_dir, img_size, batch_size=batch_size*len(gpu_ids))
    save_path = '/home/zhuxudong/comptition/ijcai2019/save/pytorch_weights/%s' %model_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('training with ',len(gpu_ids),"GPUS")
    device = torch.device('cuda')
    model = model.to(device)

    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    print("start training defence model\n ")
    start_train(model_name, model, loaders['train_data'], loaders['val_data'], device='cuda', lr=0.0001, save_path=save_path, n_ep=40, num_classes=110)



def defense(input_dir, target_model, weights_path, defense_type, defense_params, output_file, batch_size):
    Model = model_map[target_model]
    model = Model(num_classes=110)
    #loading data
    print('loading data for model %s'%target_model)
    img_size=model.input_size[0]
    loaders = load_data_for_defense(input_dir, img_size, batch_size)

    device = torch.device('cuda:0')
    model = torch.nn.DataParallel(model)
    pth_file = glob.glob(os.path.join(weights_path,'ep_*.pth'))[0]
    print('loading weights from: ',pth_file)
    model.load_state_dict(torch.load(pth_file))
    #storeing the result
    result = {'filename':[], 'predict':[]}

    #start
    model.eval()
    widgets = ['dev_data: ',Percentage(), ' ', Bar('$'), ' ', Timer(), ' ', ETA(), ' ',FileTransferSpeed()]
    pbar = ProgressBar(widgets)
    for batch_data in pbar(loaders['dev_data']):
        image = batch_data['image'].to(device)
        filename = batch_data['filename']
        with torch.no_grad():
            logits = model(image)
        y_pred = logits.max(1)[1].detach().cpu().numpy().tolist()
        result['filename'].extend(filename)
        result['predict'].extend(y_pred)
    print('write result file to:',output_file)
    pd.DataFrame(result).to_csv(output_file, header=False, index=False)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model', default='densenet121',
                        help='cnn model, e.g. , densenet121, densenet161', type=str)
    parser.add_argument('--gpu_id', default=2, nargs='+',
                        help='gpu ids to use, e.g. 0 1 2 3', type=int)
    parser.add_argument('--batch_size', default=64,
                        help='batch size, e.g. 16, 32, 64...', type=int)
    parser.add_argument('--input_dir', default='/home/zhuxudong/competition/ijcai2019/IJCAI_AAAC_2019_preprocessed/')
    parser.add_argument('--weight_path', default='/home/zhxudong/competition/ijcai2019/support/')
    return parser.parse_args()





if __name__=='__main__':
    args = parse_args()
    gpu_ids = args.gpu_id
    # gpu_ids = 2
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    batch_size = args.batch_size
    target_model = args.target_model
################# Training #######################
    train_model(target_model, gpu_ids, batch_size)
################# Defense #######################
    input_dir = args.input_dir
    weights_path = args.weight_path
    # defense(input_dir, target_model, weights_path, defense_type, defense_params, output_file, batch_size)
    # pass
