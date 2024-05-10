import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import pandas as pd
import cv2

from torch.distributions import Categorical
from src.utils import colorize_mask, oht_to_scalar
from src.data_util import get_palette, get_class_names
from PIL import Image


# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L68
class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(pixel_classifier, self).__init__()
        if numpy_class < 30:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                # nn.Dropout(p=0.2),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32), # ТОДО dropout
                nn.ReLU(),
                # nn.Dropout(p=0.2),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                # nn.Dropout(p=0.2),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                # nn.Dropout(p=0.2),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class)
            )

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        return self.layers(x)

def predict_labels_conv(args, models, features, size):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    mean_seg = None
    all_seg = []
    all_entropy = []
    seg_mode_ensemble = []

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            new_features = models[MODEL_NUMBER][1](features).view(args['dim'][-1], -1).permute(1, 0)
            preds = models[MODEL_NUMBER][0](new_features.cuda())
            entropy = Categorical(logits=preds).entropy()
            all_entropy.append(entropy)
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = softmax_f(preds)
            else:
                mean_seg += softmax_f(preds)

            img_seg = oht_to_scalar(preds)
            img_seg = img_seg.reshape(*size)
            img_seg = img_seg.cpu().detach()

            seg_mode_ensemble.append(img_seg)

        mean_seg = mean_seg / len(all_seg)

        full_entropy = Categorical(mean_seg).entropy()

        js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
        top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        img_seg_final = torch.mode(img_seg_final, 2)[0]
    return img_seg_final, top_k

def predict_labels(models, features, size):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    mean_seg = None
    all_seg = []
    all_entropy = []
    seg_mode_ensemble = []

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            preds = models[MODEL_NUMBER](features.cuda())
            entropy = Categorical(logits=preds).entropy()
            all_entropy.append(entropy)
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = softmax_f(preds)
            else:
                mean_seg += softmax_f(preds)

            img_seg = oht_to_scalar(preds)
            img_seg = img_seg.reshape(*size)
            img_seg = img_seg.cpu().detach()

            seg_mode_ensemble.append(img_seg)

        mean_seg = mean_seg / len(all_seg)

        full_entropy = Categorical(mean_seg).entropy()

        js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
        top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        img_seg_final = torch.mode(img_seg_final, 2)[0]
    return img_seg_final, top_k


def save_predictions(args, image_paths, preds, epoch=''):
    palette = get_palette(args['category'])
    os.makedirs(os.path.join(args['exp_dir'], 'predictions' + epoch), exist_ok=True)
    os.makedirs(os.path.join(args['exp_dir'], 'visualizations' + epoch), exist_ok=True)

    for i, pred in enumerate(preds):
        filename = image_paths[i].split('/')[-1].split('.')[0]
        pred = np.squeeze(pred)

        np.save(os.path.join(args['exp_dir'], 'predictions' + epoch, filename + '.npy'), pred)

        mask = colorize_mask(pred, palette)
        Image.fromarray(mask).save(
            os.path.join(args['exp_dir'], 'visualizations' + epoch, filename + '.jpg')
        )


def compute_iou(args, preds, gts, image_paths, print_per_class_ious=True, epoch=''):
    class_names = get_class_names(args['category'])

    ids = range(args['number_class'])

    unions = Counter()
    intersections = Counter()

    df = pd.DataFrame()
    

    for i, (pred, gt) in enumerate(zip(preds, gts)):
        temp_dict = dict()
        temp_dict['filename'] = image_paths[i].split('/')[-1].split('.')[0]
        class_iou_array = []
        for target_num in ids:
            if target_num == args['ignore_label']: 
                continue

            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            
            class_iou = (preds_tmp & gts_tmp).sum() / (1e-8 + (preds_tmp | gts_tmp).sum())
            temp_dict[class_names[target_num]] = class_iou
            if gts_tmp.sum() != 0:
                class_iou_array.append(class_iou)
            unions[target_num] += (preds_tmp | gts_tmp).sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()
        
        temp_dict['mIoU'] = sum(class_iou_array) / len(class_iou_array)
        df = pd.concat([df, pd.DataFrame(temp_dict, index=[0])], ignore_index = True)
    
    # df.to_csv(os.path.join(args['exp_dir'], f'predictions{epoch}', 'metrics.csv'), encoding='utf-8', index=False)
    ious = []
    for target_num in ids:
        if target_num == args['ignore_label']: 
            continue
        iou = intersections[target_num] / (1e-8 + unions[target_num])
        ious.append(iou)
        if print_per_class_ious:
            # with open(os.path.join(args['exp_dir'], 'miou_test.txt'), "a") as file:
            #     file.write(f"IOU for {class_names[target_num]} {iou:.4}\n")
            # with open(os.path.join(args['exp_dir'], f'predictions{epoch}', 'miou.txt'), "a") as file:
            #     file.write(f"IOU for {class_names[target_num]} {iou:.4}\n")
            print(f"IOU for {class_names[target_num]} {iou:.4}")
    return np.array(ious).mean()


def load_ensemble(args, device='cpu'):
    models = []
    for i in range(args['model_num']):
        model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']
        model = nn.DataParallel(pixel_classifier(args["number_class"], args['dim'][-1]))
        model.load_state_dict(state_dict)
        model = model.module.to(device)
        models.append(model.eval())
    return models

def load_ensemble_convs(args, device='cpu'):
    models = []
    for i in range(args['model_num']):
        model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']
        model = nn.DataParallel(pixel_classifier(args["number_class"], args['dim'][-1]))
        model.load_state_dict(state_dict)
        model = model.module.to(device)

        state_dict = torch.load(model_path)['conv_state_dict']
        convs = nn.Sequential(
            nn.Conv2d(args['dim'][-1], args['dim'][-1], (3, 3), dilation=3, padding=3, groups=8448),
            nn.Conv2d(args['dim'][-1], args['dim'][-1], (1, 1)),
        )
        convs.load_state_dict(state_dict)
        convs = convs.to(device)

        models.append([model.eval(), convs.eval()])
    return models
