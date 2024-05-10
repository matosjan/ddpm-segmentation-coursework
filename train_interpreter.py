import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import os
import gc
import wandb
import numpy as np
import cv2
import albumentations as A
from torchvision import transforms
from PIL import Image
import time

from torch.utils.data import DataLoader
from torchvision.utils import save_image

import argparse
from src.utils import setup_seed, multi_acc
from src.pixel_classifier import  load_ensemble_convs, load_ensemble, compute_iou, predict_labels_conv, predict_labels, save_predictions, save_predictions, pixel_classifier
from src.datasets import AugmentingDataset, ImageLabelDataset, FeatureDataset, make_transform
from src.feature_extractors import create_feature_extractor, collect_features

from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.guided_diffusion.dist_util import dev


def prepare_data(args):
    feature_extractor = create_feature_extractor(**args)
    
    print(f"Preparing the train set for {args['category']}...")
    dataset = ImageLabelDataset(
        data_dir=args['training_path'],
        resolution=args['image_size'],
        num_images=args['training_number'],
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )
    X = torch.zeros((len(dataset), *args['dim'][::-1]), dtype=torch.float)
    y = torch.zeros((len(dataset), *args['dim'][:-1]), dtype=torch.uint8)

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 

    for row, (img, label, _) in enumerate(tqdm(dataset)):
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        X[row] = collect_features(args, features).cpu()
        
        for target in range(args['number_class']):
            if target == args['ignore_label']: continue
            if 0 < (label == target).sum() < 20:
                print(f'Delete small annotation from image {dataset.image_paths[row]} | label {target}')
                label[label == target] = args['ignore_label']
        y[row] = label
    
    d = X.shape[1]
    print(f'Total dimension {d}')
    X = X.permute(1,0,2,3).reshape(d, -1).permute(1, 0)
    y = y.flatten()
    return X[y != args['ignore_label']], y[y != args['ignore_label']]

def alt_prepare_data(args):
    feature_extractor = create_feature_extractor(**args)
    
    print(f"Preparing the train set for {args['category']}...")

    if args['augmentation_scale'] > 1:
        dataset = AugmentingDataset(
            data_dir=args['training_path'],
            resolution=args['image_size'],
            num_images=args['training_number'],
            img_mask_transform = A.Compose([
                A.Resize(height=args['image_size'], width=args['image_size']),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2, p=0.4),
                A.ToGray(p=0.4),
                A.HorizontalFlip(p=0.4),
                A.CoarseDropout(max_holes=4, max_height=60, max_width=60, min_holes=2, min_height=40,
                                min_width=40, fill_value=0,
                                mask_fill_value=args['ignore_label'], always_apply=False, p=0.4),
                A.RandomResizedCrop(256, 256, scale=(0.3, 0.6), p=0.4),
                A.Rotate(border_mode=cv2.BORDER_CONSTANT, mask_value=args['ignore_label'], p=0.4),
                A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, mask_value=args['ignore_label'], p=0.4)
            ]),
        )

        extender_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

        for epoch in range(args['augmentation_scale'] - 1):
            i = 0
            for img, label, path in extender_loader:
                filename = path[0].split('.')[0]
                new_path = filename + '_' + str(epoch) + '_' + str(i)
                img = img.squeeze().permute(2, 0, 1).float() / 255

                save_image(img, new_path + ".png")
                np.save(new_path + '.npy', label.squeeze().numpy())
    
    dataset = ImageLabelDataset(
        data_dir=args['training_path'],
        resolution=args['image_size'],
        num_images=args['training_number'] * args['augmentation_scale'],
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )

    print(f'len dataset: {len(dataset)}')

    X = torch.zeros((len(dataset), *args['dim'][::-1]), dtype=torch.float)
    y = torch.zeros((len(dataset), *args['dim'][:-1]), dtype=torch.uint8)
    print(f'shapes:{X.shape}, {y.shape}')

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 

    for row, (img, label, _) in enumerate(tqdm(dataset)):
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        X[row] = collect_features(args, features).cpu()
        
        for target in range(args['number_class']):
            if target == args['ignore_label']: continue
            if 0 < (label == target).sum() < 20:
                print(f'Delete small annotation from image {dataset.image_paths[row]} | label {target}')
                label[label == target] = args['ignore_label']
        y[row] = label
    
    d = X.shape[1]
    print(f'Total dimension {d}')
    X = X.permute(1,0,2,3).reshape(d, -1).permute(1, 0)
    y = y.flatten()
    return X[y != args['ignore_label']], y[y != args['ignore_label']]


def evaluation(args, models):
    feature_extractor = create_feature_extractor(**args)
    dataset = ImageLabelDataset(
        data_dir=args['testing_path'],
        resolution=args['image_size'],
        num_images=args['testing_number'],
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None 

    preds, gts, uncertainty_scores = [], [], []
    for img, label, _ in tqdm(dataset):        
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        features = collect_features(args, features)

        x = features.view(args['dim'][-1], -1).permute(1, 0)
        pred, uncertainty_score = predict_labels(
            models, x, size=args['dim'][:-1]
        )

        pred = pred.numpy()
        label = label.numpy()

        if args['postprocessing'] is True:
            pred = pred.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
            pred = cv2.erode(pred, kernel, iterations=1) 
            pred = cv2.dilate(pred, kernel, iterations=1)

        gts.append(label)
        preds.append(pred)
        uncertainty_scores.append(uncertainty_score.item())
    
    save_predictions(args, dataset.image_paths, preds)
    miou = compute_iou(args, preds, gts, dataset.image_paths)
    with open(os.path.join(args['exp_dir'], f'predictions', 'miou.txt'), "a") as file:
        file.write(f'Overall mIoU: {miou}')
    wandb.log({'mIoU': miou})
    print(f'Overall mIoU: ', miou)
    print(f'Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}')

def test_time_aug_eval(args, models, epoch=''):
    feature_extractor = create_feature_extractor(**args)
    dataset = ImageLabelDataset(
        data_dir=args['testing_path'],
        resolution=args['image_size'],
        num_images=args['testing_number'],
        transform=make_transform(
            args['model_type'],
            args['image_size']
        )
    )

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None

    base_transforms = make_transform(args['model_type'], args['image_size'])    
    preds, gts, uncertainty_scores = [], [], []
    for img, label, path in tqdm(dataset):
        all_masks = []
        ### Real image
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        features = collect_features(args, features)

        x = features.view(args['dim'][-1], -1).permute(1, 0)
        pred, uncertainty_score = predict_labels(
            models, x, size=args['dim'][:-1]
        )
        all_masks.append(pred.numpy())
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ####### AUGMENTATIONS
        #### FIRST
        no_change_transforms1 = A.ReplayCompose([
            A.ToGray(p=1),
            A.CoarseDropout(max_holes=4, max_height=60, max_width=60, min_holes=2, min_height=40,
                                min_width=40, fill_value=0,
                                mask_fill_value=args['ignore_label'], always_apply=False, p=1),
        ])
        transformed_img = no_change_transforms1(image=img)['image']

        pil_img = Image.fromarray(transformed_img)
        aug_img = base_transforms(pil_img)[None].to(dev())
  
        features = feature_extractor(aug_img, noise=noise)
        features = collect_features(args, features)

        x = features.view(args['dim'][-1], -1).permute(1, 0)
        pred, uncertainty_score = predict_labels(
            models, x, size=args['dim'][:-1]
        )

        all_masks.append(pred.numpy())
        ####### SECOND
        change_transforms = A.ReplayCompose([
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2, p=1),
            A.HorizontalFlip(p=1)
        ])

        transformed_data = change_transforms(image=img)

        pil_img = Image.fromarray(transformed_data['image'])
        aug_img = base_transforms(pil_img)[None].to(dev())

        features = feature_extractor(aug_img, noise=noise)
        features = collect_features(args, features)

        x = features.view(args['dim'][-1], -1).permute(1, 0)
        pred, uncertainty_score = predict_labels(
            models, x, size=args['dim'][:-1]
        )

        pred = pred.numpy().squeeze().astype(np.uint8)
        label_transformed = A.ReplayCompose.replay(transformed_data['replay'], image=img, mask=pred)['mask']
        all_masks.append(label_transformed)
        #### THIRD
        change_transforms = A.ReplayCompose([
            A.CoarseDropout(max_holes=4, max_height=60, max_width=60, min_holes=2, min_height=40,
                                min_width=40, fill_value=0,
                                mask_fill_value=args['ignore_label'], always_apply=False, p=1),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2, p=1),
        ])

        transformed_data = change_transforms(image=img)

        pil_img = Image.fromarray(transformed_data['image'])
        aug_img = base_transforms(pil_img)[None].to(dev())

        features = feature_extractor(aug_img, noise=noise)
        features = collect_features(args, features)

        x = features.view(args['dim'][-1], -1).permute(1, 0)
        pred, uncertainty_score = predict_labels(
            models, x, size=args['dim'][:-1]
        )

        pred = pred.numpy().squeeze().astype(np.uint8)
        label_transformed = A.ReplayCompose.replay(transformed_data['replay'], image=img, mask=pred)['mask']
        all_masks.append(label_transformed)
        #### FOURTH
        change_transforms = A.ReplayCompose([
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2, p=1),
            A.HorizontalFlip(p=1),
            A.ChannelShuffle(p=1)
        ])

        transformed_data = change_transforms(image=img)

        pil_img = Image.fromarray(transformed_data['image'])
        aug_img = base_transforms(pil_img)[None].to(dev())

        features = feature_extractor(aug_img, noise=noise)
        features = collect_features(args, features)

        x = features.view(args['dim'][-1], -1).permute(1, 0)
        pred, uncertainty_score = predict_labels(
            models, x, size=args['dim'][:-1]
        )

        pred = pred.numpy().squeeze().astype(np.uint8)
        label_transformed = A.ReplayCompose.replay(transformed_data['replay'], image=img, mask=pred)['mask']
        all_masks.append(label_transformed)        

        ##### VOTING
        pred = torch.zeros((args['image_size'], args['image_size']))

        for row in range(args['image_size']):
            for col in range(args['image_size']):
                counts = dict()
                for mask in all_masks:
                    if mask[row, col] != args['ignore_label']:
                        counts[mask[row, col]] = counts.get(mask[row, col], 0) + 1
                pred[row, col] = max(counts, key=lambda key: counts[key])

        pred = pred.numpy().astype(np.uint8)
        label = label.numpy()

        gts.append(label)
        preds.append(pred)
        uncertainty_scores.append(uncertainty_score.item())
    if epoch == 'final':
        save_predictions(args, dataset.image_paths, preds, epoch='_tta_fixed_5mod')
    print(f'epoch: {epoch}')
    miou = compute_iou(args, preds, gts, dataset.image_paths, epoch=str(epoch))
    with open(os.path.join(args['exp_dir'], 'miou_test_tta_fixed_5mod.txt'), "a") as file:
        file.write(f'Overall mIoU on epoch {epoch}: {miou}\n')

    print(miou)
    print(f'Mean uncertainty on epoch {epoch}: {sum(uncertainty_scores) / len(uncertainty_scores)}')

# def conv_train(args):
#     features, labels = alt_prepare_data(args)
#     train_data = FeatureDataset(features, labels)

#     print(f" ********* max_label {args['number_class']} *** ignore_label {args['ignore_label']} ***********")
#     print(f" *********************** Current number data {len(features)} ***********************")

#     og_iter_num = args['training_number'] * args['dim'][0] * args['dim'][1] / args['batch_size']
#     print(f'og_len: {og_iter_num}')
#     train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, drop_last=True, num_workers=3)
#     # inf_loader = InfiniteDataLoader(train_loader)

#     # dl_iter = iter(inf_loader)
#     wandb.login(key='034de88c658e0d6f03ee9fd9a618344c075198d9')
#     wandb.init(
#         project="ddpm-segmentation",
#         name='depthwise_conv'
#     )
#     # wandb.log({'poshlo': 1})
#     print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")
#     for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):
        
#         gc.collect()
#         classifier = pixel_classifier(numpy_class=(args['number_class']), dim=args['dim'][-1])
#         convs = nn.Sequential(
#             nn.Conv2d(args['dim'][-1], args['dim'][-1], (3, 3), dilation=3, padding=3, groups=64),
#             nn.Conv2d(args['dim'][-1], args['dim'][-1], (1, 1)),
#         )
#         convs = convs.cuda()
#         classifier.init_weights()

#         classifier = classifier.cuda()
#         # classifier = classifier.cuda()
#         criterion = nn.CrossEntropyLoss()
#         optimizer = torch.optim.Adam(list(classifier.parameters()) + list(convs.parameters()), lr=0.001) # weight decay
#         # optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=1e-4)
#         classifier.train()
#         convs.train()

#         iteration = 0
#         break_count = 0
#         best_loss = 10000000
#         stop_sign = 0
#         print('alooo', flush=True)
#         for epoch in range(100):
#             for idx, (features, label) in enumerate(train_loader):
#                 print('lol', flush=True)
            
#                 # features, label = features.to(dev()), label.to(dev())
#                 # end_time = time.time()
#                 # print(end_time - start_time, flush=True)

#                 start_time = time.time()
#                 new_features = convs(features)
#                 end_time = time.time()
#                 print(end_time - start_time, flush=True)

                
#                 start_time = time.time()
#                 label = label.flatten()
#                 end_time = time.time()
#                 print(f'flat {end_time - start_time}', flush=True)
#                 start_time = time.time()
#                 new_features = new_features.reshape(args['dim'][-1], -1).permute(1, 0)
#                 end_time = time.time()
#                 print(f'reshape {end_time - start_time}', flush=True)
#                 start_time = time.time()
#                 # mask = torch.where(label != args['ignore_label'], 1, 0)
#                 # mask = torch.where(label != args['ignore_label'])[0]
#                 # new_features = new_features[mask]
#                 # label = label[mask]
#                 mask = (label != args['ignore_label'])
#                 new_features = new_features[mask]
#                 label =  label[mask]
#                 end_time = time.time()
#                 print(f'ignore {end_time - start_time}', flush=True)
#                 start_time = time.time()
#                 batch_idx = np.random.randint(len(label), size=args['batch_size'])
#                 end_time = time.time()
#                 print(f'rand {end_time - start_time}', flush=True)
#                 start_time = time.time()
#                 X_batch = new_features[batch_idx]
#                 y_batch = label[batch_idx]
#                 end_time = time.time()
#                 print(f'batch {end_time - start_time}', flush=True)
                

#                 # X_batch, y_batch = X_batch.to(dev()), y_batch.to(dev())
#                 y_batch = y_batch.type(torch.long)
#                 # end_time = time.time()
#                 # print(end_time - start_time, flush=True)
                
#                 start_time = time.time()
#                 optimizer.zero_grad()
#                 y_pred = classifier(X_batch)
#                 loss = criterion(y_pred, y_batch)
#                 acc = multi_acc(y_pred, y_batch)
                
#                 loss.backward()
                
#                 optimizer.step()
#                 end_time = time.time()
#                 print(end_time - start_time, flush=True)
                
#                 iteration += 1
#                 if idx > og_iter_num:
#                     break
#                 if iteration % 1000 == 0:
#                     wandb.log({"model_num": MODEL_NUMBER, "epoch": epoch, "iteration": iteration, "acc": acc, "loss": loss.item()})

#                     print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                
#                 if epoch > 1: # try without
#                     if loss.item() < best_loss:
#                         best_loss = loss.item()
#                         break_count = 0
#                     else:
#                         break_count += 1

#                     if break_count > 50:
#                         stop_sign = 1
#                         print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
#                         break

#             if stop_sign == 1:
#                 break

#             chkpt_path = os.path.join(args['exp_dir'], 
#                                   'model_' + str(MODEL_NUMBER) + '_' + str(epoch) + '.pth')
#             torch.save({'model_state_dict': classifier.state_dict(),
#                         'conv_state_dict': convs.state_dict()}, chkpt_path)

#         model_path = os.path.join(args['exp_dir'], 
#                                   'model_' + str(MODEL_NUMBER) + '.pth')
#         MODEL_NUMBER += 1
#         print('save to:',model_path)
#         torch.save({'model_state_dict': classifier.state_dict(),
#                     'conv_state_dict': convs.state_dict()},
#                    model_path)

class InfiniteDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)  # Reset the data loader
            data = next(self.data_iter)
        return data

# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L434
def train(args):
    features, labels = alt_prepare_data(args)

    train_data = FeatureDataset(features, labels)

    print(f" ********* max_label {args['number_class']} *** ignore_label {args['ignore_label']} ***********")
    print(f" *********************** Current number data {len(features)} ***********************")

    og_iter_num = args['training_number'] * args['dim'][0] * args['dim'][1] / args['batch_size']
    print(f'og_len: {og_iter_num}')
    train_loader = DataLoader(dataset=train_data, batch_size=args['batch_size'], shuffle=True, drop_last=True)

    inf_loader = InfiniteDataLoader(train_loader)

    dl_iter = iter(inf_loader)
    wandb.login(key='034de88c658e0d6f03ee9fd9a618344c075198d9')
    wandb.init(
        project="ddpm-segmentation",
        name='only_augs0.5_11epochs'
    )
    print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")
    for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):
        
        gc.collect()
        classifier = pixel_classifier(numpy_class=(args['number_class']), dim=args['dim'][-1])
        classifier.init_weights()

        classifier = nn.DataParallel(classifier).cuda()
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001) # weight decay
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.001, weight_decay=1e-4)
        classifier.train()

        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        for epoch in range(100):
            for idx, (X_batch, y_batch) in enumerate(dl_iter):
            # for X_batch, y_batch in train_loader:
                st = time.time()
                X_batch, y_batch = X_batch.to(dev()), y_batch.to(dev())
                y_batch = y_batch.type(torch.long)
                # print(idx, X_batch.shape, y_batch.shape)

                optimizer.zero_grad()
                y_pred = classifier(X_batch)
                loss = criterion(y_pred, y_batch)
                acc = multi_acc(y_pred, y_batch)

                loss.backward()
                optimizer.step()
                print(time.time() - st, flush=True)
                iteration += 1
                if idx > og_iter_num:
                    break
                if iteration % 1000 == 0:
                    wandb.log({"model_num": MODEL_NUMBER, "epoch": epoch, "iteration": iteration, "acc": acc, "loss": loss.item()})

                    print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                
                if epoch > 10: # try without
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 50:
                        stop_sign = 1
                        print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                        break

            if stop_sign == 1:
                break

            chkpt_path = os.path.join(args['exp_dir'], 
                                  'model_' + str(MODEL_NUMBER) + '_' + str(epoch) + '.pth')
            torch.save({'model_state_dict': classifier.state_dict()}, chkpt_path)

        model_path = os.path.join(args['exp_dir'], 
                                  'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('save to:',model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)
        
def evaluate_by_epoch(opts):
    for epoch in range(11):
        print(f'Loading pretrained models on epoch {epoch}...')
        models = []
        for i in range(5):
            model_path = os.path.join(opts['exp_dir'], f'model_{i}_{epoch}.pth')
            state_dict = torch.load(model_path)['model_state_dict']
            model = nn.DataParallel(pixel_classifier(opts["number_class"], opts['dim'][-1]))
            model.load_state_dict(state_dict)
            model = model.module.to('cuda')
            models.append(model.eval())
        
        test_time_aug_eval(opts, models, epoch)
    print(f'Loading final pretrained models...')
    models = []
    for i in range(5):
        model_path = os.path.join(opts['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']
        model = nn.DataParallel(pixel_classifier(opts["number_class"], opts['dim'][-1]))
        model.load_state_dict(state_dict)
        model = model.module.to('cuda')
        models.append(model.eval())
    
    test_time_aug_eval(opts, models, 'final')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str)
    parser.add_argument('--seed', type=int,  default=0)

    args = parser.parse_args()
    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]

    # Prepare the experiment folder 
    if len(opts['steps']) > 0:
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        # opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)
        opts['exp_dir'] = os.path.join(opts['exp_dir'], 'only_overfit_11epochs')

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

    # Check whether all models in ensemble are trained 
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'model_{i}.pth')) 
                  for i in range(opts['model_num'])]
              
    if not all(pretrained):
        # train all remaining models
        print('Need training')
        opts['start_model_num'] = sum(pretrained)
        print(torch.cuda.is_available())
        train(opts)
        # conv_train(opts)
        # alt_train(opts)
    print('Loading pretrained models on epoch...')
    models = load_ensemble(opts, device='cuda')
    alt_evaluation(opts, models)
