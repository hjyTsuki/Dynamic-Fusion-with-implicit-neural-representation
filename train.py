import json
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data_RGB import get_training_data, get_validation_data
# from methods.vit import ViT as myNet
from methods.net import Net as myNet
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from get_parameter_number import get_parameter_number
import kornia
from torch.utils.tensorboard import SummaryWriter
import argparse


######### Set Seeds ###########
random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

start_epoch = 1

parser = argparse.ArgumentParser(description='Image Deraininig')

parser.add_argument('--datasets_config', default='./configs/datasetsconfig.json', type=str, help='datasets')
parser.add_argument('--train_dataset_name', default='vt5000_tr', type=str, help='Directory of train images')
parser.add_argument('--test_dataset_name', default='vt5000_te', type=str, help='Directory of train images')
parser.add_argument('--model_save_dir', default='/result/checkpoints/', type=str, help='Path to save weights')
parser.add_argument('--pretrain_weights', default='', type=str, help='Path to pretrain-weights')
parser.add_argument('--mode', default='Salient', type=str)
parser.add_argument('--session', default='SingleScale', type=str, help='session')
parser.add_argument('--patch_size', default=256, type=int, help='patch size')
parser.add_argument('--num_epochs', default=100, type=int, help='num_epochs')
parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
args = parser.parse_args()

mode = args.mode
session = args.session
patch_size = args.patch_size

model_dir = os.path.join(args.model_save_dir, mode, 'models', session)
utils.mkdir(model_dir)

dataset_config = args.datasets_config
train_dataset_name = args.train_dataset_name
test_dataset_name = args.test_dataset_name
num_epochs = args.num_epochs
val_epochs = 100
batch_size = args.batch_size

start_lr = 1e-3
end_lr = 1e-6
img_shape = {'h': 384, 'w': 384}

######### Model ###########

model_restoration = myNet()

# print number of model
get_parameter_number(model_restoration)

model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

optimizer = optim.Adam(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - warmup_epochs, eta_min=end_lr)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

RESUME = False
Pretrain = False
model_pre_dir = ''

######### Pretrain ###########
if Pretrain:
    utils.load_checkpoint(model_restoration, model_pre_dir)

    print('------------------------------------------------------------------------------')
    print("==> Retrain Training with: " + model_pre_dir)
    print('------------------------------------------------------------------------------')

######### Resume ###########
if RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Loss ###########
criterion_BCE = losses.BCEloss()
criterion_UAL = losses.UALloss()

######### DataLoaders ###########
with open(dataset_config, 'r', encoding='utf-8') as file:
    datasets_cfg = json.load(file)
datasets_cfg_train = tuple((train_dataset_name, datasets_cfg[train_dataset_name]))
datasets_cfg_train = [datasets_cfg_train]
train_dataset = get_training_data(datasets_cfg_train, img_shape)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False,
                          pin_memory=True)
datasets_cfg_test = tuple((test_dataset_name, datasets_cfg[test_dataset_name]))
datasets_cfg_test = [datasets_cfg_test]
val_dataset = get_training_data(datasets_cfg_test, img_shape)
val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False,
                          pin_memory=True)


print('===> Start Epoch {} End Epoch {}'.format(start_epoch, num_epochs + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0
writer = SummaryWriter(model_dir)
iter = 0

for epoch in range(start_epoch, num_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None
        data = data['data']
        target_ = data['mask'].cuda()
        input_rgb = data['image1.0'].cuda()
        input_thermal = data['thermal'].cuda()

        target = kornia.geometry.transform.build_pyramid(target_, 3)
        restored = model_restoration(input_rgb, input_thermal)

        loss_bce = criterion_BCE(restored, target_)
        loss_ual = criterion_UAL(restored, target_)
        loss = loss_bce + loss_ual
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        iter += 1
        writer.add_scalar('loss/bce_loss', loss_bce, iter)
        writer.add_scalar('loss/ual_loss', loss_bce, iter)
        writer.add_scalar('loss/iter_loss', loss, iter)
    writer.add_scalar('loss/epoch_loss', epoch_loss, epoch)

    #### Evaluation ####
    if epoch % val_epochs == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()

            with torch.no_grad():
                restored = model_restoration(input_)

            for res, tar in zip(restored[0], target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        writer.add_scalar('val/psnr', psnr_val_rgb, epoch)
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

writer.close()
