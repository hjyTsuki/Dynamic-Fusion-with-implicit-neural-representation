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
parser.add_argument('--train_datasets', default='vt5000_tr', type=str, help='Directory of train images')
parser.add_argument('--model_save_dir', default='/data0/chenxiang/code/CVPR2024/checkpoints/', type=str, help='Path to save weights')
parser.add_argument('--pretrain_weights', default='', type=str, help='Path to pretrain-weights')
parser.add_argument('--mode', default='Salient', type=str)
parser.add_argument('--session', default='SingleScale', type=str, help='session')
parser.add_argument('--patch_size', default=256, type=int, help='patch size')
parser.add_argument('--num_epochs', default=3000, type=int, help='num_epochs')
parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
args = parser.parse_args()

mode = args.mode
session = args.session
patch_size = args.patch_size

model_dir = os.path.join(args.model_save_dir, mode, 'models', session)
utils.mkdir(model_dir)

dataset_config = args.datasets_config
train_datasets = args.train_datasets

num_epochs = args.num_epochs
batch_size = args.batch_size

start_lr = 1e-4
end_lr = 1e-6

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
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()
criterion_fft = losses.fftLoss()
criterion_L1 = nn.L1Loss(size_average=True)

######### DataLoaders ###########
with open(dataset_config, 'r', encoding='utf-8') as file:
    datasets_cfg = json.load(file)
train_dataset = get_training_data(list(datasets_cfg[train_datasets]))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False,
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

        target_ = data[0].cuda()
        input_ = data[1].cuda()
        target = kornia.geometry.transform.build_pyramid(target_, 3)
        restored = model_restoration(input_)

        loss_fft = criterion_fft(restored[0], target[0]) + criterion_fft(restored[1], target[1]) + criterion_fft(restored[2], target[2])
        loss_char = criterion_char(restored[0], target[0]) + criterion_char(restored[1], target[1]) + criterion_char(restored[2], target[2])
        loss_edge = criterion_edge(restored[0], target[0]) + criterion_edge(restored[1], target[1]) + criterion_edge(restored[2], target[2])
        loss_l1 = criterion_L1(restored[3], target[1]) + criterion_L1(restored[5], target[2])
        loss = loss_char + 0.01 * loss_fft + 0.05 * loss_edge + 0.1 * loss_l1
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        iter += 1
        writer.add_scalar('loss/fft_loss', loss_fft, iter)
        writer.add_scalar('loss/char_loss', loss_char, iter)
        writer.add_scalar('loss/edge_loss', loss_edge, iter)
        writer.add_scalar('loss/l1_loss', loss_l1, iter)
        writer.add_scalar('loss/iter_loss', loss, iter)
    writer.add_scalar('loss/epoch_loss', epoch_loss, epoch)

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