import json
import os
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
from data_RGB import get_test_data
from model import MultiscaleNet as mynet

from get_parameter_number import get_parameter_number
from tqdm import tqdm
from layers import *

parser = argparse.ArgumentParser(description='Image Deraining')
parser.add_argument('--datasets_config', default='./configs/datasetsconfig.json', type=str, help='datasets')
parser.add_argument('--test_dataset_name', default='vt5000_te', type=str, help='Directory of train images')
parser.add_argument('--output_dir', default='./results/VT5000', type=str, help='Directory of validation images')
parser.add_argument('--weights', default='', type=str, help='Path to weights') 
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--win_size', default=256, type=int, help='window size')
args = parser.parse_args()
result_dir = args.output_dir
win = args.win_size
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
model_restoration = mynet()
get_parameter_number(model_restoration)
utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

img_shape = {'h': 384, 'w': 384}

# dataset = args.dataset
dataset_config = args.dataset_config
test_dataset_name = args.test_dataset_name
with open(dataset_config, 'r', encoding='utf-8') as file:
    datasets_cfg = json.load(file)
datasets_cfg_test = tuple((test_dataset_name, datasets_cfg[test_dataset_name]))
datasets_cfg_test = [datasets_cfg_test]
train_dataset = get_test_data(datasets_cfg_test, img_shape)
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False,
                          pin_memory=True)

utils.mkdir(result_dir)

with torch.no_grad():
    psnr_list = []
    ssim_list = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        input_    = data_test[0].cuda()
        filenames = data_test[1]
        _, _, Hx, Wx = input_.shape
        filenames = data_test[1]
        input_re, batch_list = window_partitionx(input_, win)
        restored = model_restoration(input_re)
        restored = window_reversex(restored[0], win, Hx, Wx, batch_list)

        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        for batch in range(len(restored)):
            restored_img = restored[batch]
            restored_img = img_as_ubyte(restored[batch])
            utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)
