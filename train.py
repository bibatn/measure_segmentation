#imports
import os
import sys
import glob
import time
import random

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
#from torch.utils.tensorboard import SummaryWriter

import catalyst
from catalyst import utils
#from catalyst.utils import metrics
from catalyst.contrib.nn import RAdam, Lookahead

import numpy as np
import matplotlib.pyplot as plt
import cv2

# import albumentations as albu
# from albumentations.pytorch.transforms import ToTensor

from nami_segmentation.datasets import DatasetNami
from nami_segmentation.utils import create_tensorboard_writers, show_imgmask, visualize
from nami_segmentation.models.bisenetv2_aux_dw import BiSeNetV2_aux_dw
from nami_segmentation.loss_functions import combo_seg_loss, compute_tp_fp_fn
from nami_segmentation.training import fit
import nami_segmentation.transforms as transforms


#settings
HEIGHT = 512
WIDTH = 1024

NUM_WORKERS = 16
BATCH_SIZE = 8

CLASSES = ['person',
           'vehicle',
           'bicycle',
           'lstart/env/bin/python3 ight',
           'sign',
           'road',
           'moto',
           'zebra',
           'dashed',
           'solid',
           'doublesolid']

SEED = 42

nami_seg_folder = '/mnt/data/VideoAi/Khabibulin/DB_segmentation_copy_121121'

train_folders = [
# 'cf_short_221130_train',
# 'cf_221130_train',
'!!180115_12_kol_CVAT',
'!!180202_11_anu_CVAT',
'!!180222_12_ant_CVAT(НАМИ)',
'!!180226_11_shy_CVAT',
'!!180226_12_shu_CVAT',
'!!180226_14_ush_CVAT',
'!!180115_13_ant_CVAT',
'!!180329_15_ant_CVAT',
'!!180405_12_ant_CVAT',
'!!180510_14_anu_CVAT',
'!!180723_11_2_ush_CVAT',
'!!190116_15_ush_CVAT',
'!!190318_11_kol_CVAT',
'!!190325_15_ant_CVAT',
'!!190401_11_kol_CVAT',
'!!190401_12_anu_CVAT',
'!!190415_12_ush_CVAT',
'!!190516_11_ush_CVAT',
#'!!190516_12_ush_CVAT',
'!!190516_13_ush',
'!!190606_10_3_anu_CVAT',
'!!190613_09_shu_CVAT',
'!!190718_17_ant_CVAT',
'!!190912_15_ant_CVAT',
'!!190527_14_CVAT',
'!!201014_11_kol_CVAT(НАМИ)'
]

valid_folders = [
# 'cf_short_221130_valid',
# 'cf_221130_valid',
'!!180216_12_ant_CVAT',
'!!180226_13_kol_CVAT',
'!!180226_15_ush_CVAT',
'!!180329_14_ant_CVAT',
'!!180405_11_ant_CVAT',
'!!190408_14_kol_CVAT',
'!!180510_11_ush_CVAT',
'!!180618_10_anu_CVAT',
'!!180723_13_ush_CVAT',
'!!190712_08_ush_CVAT'
]

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0003
NUM_EPOCHS = 1500

LOG_DIR = "logs"

model = BiSeNetV2_aux_dw(len(CLASSES))
weights_path = '/mnt/work/krapukhin/projects/segmentation_new/training/bisenetv2_namicsaudi_512x1024_pe-ve-bi-li-si-ro-mo-ze-da-so-do_combo_191121/trained_models/best_585.pth'


device = utils.get_device()
print(f"device: {device}")
model = model.to(device)
model = nn.DataParallel(model)

# current_model_dict = model.state_dict()
loaded_state_dict = torch.load(weights_path)
# new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
# new_state_dict['module.head.conv_out.weight'][:-3, :, :, :] = loaded_state_dict['module.head.conv_out.weight']
# new_state_dict['module.head.conv_out.bias'][:-3] = loaded_state_dict['module.head.conv_out.bias']
# new_state_dict['module.aux2.conv_out.weight'][:-3, :, :, :] = loaded_state_dict['module.aux2.conv_out.weight']
# new_state_dict['module.aux2.conv_out.bias'][:-3] = loaded_state_dict['module.aux2.conv_out.bias']
# new_state_dict['module.aux3.conv_out.weight'][:-3, :, :, :] = loaded_state_dict['module.aux3.conv_out.weight']
# new_state_dict['module.aux3.conv_out.bias'][:-3] = loaded_state_dict['module.aux3.conv_out.bias']
# new_state_dict['module.aux4.conv_out.weight'][:-3, :, :, :] = loaded_state_dict['module.aux4.conv_out.weight']
# new_state_dict['module.aux4.conv_out.bias'][:-3] = loaded_state_dict['module.aux4.conv_out.bias']
# new_state_dict['module.aux5_4.conv_out.weight'][:-3, :, :, :] = loaded_state_dict['module.aux5_4.conv_out.weight']
# new_state_dict['module.aux5_4.conv_out.bias'][:-3] = loaded_state_dict['module.aux5_4.conv_out.bias']
# model.load_state_dict(new_state_dict, strict=False)


model.load_state_dict(loaded_state_dict)


train_writer, val_writer = create_tensorboard_writers()

utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)

print(f"torch: {torch.__version__}, catalyst: {catalyst.__version__}")



#augmentations
train_transforms = transforms.compose([
    transforms.random_crop_nami(),
    transforms.resize_transforms(HEIGHT, WIDTH),
    transforms.hard_transforms(),
    transforms.post_transforms(),
])

valid_transforms = transforms.compose([
    transforms.center_crop_nami(),
    transforms.pre_transforms(HEIGHT, WIDTH),
    transforms.post_transforms()
])

test_transforms = transforms.compose([
    transforms.pre_transforms(HEIGHT, WIDTH),
    transforms.resize_transforms(HEIGHT, WIDTH),
    transforms.post_transforms()
])


dataset = DatasetNami(nami_seg_folder, train_folders, classes=CLASSES, augmentation=train_transforms, full_road=True)
show_imgmask(random.randint(0, len(dataset) - 1), dataset, 0)