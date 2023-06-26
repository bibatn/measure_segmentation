import numpy as np
import torch
import onnxruntime as ort
import nami_segmentation.loss_functions as loss

import nami_segmentation.transforms as transforms
from nami_segmentation.datasets import DatasetNami
# from nami_segmentation.testing import create_unique_binary_mask
from tqdm import tqdm
from nami_segmentation.loss_functions import iou_metric

HEIGHT = 512
WIDTH = 1024

CLASSES = ['person',
           'vehicle',
           'bicycle',
           'light',
           'sign',
           'road',
           'moto',
           'zebra',
           'dashed',
           'solid',
           'doublesolid']

valid_folders = [
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

nami_seg_folder = '/mnt/work/krapukhin/projects/segmentation/datasets/DB_segmentation_copy_121121'

num_classes = 11

print(ort.get_available_providers())
print(ort.get_device())
session = ort.InferenceSession("bisenetv2_11.onnx", providers=['CUDAExecutionProvider'])

test_transforms = transforms.compose([
    transforms.pre_transforms(HEIGHT, WIDTH),
    transforms.post_transforms()
])

valid_dataset = DatasetNami(nami_seg_folder, valid_folders, classes=CLASSES, augmentation=test_transforms, full_road=True)

input_name = session.get_inputs()[0].name
sum = 0
tps = np.zeros(num_classes)
fps = np.zeros(num_classes)
fns = np.zeros(num_classes)
for i in tqdm(range(len(valid_dataset))):
    img_mask = valid_dataset[i]
    xx = torch.unsqueeze(img_mask['image'], 0).numpy()  # На вход нужна размерность 1,3,512,1024. (img_mask['image'] tensor)
    outputs = session.run(None, {input_name: xx})

    masks = torch.from_numpy(outputs[0])

    tp, fp, fn = loss.compute_tp_fp_fn(masks.sigmoid(), img_mask["mask"])
    tps += tp
    fps += fp
    fns += fn

    max = masks.max(1)
    masksMax = max[0]
    masksInd = max[1]
    masksInd_ = masksInd + 1

    threshold = 0.5
    threshold = np.log(threshold / (1 - threshold))  # is equal 0
    masksMax_ = masksMax > threshold
    masks_onechannel = masksMax_ * masksInd_  # убраны элементы не относящиеся к классам
    masks_onechannel = masks_onechannel.squeeze()

    masks = img_mask["mask"]
    mask = np.zeros(masks[0].shape)
    for i in range(11):
        mask = np.add(mask, (i + 1) * masks[i])
    sum += np.sum(np.sum((masks_onechannel == mask).numpy(), 0), 0) / (mask.shape[0] * mask.shape[1])
print('accuracy = ', sum / len(valid_dataset))
iou_of_classes = tps / (tps + fps + fns)
total_iou = np.sum(tps) / (np.sum(tps) + np.sum(fps) + np.sum(fns))
mean_iou = np.mean(iou_of_classes)
print(' '.join(f'{iou:.2f}' for iou in iou_of_classes), end='')
print('] {:.2f} {:.2f}'.format(total_iou, mean_iou))