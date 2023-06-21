import numpy as np
import onnxruntime
import torch
import onnxruntime as ort

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
# '!!180226_13_kol_CVAT',
# '!!180226_15_ush_CVAT',
# '!!180329_14_ant_CVAT',
# '!!180405_11_ant_CVAT',
# '!!190408_14_kol_CVAT',
# '!!180510_11_ush_CVAT',
# '!!180618_10_anu_CVAT',
# '!!180723_13_ush_CVAT',
# '!!190712_08_ush_CVAT'
]

nami_seg_folder = '/home/khabibulin/DATA'

print(ort.get_available_providers())
print(ort.get_device())
session = ort.InferenceSession("bisenetv2_11.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

test_transforms = transforms.compose([
    transforms.pre_transforms(HEIGHT, WIDTH),
    transforms.post_transforms()
])

valid_dataset = DatasetNami(nami_seg_folder, valid_folders, classes=CLASSES, augmentation=test_transforms, full_road=True)

input_name = session.get_inputs()[0].name
sum = 0
io_binding = session.io_binding()
for i in tqdm(range(len(valid_dataset))):
    img_mask = valid_dataset[i]
    # X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(torch.unsqueeze(img_mask['image'], 0).numpy(), 'cuda', 0)
    X = torch.unsqueeze(img_mask['image'], 0).contiguous()
    io_binding.bind_input(name='input', device_type='cuda', device_id=0, element_type=np.float32, shape=tuple(X.shape), buffer_ptr=X.data_ptr())
    io_binding.bind_output('output')
    session.run_with_iobinding(io_binding)
    outputs = io_binding.copy_outputs_to_cpu()
    # xx = torch.unsqueeze(img_mask['image'], 0).numpy()  # На вход нужна размерность 1,3,512,1024. (img_mask['image'] tensor)
    # outputs = session.run(None, {input_name: xx})

    masks = torch.from_numpy(outputs[0])

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