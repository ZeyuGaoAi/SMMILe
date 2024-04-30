import cv2
import torch
import openslide
import numpy as np
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torchvision.models
from torchvision import transforms
from torchvision import transforms

# from ctran import ctranspath

class ResNet50(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__()
        base_model = torchvision.models.resnet50(pretrained=pretrained)

        self.base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)

        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)

        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)

        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)

        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.avgpool = self.base_layers[8]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        x4 = self.avgpool(layer4)
        x4 = x4.view(x4.size(0), -1)
        x3 = self.avgpool(layer3)
        x3 = x3.view(x3.size(0), -1)
        x2 = self.avgpool(layer2)
        x2 = x2.view(x2.size(0), -1)
        return x4, x3, x2

def extract_features(model, patches):
    f3=[]
    f2=[]
    f1=[]
    batch_size = 20
    if len(patches)==0:
        return f3, f2, f1
    imgs_all = patches   
    num = len(patches)//batch_size + (1 if len(patches)%batch_size!=0 else 0)
    for k in range(num):
        if k == num-1:
            imgs = np.stack(imgs_all[k*batch_size:len(patches)])
        else:
            imgs = np.stack(imgs_all[k*batch_size:(k+1)*batch_size])
        imgs = img_normalize(imgs)
        imgs = imgs.cuda()
        with torch.no_grad():
            feat3,feat2,feat1=model(imgs)
        f3.append(feat3.cpu())
        f2.append(feat2.cpu())
        f1.append(feat1.cpu())
        
    f3 = torch.cat(f3)
    f2 = torch.cat(f2)
    f1 = torch.cat(f1)
    
    return f3, f2, f1


def mkdic(path):
    df = pd.read_csv(path,header=None, sep=' ')
    df.columns = ['number','uid','slide_name']
    dic = {}
    for index in df.index:
        lines = df.loc[index].values
        lines = np.array(lines)
        #print(lines)
        #dic.update({lines[0].split('.')[0]: lines[1]})
        dic.update({lines[1]: lines[2]})
    return dic

def img_normalize(imgs):
    imgs_normalize = []
    normalize = transforms.Normalize(
        mean = [0.681, 0.447, 0.678],
        std = [0.190, 0.257, 0.178])

    for img in imgs:
        img = np.float32(img) / 255.
        img = np.array(img.transpose((2,0,1)))
        img_ = torch.from_numpy(img.copy())
        img_tensor = normalize(img_)
        imgs_normalize.append(img_tensor)
    return torch.stack(imgs_normalize)

def generate_binary_mask_for_wsi(file_path, level_max=3):
    # 使用CV2的OTSU进行二值化
    oslide = openslide.OpenSlide(file_path)
    magnification = oslide.properties.get('aperio.AppMag')
#     print(oslide.level_dimensions)
#    width = oslide.dimensions[0]
#    height = oslide.dimensions[1]
    level = oslide.level_count - 1
    if level > level_max:
        level = level_max
    scale_down = oslide.level_downsamples[level]
    w, h = oslide.level_dimensions[level]
    # 防止出现没有放大倍数直接处理原图的情况
    if level < 1:
        print(file_path)
        oslide.close()
        return
    else:
        patch = oslide.read_region((0, 0), level, (w, h))
    slide_id = file_path.split('/')[-1].split('.svs')[0]
#     patch.save('{}/{}_resized.png'.format(output_folder, slide_id))
    patch = np.asarray(patch)
    img = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (61, 61), 0)
    # THRESH_TRIANGLE THRESH_OTSU
    ret, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
#     fname = '{}/{}_mask.png'.format(output_folder, slide_id)
#     cv2.imwrite(fname, img_filtered)
    oslide.close()
    
    return patch, mask, slide_id, scale_down

def cut_patches_from_wsi_bn(file_path, binary_mask, level=2, size=1000, step=500, binary_rate=0.5, output_size=512, level_max=3):
    
    patches = []
    fnames = []

    # 处理wsi
    oslide = openslide.OpenSlide(file_path)
    width = oslide.dimensions[0]
    height = oslide.dimensions[1]
    B_level = oslide.level_count - 1
    if B_level > level_max:
        B_level = level_max
    w, h = oslide.level_dimensions[B_level]
    #print(w,h)
    mag_w = width / w
    mag_h = height / h
    mag_size = size*oslide.level_downsamples[level] / mag_w
    # 读取mask图
    binary_mask = binary_mask.T


    #print(binary_mask.shape,cancer_mask.shape)
    if not (binary_mask.shape == (w, h)):
        print("Mask file not match for this WSI!")
        return patches, fnames
    corrs = []

    size_ori = size
    size = size*int(np.round(oslide.level_downsamples[level]))
    step = step*int(np.round(oslide.level_downsamples[level]))

    for x in range(1, width, step):
        for y in range(1, height, step):
            if x + size > width:
                continue
            else:
                w_x = size_ori
            if y + size > height:
                continue
            else:
                w_y = size_ori

            binary_mask_patch = binary_mask[int(x / mag_w):int(x / mag_w + mag_size),
                                int(y / mag_h):int(y / mag_h + mag_size)]
            binary_mask_number = binary_mask_patch[(binary_mask_patch == 255)].size  # 小于阈值
            # print(binary_mask_number, cancer_mask_number)
            if (binary_mask_number < binary_mask_patch.size * binary_rate):
                corrs.append((x, y, w_x, w_y))

    for corr in corrs:
        x, y, w_x, h_y = corr
        patch = oslide.read_region((x, y), level, (w_x, h_y)).convert('RGB')
        patch = patch.resize((output_size, output_size), Image.LANCZOS)
        patches.append(patch)
        fname = '{}_{}_{}.png'.format(x, y, size)
        fnames.append(fname)

    oslide.close()
    return patches, fnames

def cut_patches_from_wsi(flag, file_path, binary_mask, cancer_mask, level=0,
                 size=1000, step=500, binary_rate=0.5, cancer_rate=0.5, output_size=512, level_max=3):
    # 将wsi划窗切分成指定大小的patches
    
    patches = []
    fnames = []

    # 处理wsi
    oslide = openslide.OpenSlide(file_path)

    width = oslide.dimensions[0]
    height = oslide.dimensions[1]
    B_level = oslide.level_count - 1
    if B_level > level_max:
        B_level = level_max
    w, h = oslide.level_dimensions[B_level]
    mag_w = width / w
    mag_h = height / h
    mag_size = size*oslide.level_downsamples[level] / mag_w

    binary_mask = binary_mask.T
    
    cancer_mask = cv2.resize(cancer_mask, (w, h))
    cancer_mask = cancer_mask.T

    if not (binary_mask.shape == (w, h) and cancer_mask.shape == (w, h)):
        print("Mask file not match for this WSI!")
        return patches, fnames
    corrs = []

    size_ori = size
    size = size*int(np.round(oslide.level_downsamples[level]))
    step = step*int(np.round(oslide.level_downsamples[level]))

    for x in range(0, width, step):
        for y in range(0, height, step):
            if x + size > width:
                continue
            else:
                w_x = size_ori
            if y + size > height:
                continue
            else:
                w_y = size_ori
            # 根据mask进行过滤，大于rate个背景则不要
            binary_mask_patch = binary_mask[int(x / mag_w):int(x / mag_w + mag_size),
                                int(y / mag_h):int(y / mag_h + mag_size)]
            cancer_mask_patch = cancer_mask[int(x / mag_w):int(x / mag_w + mag_size),
                                int(y / mag_h):int(y / mag_h + mag_size)]
            binary_mask_number = binary_mask_patch[(binary_mask_patch == 255)].size  # 小于阈值
            cancer_mask_number = cancer_mask_patch[(cancer_mask_patch > 0)].size
            # print(binary_mask_number, cancer_mask_number)
            if (flag == 'normal'):
                if ((binary_mask_number < binary_mask_patch.size * binary_rate) and (cancer_mask_number == 0)):
                    corrs.append((x, y, w_x, w_y))
            elif ('cancer' in flag):
                if ((binary_mask_number < binary_mask_patch.size * binary_rate) and (
                        cancer_mask_number >= cancer_mask_patch.size * cancer_rate)):
                    corrs.append((x, y, w_x, w_y))

    for corr in corrs:
        x, y, w_x, h_y = corr
        patch = oslide.read_region((x, y), level, (w_x, h_y)).convert('RGB')
        patch = patch.resize((output_size, output_size), Image.LANCZOS)
        patches.append(patch)
        fname = '{}_{}_{}.png'.format(x, y, size)
        fnames.append(fname)
#         patch.save(fname)
    oslide.close()
    return patches, fnames