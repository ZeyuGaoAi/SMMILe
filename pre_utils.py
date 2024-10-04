import cv2
import torch
import openslide
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torchvision.models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet50_Weights

def str_to_color_list(color_str):
    # 解析颜色字符串为颜色列表
    color_list = []
    color_str_list = color_str.split(';')
    for c in color_str_list:
        color = [int(x) for x in c.split(',')]
        color_list.append(color)
    return color_list

# from ctran import ctranspath

class PatchDataset(Dataset):
    def __init__(self, patches, transform=None, target_img_size=224):
        self.patches = patches
        self.transform = transform

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        image = self.patches[idx]
        if self.transform:
            image = self.transform(image)
        return image

class ResNet50(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            base_model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            base_model = torchvision.models.resnet50(weights=None)

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
        # layer4 = self.layer4(layer3)
        
        # x4 = self.avgpool(layer4)
        # x4 = x4.view(x4.size(0), -1)
        x3 = self.avgpool(layer3)
        x3 = x3.view(x3.size(0), -1)
        # x2 = self.avgpool(layer2)
        # x2 = x2.view(x2.size(0), -1)
        return x3

def _extract_features(model, patches, encoder_name, preprocess=None, batch_size=16):
    all_embeds=[]

    if len(patches)==0:
        return all_embeds

    transform = transforms.Compose([
            transforms.Lambda(lambda x: preprocess(x))
        ])

    dataset = PatchDataset(patches, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_patches in dataloader:
            batch_patches = batch_patches.cuda()
            if encoder_name == 'conch':
                image_emb = model.encode_image(batch_patches, proj_contrast=False, normalize=False)
            else:
                image_emb = model(batch_patches)
            all_embeds.append(image_emb.cpu())
            del image_emb 
            torch.cuda.empty_cache()

    all_embeds = torch.cat(all_embeds)

    # imgs_all = patches   
    # num = len(patches)//batch_size + (1 if len(patches)%batch_size!=0 else 0)
    # for k in range(num):
    #     if k == num-1:
    #         # imgs = np.stack(imgs_all[k*batch_size:len(patches)])
    #         imgs = imgs_all[k*batch_size:len(patches)]
    #     else:
    #         # imgs = np.stack(imgs_all[k*batch_size:(k+1)*batch_size])
    #         imgs = imgs_all[k*batch_size:(k+1)*batch_size]
    #     imgs = img_normalize(imgs)
    #     imgs = imgs.cuda()
    #     with torch.no_grad():
    #         feat3,feat2,feat1 = model(imgs)
    #     f3.append(feat3.cpu())
    #     f2.append(feat2.cpu())
    #     f1.append(feat1.cpu())
        
    # f3 = torch.cat(f3)
    # f2 = torch.cat(f2)
    # f1 = torch.cat(f1)
    
    return all_embeds

def _batch_generator(data, batch_size):
    """生成器函数，每次返回一个批次的数据"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def _extract_patches(corrs, oslide, output_size, level, size):
    patches = []
    fnames = []
    size = size*int(np.round(oslide.level_downsamples[level]))

    for corr in corrs:
        x, y, w_x, h_y = corr
        patch = oslide.read_region((x, y), level, (w_x, h_y)).convert('RGB')
        patch = patch.resize((output_size, output_size), Image.LANCZOS)
        patches.append(patch)
        fname = '{}_{}_{}.png'.format(x, y, size)
        fnames.append(fname)

    return patches, fnames

def extract_and_process_patches(all_corrs, oslide, out_size, level, patch_size, patch_batch_size, feature_batch_size, model, encoder_name, preprocess):
    # 初始化存储特征的列表
    all_fea, all_fnames = [], []

    # 分批次处理correspondences
    for corrs_batch in _batch_generator(all_corrs, patch_batch_size):

        # 提取当前批次的patches和文件名
        patches, fnames = _extract_patches(corrs_batch, oslide, out_size, level, patch_size)
        # 提取当前批次的特征
        feas = _extract_features(model, patches, encoder_name, preprocess, feature_batch_size)
        del patches 
        # 存储当前批次的特征
        all_fnames += fnames
        all_fea.append(feas)

    # 将所有批次的特征合并成单一的数组
    all_fea = torch.cat(all_fea, dim=0)

    return all_fea, all_fnames

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

# def img_normalize(imgs):
#     imgs_normalize = []
#     normalize = transforms.Normalize(
#         mean = [0.485, 0.456, 0.406],
#         std = [0.229, 0.224, 0.225])
#     # resize = transforms.Resize((target_img_size, target_img_size))

#     for img in imgs:
#         # img = resize(img)
#         img = np.float32(img) / 255.
#         img = np.array(img.transpose((2,0,1)))
#         img_ = torch.from_numpy(img.copy())
#         img_tensor = normalize(img_)
#         imgs_normalize.append(img_tensor)
#     return torch.stack(imgs_normalize)

def img_normalize(imgs, target_img_size=224):
    imgs_normalize = []
    preprocess = transforms.Compose([
        transforms.ToPILImage(),  # 将numpy数组转换为PIL图像
        transforms.Resize((target_img_size, target_img_size)),  # 调整图像大小
        transforms.ToTensor(),  # 将PIL图像转换为张量，并将值归一化到[0, 1]
        transforms.Normalize(  # 使用ImageNet的均值和标准差进行归一化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    for img in imgs:
        # # 确保图像是RGB格式
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        
        # 将PIL图像转换为numpy数组
        # img = np.array(img)
        
        # 应用预处理变换
        img_tensor = preprocess(img)
        
        # 添加到结果列表中
        imgs_normalize.append(img_tensor)
    
    return torch.stack(imgs_normalize)

def generate_binary_mask_for_wsi(oslide, level_max=3):
    # 使用CV2的OTSU进行二值化
    # oslide = openslide.OpenSlide(file_path)
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
        # print(file_path)
        return
    else:
        patch = oslide.read_region((0, 0), level, (w, h))
    # slide_id = file_path.split('/')[-1].split('.svs')[0]
    # patch.save('{}/{}_resized.png'.format(output_folder, slide_id))
    patch = np.asarray(patch)
    img = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    # img = cv2.GaussianBlur(img, (61, 61), 0)
    # THRESH_TRIANGLE THRESH_OTSU
    # ret, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)

    # 去除较浅的区域
    ret, mask1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    
    # 去除较深的区域
    ret, mask2 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 结合两次结果
    mask = cv2.bitwise_and(mask1, mask2)

    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    mask[mask>0] = 255

#     fname = '{}/{}_mask.png'.format(output_folder, slide_id)
#     cv2.imwrite(fname, img_filtered)
    
    return patch, mask, scale_down

def cut_patches_from_wsi_bn(oslide, binary_mask, level=2, size=1000, step=500, binary_rate=0.5, output_size=512, level_max=3):
    
    corrs = []

    # 处理wsi
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
        return oslide, corrs

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
            binary_mask_number = binary_mask_patch[(binary_mask_patch == 255)].size  # 前景部分 255
            # print(binary_mask_number, cancer_mask_number)
            if (binary_mask_number > binary_mask_patch.size * binary_rate):
                corrs.append((x, y, w_x, w_y))

    return corrs

def cut_patches_from_wsi(flag, oslide, binary_mask, cancer_mask, level=0,
                 size=1000, step=500, binary_rate=0.5, cancer_rate=0.5, output_size=512, level_max=3):
    # 将wsi划窗切分成指定大小的patches
    corrs = []

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
        return oslide, corrs

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
            binary_mask_number = binary_mask_patch[(binary_mask_patch == 255)].size   # 前景部分 255
            cancer_mask_number = cancer_mask_patch[(cancer_mask_patch == 255)].size   # cancer区域部分 255
            # print(binary_mask_number, cancer_mask_number)
            if (flag == 'normal'):
                if ((binary_mask_number >= binary_mask_patch.size * binary_rate) and (cancer_mask_number < cancer_mask_patch.size * cancer_rate)):
                    corrs.append((x, y, w_x, w_y))
            elif ('cancer' in flag):
                if ((binary_mask_number >= binary_mask_patch.size * binary_rate) and (
                        cancer_mask_number >= cancer_mask_patch.size * cancer_rate)):
                    corrs.append((x, y, w_x, w_y))

    return corrs
