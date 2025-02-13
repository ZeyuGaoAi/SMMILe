import os
import cv2
import argparse
from tqdm import tqdm
import numpy as np
import glob
import pandas as pd
import concurrent.futures
from PIL import Image
import torchvision.transforms as transforms

from pre_utils import _extract_features
from pre_utils import ResNet50

from conch.open_clip_custom import create_model_from_pretrained


def process_directory(img_dir):
    img_id = os.path.basename(img_dir)
    labels = os.listdir(img_dir)  

    patches = []
    indexes = []
    inst_labels = []

    for label in labels:
        label = int(label)
        image_path_1cls = os.path.join(img_dir, str(label), '*')
        img_paths = glob.glob(image_path_1cls)

        for img_path in img_paths:
            img_fname = os.path.splitext(os.path.basename(img_path))[0]
            img = Image.open(img_path)
            index = img_fname
            patches.append(img)
            indexes.append(index)
            inst_labels.append(label_trans[label])

    f_all = _extract_features(model, patches, encoder_name, preprocess=preprocess, batch_size=16)
    
    feature_npy = {
        'index': indexes,
        'inst_label': inst_labels,
        'feature': f_all
    }
    feature_path = os.path.join(feature_dir, img_id + file_suffix)
    np.save(feature_path, feature_npy)
    print(f'Saved feature to {feature_path}')
    return True


# Argument parser
parser = argparse.ArgumentParser(description="Process WSI data")
parser.add_argument('--feature_dir', type=str, default='/home/shared/su123/TCGA_Embed/conch/TCGA-STAD/', help='Directory to save extracted features')
parser.add_argument('--patch_dirs', type=str, default='/home/shared/su123/TCGA_ORI/TCGA-STAD-Patch/*', help='Path to patch directories')
parser.add_argument('--encoder_name', type=str, default='conch', choices=['conch', 'resnet50'], help='Feature extractor model')
parser.add_argument('--file_suffix', type=str, default='_1_512.npy', help='Suffix for saved feature files')
parser.add_argument('--patch_size', type=int, default=2048, help='Patch size for extraction')
parser.add_argument('--num_workers', type=int, default=5, help='Number of workers for parallel processing')

args = parser.parse_args()

# Assign parsed arguments to variables
feature_dir = args.feature_dir
patch_dirs = glob.glob(args.patch_dirs)
encoder_name = args.encoder_name
file_suffix = args.file_suffix
patch_size = args.patch_size
num_workers = args.num_workers

# Define label mapping
label_trans = {0:3, 1:0, 2:1, 3:2, 4:2}

# Load encoder model
if encoder_name == 'conch':
    model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "/home/z/zeyugao/PreModel/conch/pytorch_model.bin")
elif encoder_name == 'resnet50':
    model = ResNet50(pretrained=True)
    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
else:
    raise ValueError(f"Unsupported encoder_name: {encoder_name}")

model = model.cuda()

print("Data to process: %d \n" % len(patch_dirs))

# Parallel processing
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(tqdm(executor.map(process_directory, patch_dirs), total=len(patch_dirs)))
