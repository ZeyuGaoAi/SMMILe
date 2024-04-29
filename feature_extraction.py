import os
import cv2
import argparse
from tqdm import tqdm
import numpy as np
from glob import glob
import pandas as pd
import concurrent.futures

from pre_utils import generate_binary_mask_for_wsi, cut_patches_from_wsi_bn 
from pre_utils import cut_patches_from_wsi, extract_features, ResNet50

def extract_embedding_wsi(i, params):

    patch_size = params['patch_size']
    step_size = params['step_size']
    out_size = params['out_size']
    level = params['level']
    binary_rates = params['binary_rates']
    cancer_rates = params['cancer_rates']
    wsi_dir = params['wsi_dir']
    file_list = params['file_list']
    anno_dir = params['anno_dir']
    anno_list = params['anno_list']
    feature_dir = params['feature_dir']
    model = params['model']

    # renal tcga # red, green -> tumor
    color_dict = [[255,0,0],[0,255,0]]
    
    wsi_path = wsi_dir + '%s/%s' % (file_list.iloc[i]['new_folder'], file_list.iloc[i]['filename'])
    
    feature_path = feature_dir + '%s_%s_%d.npy' % ('.'.join(file_list.iloc[i]['filename'].split('.')[:-1]), level, patch_size)
    anno_path = anno_dir + '%s.png' % (file_list.iloc[i]['folder'])
    
    if os.path.exists(feature_path):
        return False

    _, binary_mask, _, _ = generate_binary_mask_for_wsi(wsi_path, level_max=3)
    
    all_patches = []
    all_fnames = []
    inst_labels = []
    
   # for annotated wsi
    if anno_path in anno_list:
        cancer_mask = cv2.imread(anno_path)
        cancer_mask = cv2.cvtColor(cancer_mask, cv2.COLOR_BGR2RGB)

        cancer_mask_binary = np.zeros(cancer_mask.shape[:-1])
        cancer_mask_binary[(cancer_mask!=[0,0,0]).any(axis=-1)] = 255

        patches, fnames = cut_patches_from_wsi('normal', wsi_path, binary_mask, cancer_mask_binary, level, patch_size, 
                                               step_size, binary_rates, cancer_rates, out_size, level_max=3)
        all_patches += patches
        all_fnames += fnames
        inst_labels += [0] * len(fnames)

        # for cancer subtypes
        for color in color_dict:
            cancer_mask_binary = np.zeros(cancer_mask.shape[:-1])
            cancer_mask_binary[(cancer_mask==color).all(axis=-1)] = 255
            patches, fnames = cut_patches_from_wsi('cancer', wsi_path, binary_mask, cancer_mask_binary, level, patch_size, 
                                                   step_size, binary_rates, cancer_rates, out_size, level_max=3)
            all_patches += patches
            all_fnames += fnames
            inst_labels += [1] * len(fnames)
        
    else: # without annotation
#             continue
        all_patches, all_fnames = cut_patches_from_wsi_bn(wsi_path, binary_mask, level, patch_size, step_size, binary_rates, out_size)
#             inst_labels  += [0 for i in range(len(all_fnames))]
        inst_labels = []

    all_patches = [np.array(patch)[...,:3] for patch in all_patches]
    
    f3, f2, f1 = extract_features(model, all_patches)

    feature_npy = {}
    feature_npy['index'] = all_fnames
    feature_npy['inst_label'] = inst_labels
    feature_npy['feature3'] = f3
    feature_npy['feature2'] = f2
    feature_npy['feature1'] = f1
    np.save(feature_path, feature_npy)
    return False

def main(args):

    if not os.path.exists(args.feature_dir):
        os.mkdir(args.feature_dir)
    file_list = pd.read_csv(args.file_list_path)
    anno_list = glob.glob(os.path.join(args.anno_dir, '*.png'))

    wsi_path_all = glob.glob(os.path.join(args.wsi_dir, "*/*.svs"))
    wsi_path_all = pd.DataFrame(wsi_path_all)
    wsi_path_all['new_folder'] = wsi_path_all[0].apply(lambda x: x.split('/')[-2])
    wsi_path_all['filename'] = wsi_path_all[0].apply(lambda x: x.split('/')[-1])
    file_list = file_list.merge(wsi_path_all, how='inner', on='filename')

    model = ResNet50(pretrained=True)
    model = model.cuda()

    params = {
        'patch_size': args.patch_size,
        'step_size': args.step_size,
        'out_size': args.out_size,
        'level': args.level,
        'binary_rates': args.binary_rates,
        'cancer_rates': args.cancer_rates,
        'model': model,
        'wsi_dir': args.wsi_dir,
        'file_list': file_list,
        'anno_dir': args.anno_dir,
        'anno_list': anno_list,
        'feature_dir': args.feature_dir
    }

    data_list = range(file_list.shape[0])

    # 创建一个ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(extract_embedding_wsi, data_id, params): data_id for data_id in data_list}
        
        # 使用tqdm显示进度
        results = []
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(data_list)):
            data_id = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing {data_id}: {e}")

    print("Results:", results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process WSI data")
    parser.add_argument('--feature_dir', type=str, default='/home/z/zeyugao/dataset/WSIData/TCGARenal/res50/')
    parser.add_argument('--anno_dir', type=str, default='/home/z/zeyugao/dataset/WSIData/TCGARenal/annotation/')
    parser.add_argument('--wsi_dir', type=str, default='/home/z/zeyugao/dataset/TCGA-RCC/')
    parser.add_argument('--file_list_path', type=str, default='/home/z/zeyugao/dataset/WSIData/TCGARenal/annotated_slide_list.txt')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of workers for ThreadPoolExecutor')
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--step_size', type=int, default=512)
    parser.add_argument('--out_size', type=int, default=512, help='output size for the feature extraction')
    parser.add_argument('--level', type=int, default=1, help='the level of patch extraction')
    parser.add_argument('--binary_rates', type=float, default=0.5, help='background threshold for keeping patches')
    parser.add_argument('--cancer_rates', type=float, default=0.25, help='cancerous threshold for counting cancerous patches')

    args = parser.parse_args()

    main(args)