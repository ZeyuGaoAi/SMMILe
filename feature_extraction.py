import os
import cv2
import argparse
from tqdm import tqdm
import numpy as np
import glob
import pandas as pd
import concurrent.futures
import openslide
import torchvision.transforms as transforms

from pre_utils import generate_binary_mask_for_wsi, cut_patches_from_wsi_bn, str_to_color_list
from pre_utils import cut_patches_from_wsi, extract_and_process_patches, ResNet50

from conch.open_clip_custom import create_model_from_pretrained

# from conch.open_clip_custom import create_model_from_pretrained

def extract_embedding_wsi(i, params):

    patch_size = params['patch_size']
    step_size = params['step_size']
    out_size = params['out_size']
    level = params['level']
    level_max = params['level_max']
    binary_rates = params['binary_rates']
    cancer_rates = params['cancer_rates']
    wsi_dir = params['wsi_dir']
    file_list = params['file_list']
    anno_dir = params['anno_dir']
    anno_list = params['anno_list']
    feature_dir = params['feature_dir']
    model = params['model']
    encoder_name = params['encoder_name']
    preprocess = params['preprocess']
    
    wsi_path = os.path.join(wsi_dir,'%s/%s' % (file_list.iloc[i]['folder'], file_list.iloc[i]['filename']))
    
    feature_path = os.path.join(feature_dir, '%s_%s_%d.npy' % ('.'.join(file_list.iloc[i]['filename'].split('.')[:-1]), level, patch_size))
    anno_path = anno_dir + '%s.png' % (os.path.splitext(file_list.iloc[i]['filename'])[0])
    
    if os.path.exists(feature_path):
        print("feature exist: %s" % (feature_path))
        return False
    
    oslide = openslide.OpenSlide(wsi_path)

    _, binary_mask, _ = generate_binary_mask_for_wsi(oslide, level_max=level_max)
    
    all_corrs = []
    inst_labels = []
    
    
   # for annotated wsi
    if anno_path in anno_list:
        # print("find annotation %s" % (anno_path))
        cancer_mask = cv2.imread(anno_path)
        cancer_mask = cv2.cvtColor(cancer_mask, cv2.COLOR_BGR2RGB)

        cancer_mask_binary = np.zeros(cancer_mask.shape[:-1])
        cancer_mask_binary[(cancer_mask!=[0,0,0]).any(axis=-1)] = 255

        corrs = cut_patches_from_wsi('normal', oslide, binary_mask, cancer_mask_binary, level, patch_size, 
                                               step_size, binary_rates, cancer_rates, out_size, level_max=level_max)
        # all_patches += patches
        # all_fnames += fnames
        all_corrs += corrs
        inst_labels += [0] * len(corrs)

        # for cancer subtypes
        for color in args.color_dict:
            if np.any((cancer_mask == color).all(axis=-1)):
                cancer_mask_binary = np.zeros(cancer_mask.shape[:-1])
                cancer_mask_binary[(cancer_mask==color).all(axis=-1)] = 255
                corrs = cut_patches_from_wsi('cancer', oslide, binary_mask, cancer_mask_binary, level, patch_size, 
                                                    step_size, binary_rates, cancer_rates, out_size, level_max=level_max)
                # all_patches += patches
                # all_fnames += fnames
                all_corrs += corrs
                inst_labels += [1] * len(corrs)
        
    else: # without annotation
        # print("cannot find annotation %s" % (anno_path))
#             continue
        all_corrs = cut_patches_from_wsi_bn(oslide, binary_mask, level, patch_size, step_size, binary_rates, out_size, level_max=level_max)
#             inst_labels  += [0 for i in range(len(all_fnames))]
        inst_labels = []

    all_fea, all_fnames = extract_and_process_patches(all_corrs, oslide, out_size, level, patch_size, 2000, 32, model, encoder_name, preprocess)

    feature_npy = {}
    feature_npy['index'] = all_fnames
    feature_npy['inst_label'] = inst_labels
    feature_npy['feature'] = all_fea
    np.save(feature_path, feature_npy)

    print("save feature to %s" % (feature_path))

    oslide.close()
    return True

def main(args):

    if not os.path.exists(args.feature_dir):
        os.mkdir(args.feature_dir)
    file_list = pd.read_csv(args.file_list_path)
    anno_list = glob.glob(os.path.join(args.anno_dir, '*.png'))

    # fix the folder unpaired problem of two hpcs
    # wsi_path_all = glob.glob(os.path.join(args.wsi_dir, "*/*.svs"))
    # wsi_path_all = pd.DataFrame(wsi_path_all)
    # wsi_path_all['new_folder'] = wsi_path_all[0].apply(lambda x: x.split('/')[-2])
    # wsi_path_all['filename'] = wsi_path_all[0].apply(lambda x: x.split('/')[-1])
    # file_list = file_list.merge(wsi_path_all, how='inner', on='filename')
    # file_list['folder'] = file_list['new_folder']

    print("Data need to process: %d \n" % file_list.shape[0])
    print("Data have spatial annotation: %d \n" % len(anno_list))

    if args.encoder_name == 'conch':
        model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "/home/z/zeyugao/PreModel/conch/pytorch_model.bin")
    elif args.encoder_name == 'resnet50':
        model = ResNet50(pretrained=True)
        preprocess = transforms.Compose(
                                [
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ]
                            )
    else:
        raise ValueError(f"Unsupported encoder_name: {args.encoder_name}")
        
    model = model.cuda()

    params = {
        'encoder_name': args.encoder_name,
        'patch_size': args.patch_size,
        'step_size': args.step_size,
        'out_size': args.out_size,
        'level': args.level,
        'level_max': args.level_max,
        'binary_rates': args.binary_rates,
        'cancer_rates': args.cancer_rates,
        'model': model,
        'wsi_dir': args.wsi_dir,
        'file_list': file_list,
        'anno_dir': args.anno_dir,
        'anno_list': anno_list,
        'feature_dir': args.feature_dir,
        'preprocess': preprocess
    }

    start = args.start
    if args.end == -1:
        end = file_list.shape[0]
    else:
        end = args.end
    data_list = range(start, end)

    # for i in tqdm(data_list):
    #     result = extract_embedding_wsi(i, params)

    # 创建一个ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(extract_embedding_wsi, data_id, params): data_id for data_id in data_list}
        
        # 使用tqdm显示进度
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data_list)):
            data_id = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing {data_id}: {e}")

    print("Results:", results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process WSI data")
    parser.add_argument('--encoder_name', type=str, default='resnet50', help='encoder name')
    parser.add_argument('--feature_dir', type=str, default='/home/z/zeyugao/dataset/WSIData/TCGARenal/res50/', help='dictionary path for saving extracted embeddings')
    parser.add_argument('--anno_dir', type=str, default='/home/z/zeyugao/dataset/WSIData/TCGARenal/annotation/', help='dictionary path for pixel annotations')
    parser.add_argument('--wsi_dir', type=str, default='/home/z/zeyugao/dataset/TCGA-RCC/', help='dictionary path for original WSI(svs) files')
    parser.add_argument('--file_list_path', type=str, default='/home/z/zeyugao/dataset/WSIData/TCGARenal/slide_list.txt', help='list for WSIs')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of workers for ThreadPoolExecutor')
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--step_size', type=int, default=512)
    parser.add_argument('--out_size', type=int, default=224, help='output size for the feature extraction')
    parser.add_argument('--level', type=int, default=1, help='the level of patch extraction, 0 is the highest')
    parser.add_argument('--level_max', type=int, default=1, help='the max level for binary mask, 0 is the highest')
    parser.add_argument('--binary_rates', type=float, default=0.25, help='background threshold for keeping patches')
    parser.add_argument('--cancer_rates', type=float, default=0.25, help='cancerous threshold for counting cancerous patches')

    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)

    parser.add_argument('--color_dict', type=str, default="255,0,0;0,255,0;255,255,255", help='Color dictionary as a string, e.g., "255,0,0;0,255,0"')

    args = parser.parse_args()

    # renal tcga # red, green, white -> tumor
    args.color_dict = str_to_color_list(args.color_dict)

    main(args)