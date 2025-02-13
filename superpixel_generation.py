import os
import argparse
import glob
import numpy as np
from skimage import segmentation
from tqdm import tqdm
from PIL import Image

from concurrent.futures import ThreadPoolExecutor

def get_nic_with_coord(features, coords, size, inst_label):
        w = coords[:,0]
        h = coords[:,1]
        w_min = w.min()
        w_max = w.max()
        h_min = h.min()
        h_max = h.max()
        image_shape = [(w_max-w_min)//size+1,(h_max-h_min)//size+1]
        mask = np.ones((image_shape[0], image_shape[1]))
        labels = -np.ones((image_shape[0], image_shape[1]))
        features_nic = np.ones((features.shape[-1], image_shape[0], image_shape[1])) * np.nan
        coords_nic = -np.ones((image_shape[0], image_shape[1], 2))
        # Store each patch feature in the right position
        if inst_label != []:
            for patch_feature, x, y, label in zip(features, w, h, inst_label):
                coord = [x,y]
                x_nic, y_nic = (x-w_min)//size, (y-h_min)//size
                features_nic[:, x_nic, y_nic] = patch_feature
                coords_nic[x_nic, y_nic] = coord
                labels[x_nic, y_nic]=label
        else:
            for patch_feature, x, y in zip(features, w, h):
                coord = [x,y]
                x_nic, y_nic = (x-w_min)//size, (y-h_min)//size
                features_nic[:, x_nic, y_nic] = patch_feature
                coords_nic[x_nic, y_nic] = coord
            labels=[]

        # Populate NaNs
        mask[np.isnan(features_nic)[0]] = 0
        features_nic[np.isnan(features_nic)] = 0
        return features_nic, mask, labels

def get_sp_label(inst_label_nic, m_slic):
    inst_label_nic = inst_label_nic.T
    
    sp_inst_label_all = []
    
    pred_label_all = []
    
    for i in np.unique(m_slic):
        if i == 0:
            continue
        sp_inst_label = inst_label_nic[m_slic==i]
        sp_inst_label = sp_inst_label.astype(np.int32)
        sp_inst_label[sp_inst_label<0] = 0
        counts = np.bincount(sp_inst_label)
        label = np.argmax(counts)
        
        pred_label_all += [label]*sp_inst_label.shape[0]
        
        sp_inst_label_all += list(sp_inst_label)

    return sp_inst_label_all, pred_label_all

def generate_adjacency_matrix(superpixel_matrix):
    # 获取超像素矩阵的形状
    rows, cols = superpixel_matrix.shape
    num_superpixels = np.max(superpixel_matrix)+1

    # 创建邻接矩阵，并初始化为0
    adjacency_matrix = np.zeros((num_superpixels, num_superpixels), dtype=int)

    # 遍历超像素矩阵的每个像素
    for i in range(rows):
        for j in range(cols):
            current_superpixel = superpixel_matrix[i, j]
            
            if current_superpixel == 0:
                continue

            # 检查当前像素的上方和左方是否与当前超像素相邻
            if i > 0 and superpixel_matrix[i-1, j] != current_superpixel:
                adjacency_matrix[current_superpixel, superpixel_matrix[i-1, j]] = 1
                adjacency_matrix[superpixel_matrix[i-1, j], current_superpixel] = 1

            if j > 0 and superpixel_matrix[i, j-1] != current_superpixel:
                adjacency_matrix[current_superpixel, superpixel_matrix[i, j-1]] = 1
                adjacency_matrix[superpixel_matrix[i, j-1], current_superpixel] = 1

    return adjacency_matrix

def process_file(fea_path, args):
    base_name = os.path.basename(fea_path)
    record = np.load(fea_path, allow_pickle=True)
    features = record[()][args.keyword_feature] #.cpu()
    
    coords = record[()]['index']
    if isinstance(coords[0], np.ndarray):
        coords_nd = np.array(coords)
    else:
        coords_nd = np.array([[int(i.split('_')[0]),int(i.split('_')[1])] for i in coords])

    if 'inst_label' in record[()].keys():
        inst_label = record[()]['inst_label']
    else:
        inst_label = [0 for _ in range(coords_nd.shape[0])]

    features = (features - features.min()) / (features.max() - features.min()) * 255
    features_nic, mask, _ = get_nic_with_coord(features, coords_nd, args.size, inst_label)

    data = np.transpose(features_nic, (1, 2, 0))
    n_segments = max(10, int(features.shape[0] / args.n_segments_persp))
    m_slic = segmentation.slic(data, n_segments=n_segments, mask=mask, compactness=args.compactness, start_label=1).T
    m_adj = generate_adjacency_matrix(m_slic)

    sp_path = os.path.join(args.sp_dir, base_name)
    sp_contents = {'m_slic': m_slic, 'm_adj': m_adj}
    
    if np.max(m_slic) != (m_adj.shape[0] - 1):
        print(fea_path)
        print(np.max(m_slic))
        print(m_adj.shape)

    np.save(sp_path, sp_contents)

def main():
    parser = argparse.ArgumentParser(description="Process WSIs for Superpixel Segmentation")
    parser.add_argument('--size', type=int, default=2048, help='Patch size of each individual feature')
    parser.add_argument('--file_suffix', type=str, default='*1_256.npy', help='Suffix for numpy files')
    parser.add_argument('--n_segments_persp', type=int, default=16, help='Number of patches per super-patch')
    parser.add_argument('--keyword_feature', type=str, default='feature2', help='feature keyword in npy file, feature2 is the 1024 dim of resnet')
    parser.add_argument('--compactness', type=int, default=1, help='Compactness for SLIC')
    parser.add_argument('--fea_dir', type=str, default='/home/z/zeyugao/dataset/WSIData/TCGARenal/res50/', help='Directory for feature embeddings of WSIs')
    parser.add_argument('--sp_dir', type=str, default='/home/z/zeyugao/dataset/WSIData/TCGARenal/sp_n%d_c%d_%d/', help='Directory for saving superpixel segmentation results')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads')

    args = parser.parse_args()
    args.sp_dir = args.sp_dir % (args.n_segments_persp, args.compactness, args.size) # superpixel result saving directory

    print("Superpixel Segmentation Results will be saved to: %s" % args.sp_dir)

    file_list = glob.glob(os.path.join(args.fea_dir, args.file_suffix))

    if not os.path.exists(args.sp_dir):
        os.makedirs(args.sp_dir)

    # Use ThreadPoolExecutor for multi-threading
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(lambda fea_path: process_file(fea_path, args), file_list), total=len(file_list)))

if __name__ == "__main__":
    main()
