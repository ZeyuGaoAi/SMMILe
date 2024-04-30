import os
import cv2
import pandas as pd
import numpy as np
import torch
import openslide
from models.model_smmile import SMMILe, SMMILe_SINGLE
from utils.utils import *

def get_nic_with_coord(features, coords, size):
        w = coords[:,0]
        h = coords[:,1]
        w_min = w.min()
        w_max = w.max()
        h_min = h.min()
        h_max = h.max()
        image_shape = [(w_max-w_min)//size+1,(h_max-h_min)//size+1]
        mask = np.ones((image_shape[0], image_shape[1]))
        features_nic = torch.ones((features.shape[-1], image_shape[0], image_shape[1])) * np.nan
        coords_nic = -np.ones((image_shape[0], image_shape[1], 2))
        # Store each patch feature in the right position
        
        for patch_feature, x, y in zip(features, w, h):
            coord = [x,y]
            x_nic, y_nic = (x-w_min)//size, (y-h_min)//size
            features_nic[:, x_nic, y_nic] = patch_feature
            coords_nic[x_nic, y_nic] = coord

        # Populate NaNs
        mask[torch.isnan(features_nic)[0]] = 0
        features_nic[torch.isnan(features_nic)] = 0
        
        return features_nic, mask, coords_nic


def initiate_model(model_type, model_size, drop_out, drop_rate, n_classes, fea_dim, n_refs, ckpt_path): 
    model_dict = {'dropout': drop_out, 'drop_rate': drop_rate, 'n_classes': n_classes, 
                  'fea_dim': fea_dim, "size_arg": model_size, 'n_refs': n_refs}
   
    if model_type == 'smmile':
        model = SMMILe(**model_dict)
    elif model_type == 'smmile_single':
        model = SMMILe_SINGLE(**model_dict)
    else:
        raise NotImplementedError

#     print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.relocate()
    model.eval()
    return model


def summary(model, data, patch_size, sp_data,n_classes, model_type, inst_refinement):
      
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    
    patient_results = {}
    
    with torch.no_grad():
        features = data['feature2'].to(device)
        coords =data['index']
        if type(coords[0]) is np.ndarray:
            coords_nd = np.array(coords)
        else:
            coords_nd = np.array([[int(i.split('_')[0]),int(i.split('_')[1])] for i in coords])
        sp = sp_data['m_slic']
        sp = sp.transpose(1,0)
        adj = sp_data['m_adj']
        feaetures_nic, mask,coords_nic =get_nic_with_coord(features, coords_nd, patch_size)
        feaetures_nic = feaetures_nic.to(device)
        score, Y_prob, Y_hat, ref_score, results_dict = model(feaetures_nic, mask, sp, adj,label=[], 
                                                              instance_eval=inst_refinement)

        Y_prob = Y_prob[0]

        if model_type == 'smmile':
            inst_score = list(1 - ref_score[:,-1].detach().cpu().numpy())
            inst_pred = torch.argmax(ref_score, dim=1).detach().cpu().numpy()
            inst_pred = [0 if i==n_classes else 1 for i in inst_pred] # for one-class cancer
        elif model_type == 'smmile_single':
            inst_score = list(ref_score[:,1].detach().cpu().numpy())
            inst_pred = list(torch.argmax(ref_score, dim=1).detach().cpu().numpy())
        else:
            inst_score = list(1 - ref_score[:,-1].detach().cpu().numpy())
            inst_pred = torch.argmax(ref_score, dim=1).detach().cpu().numpy()
            inst_pred = [0 if i==n_classes else 1 for i in inst_pred] # for one-class cancer

        cor_h, cor_w = np.where(mask==1)
        coords = coords_nic[cor_h, cor_w]
        all_patches = [os.path.join("%s_%s_%s" % (int(coords[i][0]), int(coords[i][1]), patch_size)) for i in range(len(coords))]

        probs = Y_prob.cpu().numpy()
        preds = Y_hat.item()
    results_dict = {}
    for c in range(n_classes):
        results_dict['class_{}'.format(c)] = probs[c]
    df_inst = pd.DataFrame([all_patches,inst_score,inst_pred]).T
    df_inst.columns = ['patch', 'prob', 'pred']
    
    return results_dict, df_inst


def main_eval(npy_path,sp_path, model_type, model_size, drop_out, drop_rate, n_classes, fea_dim,patch_size, n_refs, ckpt_path, inst_refinement):
    if not os.path.exists(ckpt_path):
        ckpt_path=ckpt_path.replace('_best.pt','.pt')
    print('Load model weights from %s' % ckpt_path)
    model = initiate_model(model_type, model_size, drop_out, drop_rate, n_classes, fea_dim, n_refs, ckpt_path)
    print('Load WSI embeddings from %s' % npy_path)
    data = np.load(npy_path, allow_pickle=True)[()]
    print('Load WSI superpixel results from %s' % sp_path)
    sp_data = np.load(sp_path, allow_pickle=True)[()]
    results_dict, df_inst = summary(model, data, patch_size,sp_data,n_classes, model_type, inst_refinement)

    return results_dict, df_inst

def visualization_subtype(resized_img, wsi_cors, predict_results, patch_size_ori, scale_down):
    colors = [[255,255,255],[0, 0, 255]]

    resized_img = resized_img[..., :3]
    resized_img = resized_img.transpose((1,0,2))
    resized_img = cv2.cvtColor(resized_img,cv2.COLOR_RGB2BGR)
    w,h,_=resized_img.shape
    x_cor = [int(i[0]) for i in wsi_cors]
    y_cor = [int(i[1]) for i in wsi_cors]
    x_cor = [int(i/scale_down) for i in x_cor]
    y_cor = [int(i/scale_down) for i in y_cor]
    patch_size = int(patch_size_ori/scale_down)
    
    color_map = np.zeros_like(resized_img, dtype='uint8')
    alpha_map = np.zeros((w, h), dtype='uint8')  # Alpha channel for transparency

    for i in range(len(x_cor)):
        x = int(x_cor[i])
        y = int(y_cor[i])
        if predict_results[i] == 0:
            continue
        color = colors[predict_results[i]]
        color_map[x:(x+patch_size), y:(y+patch_size), :] = color
        # Set alpha to 255 (opaque) wherever the prediction result is not 0 (white)
        if np.any(np.array(color) != [255, 255, 255]):
            alpha_map[x:(x+patch_size), y:(y+patch_size)] = 255

    # Blur the color map to smooth the transitions
    color_map = cv2.GaussianBlur(color_map, (151, 151), 0)
    alpha_map = cv2.GaussianBlur(alpha_map, (151, 151), 0)
    
    # Combine the images using the alpha map
    foreground = cv2.bitwise_and(color_map, color_map, mask=alpha_map)
    background = cv2.bitwise_and(resized_img, resized_img, mask=cv2.bitwise_not(alpha_map))
    combined_img = cv2.add(foreground, background)
    
    return combined_img

def heat_map(svs_path, output_path, df_inst):
    oslide = openslide.OpenSlide(svs_path)
    level_max = oslide.level_count - 1
    if level_max > 4:
        level_max = 4
    w, h = oslide.level_dimensions[level_max]
    scale_down = oslide.level_downsamples[level_max]
    wsi = oslide.read_region((0,0), level_max, (w, h)).convert('RGB')
    wsi = np.array(wsi)
    wsi_cors = list(df_inst['patch'])
    wsi_cors = [[int(i.split('_')[0]),int(i.split('_')[1])] for i in wsi_cors]
    predict_results = list(df_inst['pred'])
    combined_img = visualization_subtype(wsi, wsi_cors, predict_results, patch_size, scale_down)
    cv2.imwrite(output_path, combined_img)


if __name__ == "__main__":

    svs_path = '/home/z/zeyugao/dataset/TCGA-RCC/37d08405-fd8f-4a14-8327-4afe52fd8d8d/TCGA-B0-4945-01Z-00-DX1.590b650c-c9cb-4601-886c-fde0ccd9b90d.svs'
    npy_path = '/home/z/zeyugao/dataset/WSIData/TCGARenal/res50/TCGA-B0-4945-01Z-00-DX1.590b650c-c9cb-4601-886c-fde0ccd9b90d_1_512.npy'
    sp_path = '/home/z/zeyugao/dataset/WSIData/TCGARenal/sp_n16_c50_2048/TCGA-B0-4945-01Z-00-DX1.590b650c-c9cb-4601-886c-fde0ccd9b90d_1_512.npy'
    ckpt_path = '/home/z/zeyugao/SMMILe/ckpt/smmile_renal/s_0_checkpoint_best.pt'
    output_path = './TCGA-B0-4945_viz.png'
    model_type = 'smmile'
    model_size = 'small'
    drop_out = True
    drop_rate = 0.25
    n_classes = 3
    fea_dim = 1024
    n_refs = 3
    patch_size=2048
    inst_refinement = True

    patient_result, df_inst  = main_eval(npy_path, sp_path, model_type, model_size, drop_out, drop_rate, n_classes, fea_dim, patch_size, n_refs, ckpt_path, inst_refinement)
    print('ccRCC: 0; pRCC: 1; chRCC:2')
    print(patient_result)
    print(df_inst.head())
    heat_map(svs_path, output_path, df_inst)





