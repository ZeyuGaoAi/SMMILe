import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_sqmil import SQMIL_NIC
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, classification_report
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

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
   
    if model_type == 'sqmil_nic':
        model = SQMIL_NIC(**model_dict)
    elif model_type == 'sqmil_nic_single':
        model = SQMIL_NIC_SINGLE(**model_dict)
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
        score, Y_prob, Y_hat, ref_score, results_dict = model(feaetures_nic, mask, sp, adj,label=np.array([]), 
                                                              instance_eval=inst_refinement)

        Y_prob = Y_prob[0]

        inst_score = ref_score.detach().cpu().numpy()
        inst_pred = torch.argmax(ref_score, dim=1).detach().cpu().numpy()

        cor_h, cor_w = np.where(mask==1)
        coords = coords_nic[cor_h, cor_w]
        all_patches = [os.path.join("%s_%s_%s" % (int(coords[i][0]), int(coords[i][1]), patch_size)) for i in range(len(coords))]

        probs = Y_prob.cpu().numpy()
        preds = Y_hat.item()
    results_dict = {}
    for c in range(n_classes):
        results_dict['class_{}'.format(c)] = probs[c]
    inst_results_dict = {'filename':all_patches}
    for c in range(n_classes+1):
        if c == n_classes:
            inst_results_dict.update({'normal': inst_score[:,c]})
        else:
            inst_results_dict.update({'class_{}'.format(c): inst_score[:,c]})
    df_inst = pd.DataFrame(inst_results_dict)
    
    return results_dict, df_inst


def main_eval(npy_path,sp_path, model_type, model_size, drop_out, drop_rate, n_classes, fea_dim,patch_size, n_refs, ckpt_path, inst_refinement):
    if not os.path.exists(ckpt_path):
        ckpt_path=ckpt_path.replace('_best.pt','.pt')
    print(ckpt_path)
    model = initiate_model(model_type, model_size, drop_out, drop_rate, n_classes, fea_dim, n_refs, ckpt_path)
    print(npy_path)
    data = np.load(npy_path, allow_pickle=True)[()]
    sp_data = np.load(sp_path, allow_pickle=True)[()]
    results_dict, df_inst = summary(model, data, patch_size,sp_data,n_classes, model_type, inst_refinement)

    return results_dict, df_inst



npy_path = '/home3/gzy/Gastric/feature_resnet/TCGA-BR-8367-01Z-00-DX1.25d1e251-923d-46f5-9a6b-2f6207def599_1_512.npy'
sp_path = '/home3/gzy/Gastric/sp_n9_c50_2048/TCGA-BR-8367-01Z-00-DX1.25d1e251-923d-46f5-9a6b-2f6207def599_1_512.npy'
model_type = 'sqmil_nic'
model_size = 'small'
drop_out = True
drop_rate = 0.25
n_classes = 3
fea_dim = 1024
n_refs = 3
patch_size=2048
inst_refinement = True
ckpt_path = '../../SQMILM/results_gastric/ablation/gastric_subtyping_sqmil_ml_biloss_drop_spg10_ref3mrf_w_s1_s1/s_0_checkpoint.pt'

patient_result, df_inst  = main_eval(npy_path,sp_path, model_type, model_size, drop_out, drop_rate, n_classes, fea_dim, patch_size,n_refs, ckpt_path, inst_refinement)
print(patient_result)
print(df_inst)





