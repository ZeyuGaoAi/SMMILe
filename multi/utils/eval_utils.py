import numpy as np

import torch
import torch.nn as nn
from models.model_smmile import SMMILe
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_curve, accuracy_score, classification_report
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize

def initiate_model(args, ckpt_path):
    print('Init Model')    
    
    model_dict = {'dropout': args.drop_out, 'drop_rate': args.drop_rate,
                  'multi_label': args.multi_label, 'n_classes': args.n_classes, 
                  'fea_dim': args.fea_dim, "size_arg": args.model_size,
                  'n_refs': args.n_refs}

    model = SMMILe(**model_dict)

    print_network(model)

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

def eval_(dataset, args, ckpt_path):
    print(ckpt_path)
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _, df_inst = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df, df_inst

def summary(model, loader, args):
    n_classes = args.n_classes
    model_type = args.model_type
    multi_label = args.multi_label
    inst_refinement = args.inst_refinement
    
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    all_inst_label = []
    all_silde_ids = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_pos = []
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros((len(loader), n_classes))
    all_preds = np.zeros((len(loader), n_classes))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
        
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        if multi_label:
            label = label.to(device)
            index_label = torch.nonzero(label.squeeze()).to(device)
        else:
            index_label = label.to(device)
            label = torch.zeros(n_classes)
            label[index_label.long()] = 1
            label = label.to(device)
        
        if inst_label!=[]:
            all_inst_label += inst_label
            
        
        slide_id = slide_ids.iloc[batch_idx]
            
        data = data.to(device)
        with torch.no_grad():
            mask = cors[1]
            sp = cors[2]
            adj = cors[3]
            score, Y_prob, Y_hat, ref_score, results_dict = model(data, mask, sp, adj, index_label, 
                                                                  instance_eval=inst_refinement)

#         Y_prob=torch.mean(torch.stack(Y_prob),dim=0)
        Y_prob = Y_prob[0]
                
        if inst_label!=[]:
            label_hot = label_binarize(inst_label, classes=[i for i in range(n_classes+1)])
            inst_probs_tmp = [] 
            inst_preds_tmp = []
            if not inst_refinement:
                inst_score = score.detach().cpu().numpy()

                # max-min standard all_inst_score and all_inst_score_pos
                for class_idx in range(n_classes):
                    if class_idx not in index_label:
                        inst_score[:,class_idx] = [-1]*len(inst_label)
                        continue

                    inst_score_one_class = inst_score[:,class_idx]
#                     if len(set(inst_score_one_class))>1:
                    inst_score_one_class = list((inst_score_one_class-inst_score_one_class.min())/
                                                max(inst_score_one_class.max()-inst_score_one_class.min(),1e-10))
                    inst_score_one_class = list(inst_score_one_class)
                    inst_score[:,class_idx] = inst_score_one_class

                    inst_probs[class_idx]+=inst_score_one_class
                    inst_probs_tmp.append(inst_score_one_class)                  

                    inst_preds[class_idx]+=[0 if i<0.5 else 1 for i in inst_score_one_class]
                    inst_preds_tmp.append([0 if i<0.5 else 1 for i in inst_score_one_class])

                    inst_binary_labels[class_idx]+=list(label_hot[:,class_idx])

                if inst_preds_tmp:
                    inst_preds_tmp = np.mean(np.stack(inst_preds_tmp), axis=0)
                else:
                    inst_preds_tmp = [0]*len(inst_label)
                inst_preds_tmp = [1 if i==0 else 0 for i in inst_preds_tmp]
                inst_preds[n_classes] += inst_preds_tmp
                inst_binary_labels[n_classes]+=list(label_hot[:,n_classes])

                if inst_probs_tmp:
                    neg_score = np.mean(np.stack(inst_probs_tmp), axis=0) #三类平均，越低越是neg
#                     if len(set(neg_score))>1:
                    neg_score = list((neg_score-neg_score.min())/max(neg_score.max()-neg_score.min(),1e-10))
#                     neg_score = list(neg_score)
                else:
                    neg_score = [0]*len(inst_label)



            else:
                inst_score = ref_score.detach().cpu().numpy()
                pos_score = score.detach().cpu().numpy() #final_score_sp

                # max-min standard all_inst_score and all_inst_score_pos
                for class_idx in range(n_classes):
#                     if class_idx not in Y_hat:
                    if class_idx not in index_label:
                        inst_score[:, class_idx] = -1
                        continue
                    inst_score_one_class = pos_score[:,class_idx]
#                     if len(set(inst_score_one_class))>1:
                    inst_score_one_class = list((inst_score_one_class-inst_score_one_class.min())/
                                                max(inst_score_one_class.max()-inst_score_one_class.min(),1e-10))

                    inst_probs_tmp.append(inst_score_one_class)


                if inst_probs_tmp:
                    neg_score = np.mean(np.stack(inst_probs_tmp), axis=0) #三类平均，越低越是neg
#                     if len(set(neg_score))>1:
                    neg_score = list((neg_score-neg_score.min())/max(neg_score.max()-neg_score.min(),1e-10))
#                     neg_score = list(neg_score)
                else:
                    neg_score = [0]*len(inst_label)
#                 neg_score = list((neg_score-neg_score.min())/(neg_score.max()-neg_score.min()))

#             print(len(inst_label),len(coords),inst_score.shape)
#             if len(inst_label)!=len(coords):
#                 print(data.shape,slide_ids[batch_idx])
            all_inst_score.append(inst_score)
            all_inst_score_neg += neg_score
        
            cor_h, cor_w = np.where(mask==1)
            coords = cors[0][cor_h, cor_w]
            all_silde_ids += [os.path.join(slide_ids[batch_idx], "%s_%s_%s.png" % (int(coords[i][0]), int(coords[i][1]),args.patch_size)) 
                              for i in range(coords.shape[0])]

            
        acc_logger.log(Y_hat, index_label)
        probs = Y_prob.cpu().numpy()
        Y_hat_one_hot = np.zeros(n_classes)
        Y_hat_one_hot[Y_hat.cpu().numpy()]=1
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.squeeze().detach().cpu().numpy()
        all_preds[batch_idx] = Y_hat_one_hot
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 
                                           'label': torch.nonzero(label.squeeze()).squeeze().cpu().numpy()}})
        error = calculate_error(Y_hat, label)
        test_error += error
    del data
    test_error /= len(loader)
    
    # calculate inst_auc and inst_acc   
    
    all_inst_score = np.concatenate(all_inst_score,axis=0)
    all_normal_score = [1-x for x in all_inst_score_neg] #转化为是normal类的概率 越高越好
    
    # get inst_pred inst_acc
    if not inst_refinement:
        inst_accs = []
        for i in range(n_classes+1):
            if len(inst_binary_labels[i])==0:
                continue
            inst_accs.append(accuracy_score(inst_binary_labels[i], inst_preds[i]))
            inst_acc = np.mean(inst_accs)
            print('class {}'.format(str(i)))
            print(classification_report(inst_binary_labels[i], inst_preds[i],zero_division=1))
    else:
        all_inst_pred = np.argmax(all_inst_score,axis=1)
        all_inst_pred = list(all_inst_pred)
  
        inst_acc = accuracy_score(all_inst_label, all_inst_pred)
        print(classification_report(all_inst_label, all_inst_pred,zero_division=1))
    
    # get inst_auc
    inst_aucs = []
    if not inst_refinement:
        for class_idx in range(n_classes):
            inst_score_sub = inst_probs[class_idx]
            if len(inst_score_sub)==0:
                continue
            fpr, tpr, _ = roc_curve(inst_binary_labels[class_idx], inst_score_sub)
            inst_aucs.append(calc_auc(fpr, tpr))
        fpr,tpr,_ = roc_curve(inst_binary_labels[n_classes], all_normal_score)
        inst_aucs.append(calc_auc(fpr, tpr))
    else:
        binary_inst_label = label_binarize(all_inst_label, classes=[i for i in range(n_classes+1)])
        for class_idx in range(n_classes+1):
            if class_idx in all_inst_label:
                inst_score_sub = all_inst_score[:,class_idx]
                fpr, tpr, _ = roc_curve(binary_inst_label[:, class_idx], inst_score_sub)
                inst_aucs.append(calc_auc(fpr, tpr))
            else:
                inst_aucs.append(float('nan'))
    inst_auc = np.nanmean(np.array(inst_aucs))


    aucs = []
    binary_labels = all_labels
    for class_idx in range(n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
        aucs.append(calc_auc(fpr, tpr))

    auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids}
    for c in range(args.n_classes):
        results_dict.update({'label_{}'.format(c): all_labels[:,c]})
        results_dict.update({'prob_{}'.format(c): all_probs[:,c]})
        results_dict.update({'pred_{}'.format(c): all_preds[:,c]})
    df = pd.DataFrame(results_dict)
    if not inst_refinement:
        all_inst_score = np.insert(all_inst_score, args.n_classes, values=all_normal_score, axis=1)
    inst_results_dict = {'filename':all_silde_ids,'label':all_inst_label}
    for c in range(args.n_classes+1):
        inst_results_dict.update({'prob_{}'.format(c): all_inst_score[:,c]})
    df_inst = pd.DataFrame(inst_results_dict)
    
    return patient_results, test_error, [auc_score,inst_auc,inst_acc], df, acc_logger, df_inst
