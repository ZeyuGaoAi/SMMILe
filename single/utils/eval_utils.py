import os
import torch
import numpy as np
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize

from models.model_smmile import SMMILe, SMMILe_SINGLE

def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {'dropout': args.drop_out, 'drop_rate': args.drop_rate, 'n_classes': args.n_classes, 
                  'fea_dim': args.fea_dim, "size_arg": args.model_size, 'n_refs': args.n_refs}
   
    if args.model_type == 'smmile':
        model = SMMILe(**model_dict)
    elif args.model_type == 'smmile_single':
        model = SMMILe_SINGLE(**model_dict)
    else:
        raise NotImplementedError

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
    if not os.path.exists(ckpt_path):
        ckpt_path=ckpt_path.replace('_best.pt','.pt')
    print(ckpt_path)
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, inst_auc,inst_acc, df, df_inst, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    print('inst auc and acc: ', inst_auc,inst_acc)
    return patient_results, test_error, auc, inst_auc,inst_acc, df, df_inst

def summary(model, loader, args):
    
    n_classes = args.n_classes
    model_type = args.model_type
    inst_refinement = args.inst_refinement
    superpixel = args.superpixel
    sp_smooth = args.sp_smooth
    G = args.G
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    all_silde_ids = []
    patient_results = {}
    
    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device, non_blocking=True)

            slide_id = slide_ids.iloc[batch_idx]

            mask = cors[1]
            sp = cors[2]
            adj = cors[3]

            score, Y_prob, Y_hat, ref_score, results_dict = model(data, mask, sp, adj, label, 
                                                                  superpixels = superpixel,
                                                                  sp_smooth = sp_smooth,
                                                                  group_numbers = G,
                                                                  instance_eval=inst_refinement)

            Y_prob = Y_prob[0]
            if inst_label!=[] and sum(inst_label)!=0:
                inst_label = [1 if patch_label>0 else patch_label for patch_label in inst_label] # normal vs cancer, keep -1
                all_inst_label += inst_label

                if not inst_refinement:
                    if model_type == 'smmile':
                        inst_score = score[:,Y_hat].detach().cpu().numpy()[:]
                    elif model_type == 'smmile_single':
                        inst_score = score[:, 0].detach().cpu().numpy()[:]
                    else:
                        inst_score = score[:,Y_hat].detach().cpu().numpy()[:]
                    inst_score=inst_score.squeeze()
                    inst_score = list((inst_score-inst_score.min())/max((inst_score.max()-inst_score.min()), 1e-10))
                    inst_pred = [1 if i>0.5 else 0 for i in inst_score]
                else:
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
                all_inst_score += inst_score
                all_inst_pred += inst_pred
                
                cor_h, cor_w = np.where(mask==1)
                coords = cors[0][cor_h, cor_w]

            if inst_label!=[] and sum(inst_label)!=0:
                all_silde_ids += [os.path.join(str(slide_ids[batch_idx]), "%s_%s_%s.png" %
                                               (int(coords[i][0]), int(coords[i][1]), args.patch_size)) 
                                  for i in range(len(coords))]
            
            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()
            all_preds[batch_idx] = Y_hat.item()

            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
            error = calculate_error(Y_hat, label)
            test_error += error

    test_error /= len(loader)

    # excluding -1 
    all_inst_label_sub = np.array(all_inst_label)
    all_inst_score_sub = np.array(all_inst_score)
    all_inst_pred_sub = np.array(all_inst_pred)
    
    all_inst_score_sub = all_inst_score_sub[all_inst_label_sub!=-1]
    all_inst_pred_sub = all_inst_pred_sub[all_inst_label_sub!=-1]
    all_inst_label_sub = all_inst_label_sub[all_inst_label_sub!=-1]   
    
    
    inst_auc = roc_auc_score(all_inst_label_sub, all_inst_score_sub)
    print("inst level aucroc: %f" % inst_auc)
    inst_acc = accuracy_score(all_inst_label_sub, all_inst_pred_sub)
    
    print(classification_report(all_inst_label_sub, all_inst_pred_sub, zero_division=1))

    if model_type == 'smmile_single':
        auc_score = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        aucs = []
        
        if n_classes == 2:
            binary_labels = label_binarize(all_labels, classes=[0,1,2])[:,:n_classes]
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
#     print(all_inst_score)
    df_inst = pd.DataFrame([all_silde_ids, all_inst_label, all_inst_score,all_inst_pred]).T
    df_inst.columns = ['filename', 'label', 'prob', 'pred']
#     print(df_inst)
    
    return patient_results, test_error, auc_score, inst_auc, inst_acc, df, df_inst, acc_logger
