from __future__ import print_function
import argparse
import torch
import pdb
import os
import pandas as pd
from utils.utils import *
from datasets.dataset_nic import Generic_MIL_SP_Dataset as NIC_MIL_SP_Dataset
import h5py
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default='/home3/gzy/Renal/feature_resnet',
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results_renal/ablation',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default='renal_subtyping_smmile_res50_1512_5fold_s1',
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['smmile','smmile_single'], default='smmile', 
                    help='type of model (default: smmile')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--drop_rate', type=float, default=0.25,
                    help='drop_rate for official dropout')
parser.add_argument('--k', type=int, default=5, help='number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['renal_subtype_yfy','camelyon','renal_subtype','lung_subtype','ovarian_subtype'])
parser.add_argument('--fea_dim', type=int, default=1024,
                     help='the original dimensions of patch embedding')
parser.add_argument('--n_classes', type=int, default=3,
                     help='the number of types')

### smmile specific options

parser.add_argument('--drop_with_score', action='store_true', default=False,
                     help='enable weighted drop')
parser.add_argument('--D', type=int, default=4,
                     help='drop out times D')
parser.add_argument('--data_sp_dir', type=str, default=None, 
                    help='data directory of sp')
parser.add_argument('--superpixel', action='store_true', default=False,
                     help='enable superpixel sampling')
parser.add_argument('--sp_smooth', action='store_true', default=False,
                     help='enable superpixel average smooth')
parser.add_argument('--G', type=int, default=4,
                     help='one sample split to G')
parser.add_argument('--ref_start_epoch', type=int, default=75,
                     help='the inst loss back-propagation is start on this epoch')
parser.add_argument('--inst_refinement', action='store_true', default=False,
                     help='enable instance-level refinement')
parser.add_argument('--inst_rate', type=float, default=0.01,
                    help='sample rate for inst_refinement')
parser.add_argument('--n_refs', type=int, default=3,
                     help='the number of refinement layers')
parser.add_argument('--mrf', action='store_true', default=False,
                     help='enable MRF constraint for refinement')
parser.add_argument('--tau', type=float, default=1,
                     help='controling smoothness of mrf')


args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# args.save_exp_code=args.models_exp_code
args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.models_exp_code)+str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

conf_file = 'experiment_' + '_'.join(args.models_exp_code.split('_')[:-1])+'.txt'
fr = open(os.path.join(args.models_dir,conf_file),'r')
conf = eval(fr.read())
fr.close()

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir
    
assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

# update configs
args.task=conf['task']
args.fea_dim=conf['fea_dim']
args.model_type=conf['model_type']
args.model_size=conf['model_size']
args.drop_out=conf['use_drop_out']
args.drop_with_score=conf['drop_with_score']
args.D=conf['D']
args.superpixel=conf['superpixel']
args.G=conf['G']
if 'sp_smooth' in conf:
    args.sp_smooth=conf['sp_smooth']
args.inst_refinement=conf['inst_refinement']
args.n_refs=conf['n_refs']
args.inst_rate=conf['inst_rate']
if 'mrf' in conf:
    args.mrf=conf['mrf']
if 'tau' in conf:
    args.tau=conf['tau']

settings = {'task': args.task,
        'split': args.split,
        'save_dir': args.save_dir, 
        'models_dir': args.models_dir,
        'model_type': args.model_type,
        'model_size': args.model_size,
        'drop_out': args.drop_out,
        'drop_rate': args.drop_rate,
        'drop_with_score':args.drop_with_score,
        'superpixel':args.superpixel,
        'sp_smooth': args.sp_smooth,
        'inst_refinement':args.inst_refinement
        }

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'camelyon':
    args.n_classes=2
    args.patch_size=512
    if args.model_type in ['smmile_single']:
            dataset = NIC_MIL_SP_Dataset(csv_path = 'dataset_csv/camelyon_npy.csv',
                                data_dir = os.path.join(args.data_root_dir),
                                data_mag = '0_512',
                                sp_dir = os.path.join(args.data_sp_dir),
                                size = 512,
                                shuffle = False, 
                                seed = 10, 
                                print_info = True,
                                label_dict = {'normal':0, 'tumor':1},
                                patient_strat= False,
                                ignore=[])
elif args.task == 'renal_subtype':
    args.n_classes=3
    args.patch_size=2048
    if args.model_type in ['smmile']:
            dataset = NIC_MIL_SP_Dataset(csv_path = 'dataset_csv/renal_subtyping_npy.csv',
                                data_dir = os.path.join(args.data_root_dir),
                                data_mag = '1_512',
                                sp_dir = os.path.join(args.data_sp_dir),
                                size = 2048,
                                shuffle = False, 
                                seed = 10, 
                                print_info = True,
                                label_dict = {'ccrcc':0, 'prcc':1, 'chrcc':2},
                                patient_strat= False,
                                ignore=[])
elif args.task == 'renal_subtype_yfy':
    args.n_classes=4
    args.patch_size=1024
    if args.model_type in ['smmile']:
            dataset = NIC_MIL_SP_Dataset(csv_path = 'dataset_csv/renal_subtyping_yfy_npy.csv',
                                data_dir = os.path.join(args.data_root_dir),
                                data_mag = '0_1024',
                                sp_dir = os.path.join(args.data_sp_dir),
                                size = 1024,
                                shuffle = False, 
                                seed = 10, 
                                print_info = True,
                                label_dict = {'ccrcc':0, 'prcc':1, 'chrcc':2, 'rocy':3},
                                patient_strat= False,
                                ignore=[])
elif args.task == 'lung_subtype':
    args.n_classes=2
    args.patch_size=2048 #2048
    if args.model_type in ['smmile']:
            dataset = NIC_MIL_SP_Dataset(csv_path = 'dataset_csv/lung_subtyping_npy.csv',
                                data_dir = os.path.join(args.data_root_dir),
                                data_mag = '1_512', #1_512
                                sp_dir = os.path.join(args.data_sp_dir),
                                size = 2048, #2048
                                shuffle = False, 
                                seed = 10, 
                                print_info = True,
                                label_dict = {'luad':0, 'lusc':1},
                                patient_strat= False,
                                ignore=[])
            
elif args.task == 'ovarian_subtype':
    args.n_classes=5
    args.patch_size=512
    if args.model_type in ['smmile']:
            dataset = NIC_MIL_SP_Dataset(csv_path = 'dataset_csv/ovarian_subtyping_npy.csv',
                                data_dir = os.path.join(args.data_root_dir),
                                data_mag = '0_512',
                                sp_dir = os.path.join(args.data_sp_dir),
                                task = args.task,
                                size = 512,
                                shuffle = False, 
                                seed = 10, 
                                print_info = True,
                                label_dict = {'HGSC':0, 'EC':1, 'CC':2, 'LGSC':3, 'MC':4},
                                patient_strat= False,
                                ignore=[])
            
else:
    print(args.task)
    print(args.model_type)
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)

ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint_best.pt'.format(fold)) for fold in folds]
# ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    all_inst_auc = []
    all_inst_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        patient_results, test_error, auc, inst_auc, inst_acc, df, df_inst  = eval_(split_dataset, args, ckpt_paths[ckpt_idx])
        
        all_results.append(patient_results)
        all_auc.append(auc)
        all_inst_auc.append(inst_auc)
        all_inst_acc.append(inst_acc)
        all_acc.append(1-test_error)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)
        df_inst.to_csv(os.path.join(args.save_dir, 'smmile_inst_fold_{}_inst.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc, 'test_iauc': all_inst_auc, 'test_iacc':all_inst_acc})
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
