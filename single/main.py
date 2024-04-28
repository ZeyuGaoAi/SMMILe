from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_nic import Generic_MIL_SP_Dataset as NIC_MIL_SP_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

import yaml

def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_iauc = []
    all_val_iauc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc, test_iauc, val_iauc = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        all_test_iauc.append(test_iauc)
        all_val_iauc.append(val_iauc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc,
        'test_iauc':all_test_iauc, 'val_iauc': all_val_iauc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--config', type=str, default='config.yaml',
                     help='the path to config file')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--drop_rate', type=float, default=0.25,
                    help='drop_rate for official dropout')
parser.add_argument('--bag_loss', type=str, choices=['ce', 'bce','bibce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['smmile','smmile_single'], default='smmile', 
                    help='type of model (default: smmile')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model')
parser.add_argument('--task', type=str, choices=['camelyon','renal_subtype','lung_subtype','ovarian_subtype'])
parser.add_argument('--fea_dim', type=int, default=1024,
                     help='the original dimensions of patch embedding')
parser.add_argument('--models_dir', type=str, default=None,
                     help='the path to ckpt')
parser.add_argument('--n_classes', type=int, default=3,
                     help='the number of types')
parser.add_argument('--n_refs', type=int, default=3,
                     help='the times of refinement')

### smmile specific options
parser.add_argument('--drop_with_score', action='store_true', default=False,
                     help='enable weighted drop')
parser.add_argument('--data_sp_dir', type=str, default=None, 
                    help='data directory of sp')
parser.add_argument('--superpixel', action='store_true', default=False,
                     help='enable superpixel sampling')
parser.add_argument('--G', type=int, default=4,
                     help='one sample split to G')
parser.add_argument('--ref_start_epoch', type=int, default=75,
                     help='the inst loss back-propagation is start on this epoch')
parser.add_argument('--inst_refinement', action='store_true', default=False,
                     help='enable instance-level refinement')
parser.add_argument('--inst_rate', type=float, default=0.01,
                    help='sample rate for inst_refinement')
parser.add_argument('--mrf', action='store_true', default=False,
                     help='enable MRF constraint for refinement')
parser.add_argument('--tau', type=float, default=1,
                     help='controling smoothness of mrf')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
# read config
with open(args.config) as fp:
    cfg = yaml.load(fp, Loader=yaml.CLoader)
args.seed=cfg['seed']
args.log_data = cfg['log_data']
args.testing = cfg['testing']
args.reg= cfg['reg']
args.n_classes=cfg['n_classes']
args.early_stopping=cfg['early_stopping']
args.k = cfg['k']
args.k_start = cfg['k_start']
args.k_end = cfg['k_end']
args.task = cfg['task']
args.data_root_dir=cfg['data_root_dir']
args.max_epochs = cfg['max_epochs']
args.results_dir = cfg['results_dir']
args.lr = cfg['lr']
args.exp_code = cfg['exp_code']
args.label_frac = cfg['label_frac']
args.bag_loss = cfg['bag_loss']
args.model_type = cfg['model_type']
args.model_size = cfg['model_size']
args.drop_out = cfg['drop_out']
args.drop_rate = cfg['drop_rate']
args.weighted_sample = cfg['weighted_sample']
args.fea_dim = cfg['fea_dim']
args.opt = cfg['opt']
args.models_dir = cfg['models_dir']
### SMMILe specific options
args.consistency = cfg['consistency']
args.drop_with_score = cfg['drop_with_score']
args.superpixel = cfg['superpixel']
args.data_sp_dir = cfg['data_sp_dir']
args.G = cfg['G']
args.inst_refinement = cfg['inst_refinement']
args.inst_rate = cfg['inst_rate']
args.n_refs = cfg['n_refs']
args.ref_start_epoch = cfg['ref_start_epoch']
args.mrf = cfg['mrf']
args.tau = cfg['tau']
####
    
seed_torch(args.seed)

settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'fea_dim': args.fea_dim,
            'opt': args.opt,
            'models_dir': args.models_dir}

settings.update({'consistency': args.consistency})
settings.update({'drop_with_score': args.drop_with_score})
settings.update({'superpixel': args.superpixel})
settings.update({'data_sp_dir': args.data_sp_dir})
settings.update({'G': args.G})
settings.update({'inst_refinement': args.inst_refinement})
settings.update({'inst_rate': args.inst_rate})
settings.update({'n_refs':args.n_refs})
settings.update({'ref_start_epoch': args.ref_start_epoch})
settings.update({'mrf': args.mrf})
settings.update({'tau': args.tau})
    

print('\nLoad Dataset')

if args.task == 'camelyon':
    args.n_classes=2
    if args.model_type in ['smmile_single']:
            dataset = NIC_MIL_SP_Dataset(csv_path = 'dataset_csv/camelyon_npy.csv',
                                data_dir = os.path.join(args.data_root_dir),
                                data_mag = '0_512',
                                sp_dir = os.path.join(args.data_sp_dir),
                                task = args.task,
                                size = 512,
                                shuffle = False, 
                                seed = 10, 
                                print_info = True,
                                label_dict = {'normal':0, 'tumor':1},
                                patient_strat= False,
                                ignore=[])
elif args.task == 'renal_subtype':
    args.n_classes=3
    if args.model_type in ['smmile']:
            dataset = NIC_MIL_SP_Dataset(csv_path = 'dataset_csv/renal_subtyping_npy.csv',
                                data_dir = os.path.join(args.data_root_dir),
                                data_mag = '1_512',
                                sp_dir = os.path.join(args.data_sp_dir),
                                task = args.task,
                                size = 2048,
                                shuffle = False, 
                                seed = 10, 
                                print_info = True,
                                label_dict = {'ccrcc':0, 'prcc':1, 'chrcc':2},
                                patient_strat= False,
                                ignore=[])
            
elif args.task == 'lung_subtype':
    args.n_classes=2
    if args.model_type in ['smmile']:
            dataset = NIC_MIL_SP_Dataset(csv_path = 'dataset_csv/lung_subtyping_npy.csv',
                                data_dir = os.path.join(args.data_root_dir),
                                data_mag = '1_512',
                                sp_dir = os.path.join(args.data_sp_dir),
                                task = args.task,
                                size = 2048,
                                shuffle = False, 
                                seed = 10, 
                                print_info = True,
                                label_dict = {'luad':0, 'lusc':1},
                                patient_strat= False,
                                ignore=[])
elif args.task == 'ovarian_subtype':
    args.n_classes=5
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
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))    
    
if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


