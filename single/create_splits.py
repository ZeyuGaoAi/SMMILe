import os
import pandas as pd
import numpy as np

task = 'renal_subtype'
csv_path = './dataset_csv/renal_subtyping_npy.csv'
k_folds = 5
split_dir = 'splits/' + str(task) + '_{}'.format(int(100/k_folds*(k_folds-1)))
os.makedirs(split_dir, exist_ok=True)

dataset = pd.read_csv(csv_path)

# for tcga
dataset['case_id'] = dataset['slide_id'].map(lambda x: x[:12]) # TCGA-HE-A5NF

# for ih-rcc
# dataset['case_id'] = dataset['slide_id'].map(lambda x: x[:10]) # B201007978

label_set = np.unique(dataset['label'])
dataset['k_fold'] = -1
case_id_to_fold = {}
for label in label_set:
    dataset_sub = dataset[dataset['label'] == label]
    dataset_sub = dataset_sub.reset_index(drop=True)
    # Shuffle the dataset by case_id
    case_ids = dataset_sub['case_id'].unique()
    np.random.shuffle(case_ids)

    # Assign each case_id to a fold
    case_id_to_fold.update({case_id: i % k_folds for i, case_id in enumerate(case_ids)})
    
    # Add the fold assignment to the dataset
dataset['k_fold'] = dataset['case_id'].map(case_id_to_fold)

# Split the dataset
for i in range(k_folds):
    train_val_set = dataset[dataset['k_fold'] != i].reset_index(drop=True)
    test_set = dataset[dataset['k_fold'] == i].reset_index(drop=True)
    
    train_set = []
    val_set = []
    for label in label_set:
        train_val_class = train_val_set[train_val_set['label'] == label].copy()
        case_ids = train_val_class['case_id'].unique()
        np.random.shuffle(case_ids)
        
        # Split the cases into train and validation sets
        split_idx = int(len(case_ids) * 0.9)
        train_case_ids = case_ids[:split_idx]
        val_case_ids = case_ids[split_idx:]
        
        train_set.append(train_val_class[train_val_class['case_id'].isin(train_case_ids)])
        val_set.append(train_val_class[train_val_class['case_id'].isin(val_case_ids)])
    
    train_set = pd.concat(train_set, ignore_index=True)
    val_set = pd.concat(val_set, ignore_index=True)
    
    df_split = pd.DataFrame({
        'train': train_set['slide_id'],
        'val': val_set['slide_id'],
        'test': test_set['slide_id']
    })

    # df_split['val'] = df_split['val'].astype('Int64')
    # df_split['test'] = df_split['test'].astype('Int64')

    print(df_split)
    
    df_split.to_csv(os.path.join(split_dir, 'splits_{}.csv'.format(i)), index=False)
