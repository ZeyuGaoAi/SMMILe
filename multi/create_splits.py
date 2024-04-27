import os
import pandas as pd
import numpy as np

task = 'gastric_subtype'
csv_path = './dataset_csv/gastric_subtyping_npy.csv'
k_folds = 5
split_dir = 'splits/'+ str(task) + '_{}'.format(int(100))
os.makedirs(split_dir, exist_ok=True)

dataset = pd.read_csv(csv_path)
label_set = np.unique(dataset['label'])

# create fold-label
dataset_new = []
for label in label_set:
    dataset_class = dataset[dataset['label'] == label].copy()
    dataset_class.sample(frac=1.)
    dataset_class['k_fold'] = [i%k_folds for i in range(dataset_class.shape[0])]
    dataset_new.append(dataset_class)
dataset_new = pd.concat(dataset_new, axis=0)
dataset_new['slide_id'] = dataset_new['slide_id'].apply(lambda x: str(x))

# split
for i in range(k_folds):
    train_val_set = dataset_new[dataset_new['k_fold']!=i].reset_index()
    train_set = []
    val_set = []
    for label in label_set:
        train_val_class = train_val_set[train_val_set['label']==label].copy()
        train_class = train_val_class.sample(frac=0.9)
        train_set.append(train_class)
        val_class = train_val_class[~train_val_class['slide_id'].isin(list(train_class['slide_id']))]
        val_set.append(val_class)
    train_set = pd.concat(train_set,ignore_index=True)
    val_set = pd.concat(val_set,ignore_index=True)
    test_set = dataset_new[dataset_new['k_fold']==i].reset_index()
    df_split = [train_set['slide_id'], val_set['slide_id'], test_set['slide_id']]
    df_split = pd.concat(df_split, ignore_index=True, axis=1)
    df_split.columns = ['train', 'val', 'test']
    df_split.to_csv(os.path.join(split_dir, 'splits_{}.csv'.format(i)))