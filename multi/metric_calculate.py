import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(filename):
    # 读取CSV文件
    data = pd.read_csv(filename)
    
    # 过滤掉 true_labels 为 -1 的行
    filtered_data = data[data['label'] != -1]
    true_labels = filtered_data['label']
    # 获取概率列并转化为numpy数组
    probabilities = filtered_data.filter(like='prob').values
    # 使用概率最大值对应的索引作为预测结果
    predictions = np.argmax(probabilities, axis=1)
    
    # 多分类处理：标签二值化
    classes = np.unique(true_labels)
    true_labels_binarized = label_binarize(true_labels, classes=classes)
    
    # # AUC计算，处理多类情况，使用one-vs-rest方法
    # if len(classes) > 2:
    #     auc = roc_auc_score(true_labels_binarized, probabilities, multi_class='ovr', average='macro')
    # else:
    #     # 二分类不需要binarize标签
    #     auc = roc_auc_score(true_labels, probabilities[:, 1])  # 假设正类标签在第二列

    print(classes)

    inst_aucs = []
    for class_idx in classes:

        class_idx = int(class_idx)

        probabilities_1class = probabilities[:, class_idx]
        true_labels_binarized_1class = true_labels_binarized[:, class_idx]

        true_labels_binarized_1class = true_labels_binarized_1class[probabilities_1class!=-1]
        probabilities_1class = probabilities_1class[probabilities_1class!=-1]

        inst_aucs.append(roc_auc_score(true_labels_binarized_1class, probabilities_1class))
    
    auc = np.nanmean(np.array(inst_aucs))
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro', zero_division=1)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=1)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=1)
    
    return auc, accuracy, precision, recall, f1

def process_files(directory):
    results = []
    
    # 遍历文件夹
    for file in os.listdir(directory):
        if file.endswith('inst.csv'):
            filepath = os.path.join(directory, file)
            # try:
            metrics = calculate_metrics(filepath)
            results.append([file] + list(metrics))
            # except Exception as e:
            #     print(f"Error processing file {file}: {e}")
    
    # 将结果保存到新的CSV文件
    results_df = pd.DataFrame(results, columns=['Filename', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    results_df.to_csv(os.path.join(directory,'instance_metrics_summary.csv'), index=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    process_files(directory_path)
