import os
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(filename):
    # 读取CSV文件
    data = pd.read_csv(filename)
    data = data[data['label'] != -1]
    true_labels = data['label']
    probabilities = data['prob']
    predictions = data['pred']
    
    # 计算各种指标
    auc = roc_auc_score(true_labels, probabilities)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    return auc, accuracy, precision, recall, f1

def process_files(directory):
    results = []
    
    # 遍历文件夹
    for file in os.listdir(directory):
        if file.endswith('inst.csv'):
            filepath = os.path.join(directory, file)
            try:
                metrics = calculate_metrics(filepath)
                results.append([file] + list(metrics))
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    
    # 将结果保存到新的CSV文件
    results_df = pd.DataFrame(results, columns=['Filename', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    results_df.to_csv(os.path.join(directory,'instance_metrics_summary.csv'), index=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    process_files(directory_path)
