import csv
import json
import os.path
import pprint

from valentine import valentine_match, valentine_metrics


# 从JSON文件加载数据
def mk_path(folder_path):
    if not os.path.exists(folder_path):
        print(folder_path)
        os.makedirs(folder_path)


def Ground_truth(path, matches, save_path,method):
    with open(path) as f:
        data = json.load(f)
    # 提取source_column和target_column
    matches_part = data['matches']
    ground_truth = [(match['source_column'], match['target_column']) for match in matches_part]
    print(ground_truth)
    """
    ground_truth = [('Cited by', 'Cited by'),
                    ('Authors', 'Authors'),
                    ('EID', 'EID')]
    """
    mk_path(save_path)
    metrics = valentine_metrics.all_metrics(matches, ground_truth)
    save_pairs(save_path, matches,method)
    save_metrics(save_path, metrics,method)
    pp = pprint.PrettyPrinter(indent=4)

    pp.pprint(matches)
    print("\nAccording to the ground truth:")
    pp.pprint(metrics)


def save_pairs(path, matches,method):
    header = ['Column1', 'Column2', 'Similarity']
    filename = os.path.join(path, method+'_pair_result.csv')
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(matches)


def save_metrics(path, metrics,method):
    filename = os.path.join(path, method+'_metrics.csv')
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Key', 'Value'])
        for key, value in metrics.items():
            writer.writerow([key, value])

