import csv
import json
import os.path
import pprint

from valentine import valentine_match, valentine_metrics


def mk_path(folder_path):
    if not os.path.exists(folder_path):
        print(folder_path)
        os.makedirs(folder_path)


def Ground_truth(eval_path,path, matches, save_path,method):
    with open(path) as f:
        data = json.load(f)

    matches_part = data['matches']
    tables = [fn for fn in os.listdir(eval_path) if '.csv' in fn]
    if tables[0] == matches_part[0]["source_table"]+'.csv':
        ground_truth = [(match['source_column'], match['target_column']) for match in matches_part]

    else:
        ground_truth = [(match['target_column'], match['source_column']) for match in matches_part]
    print(ground_truth)
    mk_path(save_path)
    metrics = valentine_metrics.all_metrics(matches, ground_truth)
    save_pairs(save_path, matches,method)
    save_metrics(save_path, metrics,method)
    pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)
    pp.pprint(matches) 
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

