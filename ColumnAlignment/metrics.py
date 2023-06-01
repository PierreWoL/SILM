import json
import pprint

from valentine import valentine_match, valentine_metrics

# 从JSON文件加载数据

def Ground_truth(path,matches):
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
    metrics = valentine_metrics.all_metrics(matches, ground_truth)

    pp = pprint.PrettyPrinter(indent=4)

    print("\nAccording to the ground truth:")
    pp.pprint(metrics)
