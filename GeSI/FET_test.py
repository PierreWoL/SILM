import json
import os

import pandas as pd

from Metrics.dataGeneration import testMetrics

model_name = 'qwen'
for shot in [3]:
    for mode in [0, 1, 2,3,4,5]:
        if mode == 3:
            fold_name = "filter"
        elif mode == 0:
            fold_name = "s1"
        elif mode == 1:
            fold_name = "s2"
        elif mode == 2:
            fold_name = "s3"
        elif mode == 4:
            fold_name = "cutoff"
        elif mode == 5:
            fold_name = "cutoff_filter"
        for dataset in ["WDC"]:#, "GDS"
            json_FET_table = []
            json_FET_type = []
            print("START ITERATION", shot, mode, dataset)
            result_path = f"Result/{dataset}/Prompt0/{shot}/fet/{model_name}/{fold_name}/"
            print(result_path)
            with open(os.path.join(result_path, "FEET_results.jsonl"), "r", encoding='utf-8') as f:
                for line in f:
                    json_FET_table.append(json.loads(line)["id"])
                    json_FET_type.append(json.loads(line)["type"])
            GT_type = pd.read_csv(f"datasets/{dataset}/groundTruth.csv")

            """ 
            gt_type = []
            delete = []
            for index, i in enumerate(json_FET_table):
                matched = GT_type.loc[GT_type['fileName'] == i, 'class']
                # print(i, matched.iloc[0])
                if not matched.empty:
                    gt_type.append(matched.iloc[0])
                else:
                    print(f"[Warning] No match found for fileName: {i}")
                    delete.append(index)
            """
            gt_type = [GT_type.loc[GT_type['fileName'] == i, 'class'].iloc[0] for i in json_FET_table]
            #json_FET_type = [i for index, i in enumerate(json_FET_type) if index not in delete]
            print(len(json_FET_type), json_FET_type)
            print(len(gt_type),gt_type)
            # print(delete)
            overall_metric, overallR = testMetrics(gt_type, json_FET_type)
            p, r, f1 = overall_metric["bertscore"]
            data = [json_FET_table, gt_type, json_FET_type,
                    p, r, f1, overall_metric["cosine similarity"]]

            specific_df = pd.DataFrame(data).T
            specific_df.columns = ['fileName', 'groundTruth type', 'inferred type',
                                   'BERT score precision', 'BERT score recall',
                                   'BERT score F1', 'cosine similarity']

            df = pd.DataFrame.from_dict(overallR, orient='index')
            print(df)
            df.to_csv(result_path + "overallScore.csv")
            specific_df.to_csv(result_path + "specificScore.csv")
