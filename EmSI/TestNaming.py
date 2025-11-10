import os
import random
import pickle
import pandas as pd

from EndToEnd.EndToEnd import type_info, hierarchy
from Naming import GPTnaming,TableType
from TableCluster.tableClustering import  typeInference
from argparse import Namespace
import re

def read_clusters_tables(dataset, cluster):
    tables = []
    for table_name in cluster:
        table = pd.read_csv(f"datasets/{dataset}/Test/{table_name}.csv")
        tables.append(table)

    return tables




def testNaming(hp: Namespace,  groundtruth = False, format=4, sampling=None, sampling_number=2, header=True, table_names=None, instruction = None, shorts = None, AI_instruction=None):
    if groundtruth is False:
        cluster_dict = typeInference(hp.P1Embed, hp)["Agglomerative"]
        """
        name_dict = {row["table"]: row["name"] for index, row in
                     pd.read_csv(f"datasets/{hp.dataset}/naming.csv").iterrows()}
        cluster_dict_all = type_info(cluster_dict, hp.dataset, nameDict=name_dict)
        
        print("top level type number: ", len(cluster_dict), len(cluster_dict_all))
        del cluster_dict
        hierarchy(cluster_dict_all, hp, name_dict)
        print(cluster_dict_all)"""
        #with open(f"datasets/{hp.dataset}/naming", "rb") as f:
            #cluster_dict = pickle.load(open(f"datasets/{hp.dataset}"))
        results = pd.read_csv(f"result/P1/{hp.dataset}/All/{hp.P1Embed[:-4]}/purityCluster.csv", index_col=0)
    else:
        gt_df = pd.read_csv(f"datasets/{hp.dataset}/groundTruth.csv")
        gt_df["fileName"] = gt_df["fileName"].apply(lambda x: x[:-4])
        print(gt_df)
        cluster_dict = gt_df.groupby('superclass')['fileName'].apply(list).to_dict()
        if "gt_naming.csv" not in os.listdir(f"datasets/{hp.dataset}"):
            lowestLeveldict =  gt_df.groupby('superclass')['class'].apply(list).to_dict()
            results = pd.DataFrame(list(lowestLeveldict.items()), columns=["superclass", "class"]) # index=range(len(cluster_dict))
            results.set_index('superclass', inplace=True)
        else:
            results = pd.read_csv(f"datasets/{hp.dataset}/gt_naming.csv", index_col=0)
    new_col = 'Naming' + str(len(results.columns)+1)
    print(new_col)
    results[new_col] = ''
    task_element = ""
    if format == 2:
        task_element = "tables"
    elif format == 3:
        task_element = "tables with <sc> and </sc> marking the subject attributes"
    elif format == 4:
        task_element = "subject attributes of tables"
    task = (f"Given a cluster of {task_element} separated by <table> ... </table>, identify and provide a name for the conceptual type that best describes the tables in the cluster." )
    overall_dict = {}
    for key, cluster in cluster_dict.items():
        overall_dict[key] = {}
        names = None
        if table_names is not None:
            names = [table_names[i] for i in cluster]
        naming = GPTnaming(apiKey="", format=format,
                           sampling=sampling, sampling_number=sampling_number, header=header, table_names=names)
        cluster_df = read_clusters_tables(hp.dataset,cluster)
        reply = naming.generate_answers(cluster_df, task, reply=None, instructions= instruction, shorts = shorts, AI_instructions=AI_instruction, newMessage=True)
        results.loc[key, new_col] = reply
        print(reply, len(cluster))
        overall_dict[key]['cluster type']=reply
        #break

        """
        overall_dict[key]['table types'] = {}
        for index, table_name in enumerate(cluster):
            overall_dict[key]['table types'][table_name]={}
            table_df = cluster_df[index]
            type_gpt = TableType(apiKey="", table=table_df)
            type_specific = type_gpt.table_type()
            print(f"{table_df.head(3)} \n specific type's candidate of table is :\n{type_specific}")
            overall_dict[key]['table types'][table_name]['specific type']=type_specific
            judge = type_gpt.judge_ancestor( type_specific, reply)
            print(f"is Subtype of {reply}: {judge}")
            overall_dict[key]['table types'][table_name]['isSubType']=judge"""

    with open(f'result/P1/{hp.dataset}/All/{hp.P1Embed[:-4]}/data_types_no_tp_mention.pickle', 'wb') as file:
        pickle.dump(overall_dict, file)
    output = f"result/P1/{hp.dataset}/All/{hp.P1Embed[:-4]}/purityCluster1.csv" if groundtruth is False else f"datasets/{hp.dataset}/gt_naming.csv"
    results.to_csv(output)



