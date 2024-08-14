import argparse
import pickle

import pandas as pd

from TableCluster.tableClustering import P1, P2, P3, baselineP1
from EndToEnd.EndToEnd import endToEnd
from RelationshipSearch.SearchRelationship import relationshipDiscovery
from Runtime import Running
from TestNaming import testNaming
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="GDS")
    parser.add_argument("--embed", type=str, default='sbert')
    parser.add_argument("--baseline", dest="baseline", action="store_true")
    parser.add_argument("--clustering", type=str, default='Agglomerative')
    parser.add_argument("--iteration", type=int, default=1)

    """ This is for the testing of different steps """
    parser.add_argument("--step", type=int, default=1)
    """ This is the parameter for P1"""
    parser.add_argument("--estimateNumber", type=int, default=8)
    """ Parameter for slicing the dendrogram in Step3 """
    parser.add_argument("--intervalSlice", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.08)
    """ Step 4 parameter """
    parser.add_argument("--similarity", type=float, default=0.85)
    parser.add_argument("--portion", type=float, default=0.8)
    parser.add_argument("--Euclidean", dest="Euclidean", action="store_true")
    """ single-column mode without table context """
    parser.add_argument("--subjectCol", dest="subjectCol", action="store_true")
    # row / column-ordered for preprocessing
    """Parameter for End To End """
    parser.add_argument("--P1Embed", type=str, default='cl_SCT6_lm_sbert_head_column_0_none_subCol.pkl')
    #cl_SCT8_lm_sbert_head_column_0_none_subCol.pkl
    parser.add_argument("--P23Embed", type=str, default='cl_SC8_lm_sbert_head_column_0_header_column.pkl')
    parser.add_argument("--P4Embed", type=str, default='cl_SC8_lm_bert_head_column_0_none_column.pkl')
    # parser.add_argument("--slice_start", type=int, default=0)
    # parser.add_argument("--slice_stop", type=int, default=1)
    """ This is randomly selecting tables from the datasets """
    parser.add_argument("--SelectType", type=str, default='')
    """ This is for testing running time, the total number of tables """
    parser.add_argument("--tableNumber", type=int, default=0)

    hp = parser.parse_args()
    if hp.step == 0:
        Running(hp)
    if hp.step == 1:
        P1(hp) if hp.baseline is False else baselineP1(hp)
    elif hp.step == 2:
        P2(hp)
    elif hp.step == 3:
        P3(hp)
    elif hp.step == 4:
        relationshipDiscovery(hp)
    elif hp.step == -1:
        endToEnd(hp)
    elif hp.step == 9:

        names = pd.read_csv(f"datasets/{hp.dataset}/naming.csv")
        name_dict = {str(iter_n[0])[:-4]: iter_n[1] for index, iter_n in names.iterrows()}
        print(name_dict)
        with open(f"datasets/{hp.dataset}/graphGroundTruth.pkl", "rb") as f:
            graph = pickle.load(f)
        top_level_nodes = [node for node, in_degree in graph.in_degree() if in_degree == 0]
        top_level_and_children = {node: list(graph.successors(node)) for node in top_level_nodes}
        print(top_level_and_children)
        AI_Ins = [f"The tables are web tables. The cluster possibly indicates an top level conceptual type in Schema.org. Schema.org's top-level types are {top_level_nodes}"] # their closest children types
        instruction = ["Background: The cluster of tables is derived from a hierarchical clustering algorithm applied to the embeddings of tables."
            "This clustering indicates a mutual conceptual entity type among the tables.",

             "Data Provided: all tables in the cluster."
             "Each table is separated by <table> </table> ."
                      "Each TABLE follows the format: <table> TABLE_INDEX: HEADER1|HEADER2|HEADER3|\n Instance11| Instance12|Instance13| \n Instance21| Instance22|Instance23| \n....</table>.",

          "Task: Analyze the topic/theme of each of the provided table within the cluster."
          "Based on the context and theme of the tables, determine the mutual entity type of the tables. Provide a suitable name that describes this mutual entity type.",

        "Expected Output: The output should be a single word or a short phrase that clearly describes the mutual "
        "entity type of the tables in the cluster."
        "The name should be descriptive and relevant to the context of the data. AND NO OTHER TEXT!"]


        testNaming(hp, groundtruth = False, format=2, sampling=None, sampling_number=5, header=False, table_names=None,
                   instruction=instruction, AI_instruction=AI_Ins)



'''
"Background: The cluster of tables is derived from a hierarchical clustering algorithm applied to the embeddings of subject attributes of tables."
"This clustering indicates a mutual conceptual entity type among the tables.",

"Data Provided: The cluster includes all tables in the cluster."
"Each subject attribute is separated by <sc></sc>."
"Each subject attribute follows the format: <sc> Column name : instance1, instance2, instance3,....</sc>.",

"Task: Analyze the provided subject attributes within the cluster. "
"Based on the content and pattern of the subject attributes, determine the mutual entity type of the tables. Provide a suitable name that describes this mutual entity type.",

"Expected Output: The output should be a single word or a short phrase that clearly describes the mutual "
"entity type of the tables."
"The name should be descriptive and relevant to the context of the data. AND NO OTHER TEXT!"'''