import os.path
import pickle
import sys
import pandas as pd
import io
import requests

from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def query_wikidata_api(wiki_url):
    # Wikidata SPARQL endpoint URL
    endpoint_url = "https://query.wikidata.org/sparql"

    # Define the request headers with the user agent
    headers = {
        "User-Agent": "My-App/1.0"
    }

    # Define the request parameters
    sparql_query = f"""
    SELECT ?x ?xLabel ?classLabel ?superclassLabel ?superclass2Label ?superclass3Label ?superclass4Label ?superclass5Label ?superclass6Label WHERE {{
    <{wiki_url}> schema:about ?x.
    ?x wdt:P31 ?class. #instance Of
    ?class wdt:P279 ?superclass. #subclass of superclass1
    ?superclass wdt:P279 ?superclass2. #subclass of superclass2
    ?superclass2 wdt:P279 ?superclass3. #subclass of superclass3
    ?superclass3 wdt:P279 ?superclass4. #subclass of superclass4
    ?superclass4 wdt:P279 ?superclass5. #subclass of superclass5
    ?superclass5 wdt:P279 ?superclass6. #subclass of superclass6
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """

    # Send the GET request to the Wikidata API
    response = requests.get(endpoint_url, params={"format": "json", "query": sparql_query}, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: Unable to fetch data for {wiki_url}.")
        return None


def query_wikidata_parallel(wiki_urls):
    # Query the Wikidata API for each URL in parallel

    with ThreadPoolExecutor() as executor:
        results_json_list = list(executor.map(query_wikidata_api, list(wiki_urls.values())))
    # Convert the JSON results to DataFrames
    dataframes = []
    for index, results_json in enumerate(results_json_list):
        if results_json:
            rows = []
            for item in results_json["results"]["bindings"]:
                row = {
                    "x": item["x"]["value"],
                    "xLabel": item["xLabel"]["value"],
                    "classLabel": item["classLabel"]["value"],
                    "superclassLabel": item["superclassLabel"]["value"],
                    "superclass2Label": item["superclass2Label"]["value"],
                    "superclass3Label": item["superclass3Label"]["value"],
                    "superclass4Label": item["superclass4Label"]["value"],
                    "superclass5Label": item["superclass5Label"]["value"],
                    "superclass6Label": item["superclass6Label"]["value"]
                }
                rows.append(row)

            if len(rows):
                df = pd.DataFrame(rows)
                data_path = os.path.join(os.getcwd(), "datasets/TabFact/Label", list(wiki_urls.keys())[index])
                df.to_csv(data_path)
                dataframes.append(df)

    return dataframes


def parallel_crawling(gt_csv: dict):
    for i in range(0, 11):  # 160

        end = (i + 1) * 400 - 1
        if end >= len(gt_csv):
            end = len(gt_csv)
        start = i * 400
        slice = gt_csv[start:end]
        result_diction = dict(zip(slice.iloc[:, 0], slice.iloc[:, 2]))
        dataframes_list = query_wikidata_parallel(result_diction)


abstract = ['PhysicalActivity', 'object', 'result', 'temporal entity', 'inconsistency', 'noun', 'noun phrase',
            'remains', 'use', 'independent continuant', 'observable entity', 'artificial entity',
            'natural physical object',
            'occurrence', 'relation', 'group of physical objects', 'economic entity', 'group of works',
            'concrete object', 'three-dimensional object', 'part', 'geographic entity', 'artificial geographic entity',
            'source', 'group or class of physical objects', 'role', 'phenomenon', 'physical entity', 'means',
            'spatio-temporal entity', 'spatial entity', 'one-dimensional space', 'physical object',
            'continuant', 'collective entity', 'space object', 'type', 'information', 'anatomical entity',
            'output', 'abstract object', 'class', 'non-physical entity', 'integral', 'quantity', 'former entity',
            'occurrent', 'cause', 'idiom', 'lect', 'modification', 'alteration', 'control', 'consensus',
            'social relation', 'process', 'rivalry', 'mental process', 'condition', 'military activity',
            'human aerial activity', 'EntryPoint',
            'social phenomenon', 'manifestation', 'work', 'source of information', 'knowledge type', 'action',
            'time interval', 'interaction', 'record', 'language variety', 'intentional human activity',
            'status', 'group of living things', 'agent', 'sign', 'content', 'converter', 'resource', 'metaclass',
            'unit', 'human activity', 'effect', 'archives', 'sub-fonds', 'evaluation',
            'interface', 'contributing factor', 'undesirable characteristic', 'structure', 'method', 'matter', 'change',
            'physical phenomenon', 'binary relation', 'building work', 'power', 'management', 'long, thin object',
            'definite integral', 'physical property', 'multi-organism process', 'data', 'multiset', 'line',
            'proper noun', 'physicochemical process', 'group', 'collection', 'historical source',
            'interaction', 'information resource', 'list', 'plan', 'scale', 'memory', 'social structure',
            'source text', 'open content', 'written work', 'strategy', 'group of humans', 'system', 'deformation',
            'representation', 'multicellular organismal process', 'operator', 'social system']

ground_label_name1 = "01SourceTables.csv"
data_path = os.path.join(os.getcwd(), "datasets/TabFact/", ground_label_name1)
ground_truth_csv = pd.read_csv(data_path, encoding='latin-1')
result_dict = dict(zip(ground_truth_csv.iloc[:, 0], ground_truth_csv.iloc[:, 2]))
names = ground_truth_csv["fileName"].unique()
labels = os.listdir(os.path.join(os.getcwd(), "datasets/TabFact/Label"))
no_labels = [i for i in names if i not in labels]
# ground_truth_csv = ground_truth_csv[ground_truth_csv["fileName"].isin(no_labels)]
ground_truth = dict(zip(ground_truth_csv.iloc[:, 0], ground_truth_csv.iloc[:, 4]))

similar_words = {}
with open("filter_sim_all.pkl", "rb") as file:
    all_sims = pickle.load(file)
for key, value in all_sims.items():
    # print(value)
    for tuple in value.keys():
        word = tuple[0]
        if tuple[0] in similar_words.keys():
            if tuple[1] not in similar_words[word]:
                similar_words[word].append(tuple[1])
        else:
            similar_words[word] = [tuple[1]]

for word, similar_word_list in similar_words.items():
    # print(word, similar_word_list)
    if len(similar_word_list) == 1:
        similar_words[word] = similar_word_list[0]
df = pd.DataFrame(list(similar_words.items()), columns=['Key', 'Value'])

top = ['Place', 'Action', 'Intangible', 'Organization', 'CreativeWork', 'MedicalEntity', 'BioChemEntity', 'Event',
       'Product', 'Person', 'Taxon']

similarity_csv = pd.read_csv("datasets/TabFact/similar.csv")
similarity_dict = dict(zip(similarity_csv.iloc[:, 0], similarity_csv.iloc[:, 1]))
print(similarity_dict)

# unique_items = list(set(similar_words.values()))


node_length = 0
G = nx.DiGraph()
previous_values = 0

"""
def prune_graph(G, top_list, similar_list):
    def prune_from_node(source):
        targets = list(G.successors(source))
        if not targets:
            return False
        for target in targets:
            if target in top_list:
                G.remove_node(source)
                return True
            elif target in similar_list:
                G.remove_node(source)
                return True
            else:
                if prune_from_node(target):
                    G.remove_node(source)
                    return True
        return False
    top_nodes = [node for node, degree in G.in_degree() if degree == 0]
    for top_node in top_nodes:
        prune_from_node(top_node)
    return G
"""

"""def prune_graph_around_node(G, node):

    descendants_of_node = nx.descendants(G, node)
    valid_nodes = descendants_of_node | {node}
    nodes_to_remove = set(G.nodes()) - valid_nodes
    G.remove_nodes_from(nodes_to_remove)

    return G"""


def prune_graph_around_nodes(tree, nodes):
    valid_nodes = set()

    # 对每个节点执行相同的操作
    for node in nodes:
        # 找到从指定节点派生的所有节点
        descendants_of_node = nx.descendants(tree, node)
        valid_nodes = valid_nodes | descendants_of_node | {node}

    # 找到需要删除的节点
    nodes_to_remove = set(tree.nodes()) - valid_nodes

    # 从图中删除这些节点
    tree.remove_nodes_from(nodes_to_remove)

    return tree


def prune_graph(tree, top_list, similar_list):
    def prune_from_node(source):
        targets = list(tree.successors(source))
        if not targets:
            return False
        pruned = False
        for target in targets:
            if target in top_list:
                tree.remove_node(source)
                return True
            elif target in similar_list:
                tree.remove_node(source)
                return True
            else:
                pruned_child = prune_from_node(target)
                if pruned_child:
                    pruned = True

        return pruned

    # 获取所有没有前驱的节点，即图中的顶部节点
    # top_nodes = [node for node, degree in G.in_degree() if degree == 0]
    node_keep_top = [i for i in tree.nodes() if i in top_list]
    node_keep_all = [i for i in tree.nodes() if i in similar_list]

    if len(node_keep_top) > 0:
        prune_graph_around_nodes(tree, node_keep_top)
        isolates = list(nx.isolates(tree))
        tree.remove_nodes_from(isolates)
        return tree
    if len(node_keep_all) > 0:
        prune_graph_around_nodes(tree, node_keep_all)
        isolates = list(nx.isolates(tree))
        tree.remove_nodes_from(isolates)
        return tree
    return tree


"""
lists = []

exception_list = []
for index, row in ground_truth_csv.iterrows(): #.iloc[200:1000]
    if row["fileName"] in labels:

        label_path = os.path.join(os.getcwd(), "datasets/TabFact/Label")
        df = pd.read_csv(os.path.join(label_path, row["fileName"]), encoding='UTF-8').iloc[:, 3:9]
        df_cells_list = df.values.flatten().tolist()
        G_cur = nx.DiGraph()
        intersection1 = list(set(df_cells_list) & set(similarity_dict.keys()))
        intersection2 = list(set(df_cells_list) & set(top))
        if len(intersection1) != 0 or len(intersection2) != 0:
            for _, row2 in df.iterrows():
                labels_table = row2.dropna().tolist()
                for i in range(len(labels_table) - 1):
                    if labels_table[i + 1] != labels_table[i]:
                        # if labels_table[i + 1] not in abstract and labels_table[i] not in abstract:

                        child_type = labels_table[i]
                        if child_type in similarity_dict.keys():
                            child_type = similarity_dict[child_type]
                        parent_type = labels_table[i + 1]
                        if parent_type in similarity_dict.keys():
                            parent_type = similarity_dict[parent_type]

                        if parent_type in G_cur.nodes():
                            if labels_table[i] not in nx.ancestors(G_cur, parent_type):
                                if parent_type != child_type:
                                    G_cur.add_edge(parent_type, child_type)
                        else:
                            if parent_type in similarity_dict.keys():
                                parent_type = similarity_dict[parent_type]
                            if parent_type != child_type:
                                G_cur.add_edge(parent_type, child_type)

            G_cur = prune_graph(G_cur, top, similarity_dict.values())
            G.add_nodes_from(G_cur.nodes())
            G.add_edges_from(G_cur.edges())
            if G.has_node("event"):
                    print(row["fileName"],df_cells_list)
                    break
        else:

            exception_list.append(index)
            for _, row2 in df.iterrows():
                labels_table = row2.dropna().tolist()
                for i in range(len(labels_table) - 1):
                    if labels_table[i + 1] != labels_table[i]:
                        # if labels_table[i + 1] not in abstract and labels_table[i] not in abstract:
                        child_type = labels_table[i]
                        if labels_table[i+1] not in abstract and child_type not in top:
                            if labels_table[i + 1] in G.nodes():
                                if labels_table[i] not in nx.ancestors(G, labels_table[i + 1]):
                                    if labels_table[i + 1] != child_type:
                                        G.add_edge(labels_table[i + 1], child_type)
                                        continue

                            else:
                                if labels_table[i + 1] != child_type:
                                    G.add_edge(labels_table[i + 1], child_type)
                                    continue
        #if list(intersection) not in lists and len(intersection)>0:
            #graph_layout = nx.drawing.nx_agraph.graphviz_layout(G_cur, prog="dot", args="-Grankdir=TB")
            #plt.figure(figsize=(12, 12))
            #nx.draw(G_cur, pos=graph_layout, with_labels=True, node_size=1500, node_color="skyblue", arrowsize=20)
            #plt.show()

            

            #previous_values = len(G_cur.nodes)
            #graph_layout = nx.drawing.nx_agraph.graphviz_layout(G_cur, prog="dot", args="-Grankdir=TB")
            #plt.figure(figsize=(12, 12))
            #nx.draw(G_cur, pos=graph_layout, with_labels=True, node_size=1500, node_color="skyblue", arrowsize=20)
            #plt.show()
            #lists.append(list(intersection))

    else:
        child_type = row["class"]
        parent_type = row["superclass"]
        if child_type in similarity_dict.keys():
            child_type = similarity_dict[child_type]
        if parent_type in similarity_dict.keys():
            parent_type = similarity_dict[parent_type]

        if row["class"] != "" or row["class"] != " ":
            G.add_edge(parent_type,child_type)

    # if previous_values<len(G.nodes):
    # previous_values = len(G.nodes)
    # graph_layout = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
    # plt.figure(figsize=(23, 23))
    # nx.draw(G, pos=graph_layout, with_labels=True, node_size=1500, node_color="skyblue", arrowsize=20)
    # plt.show()

#previous_values = len(G.nodes)
#graph_layout = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
#plt.figure(figsize=(23, 23))
#nx.draw(G, pos=graph_layout, with_labels=True, node_size=1500, node_color="skyblue", arrowsize=20)
#plt.show()

target_path = os.path.join(os.getcwd(), "datasets/TabFact/")
with open(os.path.join(target_path, "graphGroundTruth1106.pkl"), "wb") as file:
  pickle.dump(G, file)
"""
"""unimportant = ['Thing', 'Boolean', 'Text', '2000/01/rdf-schema#Class', 'Date', 'DateTime', 'False', 'Number',
               'Time', 'True', 'DataType', 'Float', 'Integer', 'URL', 'XPathType',
               'PronounceableText', 'CssSelectorType', 'StupidType']
target_path = os.path.join(os.getcwd(), "datasets/TabFact/")
with open(os.path.join(target_path, "graphGroundTruth1106.pkl"), "rb") as file:
    G = pickle.load(file)

dataSchema = pd.read_csv("schemaorg-all-http-types.csv")
prefix = "http://schema.org/"


def add_nodes_schema(tree, topNodes):
    print("\n")
    for node in topNodes:
        select_rows = dataSchema[dataSchema['label'] == node]
        if len(select_rows) != 0:
            parent_type = str(list(select_rows["subTypeOf"])[0]).replace(prefix, "")

            if parent_type not in unimportant:

                if "," in parent_type:
                    list_parent = parent_type.split(", ")
                    for i in list_parent:

                        if i != 'PhysicalActivity':
                            print(node, i)
                            tree.add_edge(i, node)
                else:

                    if node !='PhysicalActivity':
                        print(node, parent_type)
                        tree.add_edge(parent_type, node)
    return tree


current_len = 0
while len(G.nodes()) > current_len:
    current_len = len(G.nodes())
    top_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    print(top_nodes, len(top_nodes))
    G = add_nodes_schema(G, top_nodes)

with open(os.path.join(target_path, "graphGroundTruth.pkl"), "wb") as file:
    pickle.dump(G, file)
"""

# exc = df.iloc[exception_list]
# exc.to_csv('datasets/TabFact/selected_rows.csv', index=False)  # 保存为CSV文件
"""
node_length = 0
G = nx.DiGraph()
previous_values =0
for index, row in ground_truth_csv.iterrows():
    if row["fileName"] in labels:
        label_path = os.path.join(os.getcwd(), "datasets/TabFact/Label")
        df = pd.read_csv(os.path.join(label_path, row["fileName"]), encoding='UTF-8').iloc[:, 3:9]
        for _, row2 in df.iterrows():
            labels_table = row2.dropna().tolist()
            for i in range(len(labels_table) - 1):
                if labels_table[i + 1] != labels_table[i]:
                    # if labels_table[i + 1] not in abstract and labels_table[i] not in abstract:
                    child_type = labels_table[i]

                    if labels_table[i + 1] in G.nodes():
                        if labels_table[i] not in nx.ancestors(G, labels_table[i + 1]):
                            if labels_table[ i + 1] != child_type:
                                if labels_table[ i + 1] in similar_words.keys():
                                    print(labels_table[i + 1], child_type)

                                # and "process" not in labels_table[i + 1].lower() and "process" not in child_type.lower()
                                G.add_edge(labels_table[i + 1], child_type)
                                continue

                    else:

                        if labels_table[i + 1] != child_type:
                            G.add_edge(labels_table[i + 1], child_type)
                            continue
    else:
        if row["class"]!="" or row["class"]!=" ":
            G.add_edge(row["superclass"], row["class"])


    #if previous_values<len(G.nodes):
        #previous_values = len(G.nodes)
       # graph_layout = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
       # plt.figure(figsize=(23, 23))
        #nx.draw(G, pos=graph_layout, with_labels=True, node_size=1500, node_color="skyblue", arrowsize=20)
        #plt.show()
target_path = os.path.join(os.getcwd(), "datasets/TabFact/")



with open(os.path.join(target_path, "graphGroundTruth3.pkl"), "wb") as file:
    pickle.dump(G, file)"""

# Setting graph attributes for top-to-bottom layout
"""
# Load the graph from the pickle file
"""

# Drawing the directed graph
"""
"""
"""

# results = [(ground_truth_csv.iloc[i, 0], ground_truth_csv.iloc[i, 2]) for i in range(0, len(ground_truth_csv))]

#

"""  # Process the DataFrames as needed




ground_label_name1 = "01SourceTables.csv"
data_path = os.path.join(os.getcwd(), "datasets/TabFact/", ground_label_name1)
ground_truth_csv = pd.read_csv(data_path, encoding='latin-1')


# Below needs reconstruct and important
target_path = os.path.join(os.getcwd(), "datasets/TabFact/")
with open(os.path.join(target_path, "graphGroundTruth1106.pkl"), "rb") as file:
    G = pickle.load(file)
labels = os.listdir(os.path.join(os.getcwd(), "datasets/TabFact/Label"))
for index, row in ground_truth_csv.iterrows():
    parent_top_pers = []
    lowest = None
    if row["fileName"] in labels:
        label_path = os.path.join(os.getcwd(), "datasets/TabFact/Label")
        lowest_type = pd.read_csv(os.path.join(label_path, row["fileName"]), encoding='UTF-8').iloc[:, 3]
        lowest_type_unqi = lowest_type.unique()
        for type_low in lowest_type_unqi:
            if type_low in G.nodes:
                parent_top_per = [item for item in nx.ancestors(G, type_low) if
                                  G.in_degree(item) == 0]
                parent_top_pers.extend(parent_top_per)
        lowest = list(lowest_type_unqi)

    else:
        if row["class"] != " ":
            type_low = row["class"]
            if type_low in G.nodes():
                parent_top_per = [item for item in nx.ancestors(G, type_low) if
                                  G.in_degree(item) == 0]
                parent_top_pers.extend(parent_top_per)
            lowest = list([type_low])
    ground_truth_csv.iloc[index, 4] = lowest
    ground_truth_csv.iloc[index, 5] = list(set(parent_top_pers))

ground_truth_csv.to_csv(os.path.join(os.getcwd(), "datasets/TabFact/Try.csv"))





"""
multi.dropna(subset=['TopClass'], inplace=True)

multi.to_csv(os.path.join(os.getcwd(), "datasets/TabFact/Try02.csv"))
print(len(top_nodes - set(all_p)))
a = []
b = []
for i in range(0, len(Label_Low)):
    ai = list(Label_Low.values())[i]
    bi = list(Top_column_label.values())[i]
    if ai not in a:
        a.append(ai)
    if bi not in b:
        b.append(bi)
print(len(a), len(b))
"""
