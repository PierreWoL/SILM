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


ground_label_name1 = "01SourceTables.csv"
data_path = os.path.join(os.getcwd(), "datasets/TabFact/", ground_label_name1)
ground_truth_csv = pd.read_csv(data_path, encoding='latin-1')
result_dict = dict(zip(ground_truth_csv.iloc[:, 0], ground_truth_csv.iloc[:, 2]))
names = ground_truth_csv["fileName"].unique()
labels = os.listdir(os.path.join(os.getcwd(), "datasets/TabFact/Label"))
no_labels = [i for i in names if i not in labels]
# ground_truth_csv = ground_truth_csv[ground_truth_csv["fileName"].isin(no_labels)]
ground_truth = dict(zip(ground_truth_csv.iloc[:, 0], ground_truth_csv.iloc[:, 4]))


def parallel_crawling():
    for i in range(0, 11):  # 160

        end = (i + 1) * 400 - 1
        if end >= len(ground_truth_csv):
            end = len(ground_truth_csv)
        start = i * 400
        slice = ground_truth_csv[start:end]

        result_diction = dict(zip(slice.iloc[:, 0], slice.iloc[:, 2]))

        dataframes_list = query_wikidata_parallel(result_diction)


abstract = [ 'PhysicalActivity','object', 'result', 'temporal entity', 'inconsistency', 'noun', 'noun phrase', 'remains', 'use',
            'independent continuant', 'observable entity', 'artificial entity', 'natural physical object',
            'occurrence', 'relation', 'group of physical objects', 'economic entity', 'group of works',
            'concrete object', 'three-dimensional object', 'part', 'geographic entity', 'artificial geographic entity',
            'source', 'group or class of physical objects', 'role', 'phenomenon', 'physical entity', 'means',
            'spatio-temporal entity', 'spatial entity', 'one-dimensional space', 'physical object',
            'continuant', 'collective entity', 'space object', 'type', 'information', 'anatomical entity',
            'output', 'abstract object', 'class', 'non-physical entity', 'integral', 'quantity', 'former entity',
            'occurrent', 'cause', 'idiom', 'lect', 'modification', 'alteration', 'control', 'consensus',
            'social relation', 'process', 'rivalry', 'mental process', 'condition',
            'social phenomenon', 'manifestation', 'work', 'source of information', 'knowledge type', 'action',
            'time interval', 'interaction', 'record', 'language variety', 'intentional human activity',
            'status', 'group of living things', 'agent', 'sign', 'content', 'converter', 'resource', 'metaclass',
            'unit', 'human activity','effect', 'archives', 'sub-fonds', 'evaluation',
            'interface', 'contributing factor', 'undesirable characteristic', 'structure', 'method', 'matter', 'change',
            'physical phenomenon', 'binary relation', 'building work', 'power', 'management', 'long, thin object',
            'definite integral', 'physical property', 'multi-organism process', 'data', 'multiset', 'line',
            'proper noun', 'physicochemical process', 'group', 'collection', 'historical source'
            'interaction', 'information resource', 'list', 'plan', 'scale', 'memory', 'social structure',
            'source text', 'open content', 'written work', 'strategy', 'group of humans', 'system', 'deformation',
            'representation', 'multicellular organismal process', 'operator', 'social system']
top = ['Place', 'Action', 'Intangible', 'Organization', 'CreativeWork', 'MedicalEntity', 'BioChemEntity', 'Event', 'Product', 'Person', 'Taxon']
similar_words = {}
with open("filter_sim_all.pkl", "rb") as file:
    all_sims = pickle.load(file)
for key, value in all_sims.items():
    for tuple in value.keys():
        word = tuple[0]
        if tuple[0] in similar_words.keys():
            if tuple[1] not in similar_words[word]:
                similar_words[word].append(tuple[1])
        else:
            similar_words[word] = [tuple[1]]

for word, similar_word_list in similar_words.items():
    if len(similar_word_list) == 1:
        similar_words[word] = similar_word_list[0]

#unique_items = list(set(similar_words.values()))



node_length = 0
G = nx.DiGraph()
for index, row in ground_truth_csv.iterrows():
    if row["fileName"] in labels:
        label_path = os.path.join(os.getcwd(), "datasets/TabFact/Label")
        df = pd.read_csv(os.path.join(label_path, row["fileName"]), encoding='UTF-8').iloc[:, 3:9]
        for _, row2 in df.iterrows():
            labels_table = row2.dropna().tolist()
            for i in range(len(labels_table) - 1):
                if labels_table[i + 1] != labels_table[i]:
                    if labels_table[i + 1] not in abstract and labels_table[i] not in abstract:
                        child_type = labels_table[i]


                        if labels_table[i + 1] in G.nodes():
                                if labels_table[i] not in nx.ancestors(G, labels_table[i + 1]):
                                        if labels_table[i + 1] != child_type \
                                                and "process" not in labels_table[i + 1].lower() \
                                                and "process" not in child_type.lower():
                                            G.add_edge(labels_table[i + 1], child_type)
                                            continue

                        else:

                                    if labels_table[i + 1] != child_type and "process" not in labels_table[
                                        i + 1].lower() \
                                            and "process" not in child_type.lower():
                                        G.add_edge(labels_table[i + 1], child_type)
                                        continue


"""for index, row in ground_truth_csv.iterrows():
    if row["fileName"] in labels:
        label_path = os.path.join(os.getcwd(), "datasets/TabFact/Label")
        df = pd.read_csv(os.path.join(label_path, row["fileName"]), encoding='UTF-8').iloc[:, 3:9]
        for _, row2 in df.iterrows():
            labels_table = row2.dropna().tolist()
            for i in range(len(labels_table) - 1):
                if labels_table[i + 1] != labels_table[i]:
                    if labels_table[i + 1] not in abstract and labels_table[i] not in abstract:
                        child_type = similar_words[labels_table[i]] \
                            if labels_table[i] in similar_words.keys() else labels_table[i]
                        if child_type in top:
                            break
                        else:
                            if labels_table[i + 1] in G.nodes():
                                if labels_table[i] not in nx.ancestors(G, labels_table[i + 1]):
                                    if labels_table[i + 1] not in similar_words.keys():
                                        if labels_table[i + 1] != child_type \
                                                and "process" not in labels_table[i + 1].lower() \
                                                and "process" not in child_type.lower():
                                            G.add_edge(labels_table[i + 1], child_type)

                                            continue
                                    else:
                                        if similar_words[labels_table[i + 1]] != child_type and "process" not in \
                                                labels_table[i + 1].lower() \
                                                and "process" not in child_type.lower():
                                            G.add_edge(similar_words[labels_table[i + 1]], child_type)
                                            break
                            else:
                                if labels_table[i + 1] not in similar_words.keys():
                                    if labels_table[i + 1] != child_type and "process" not in labels_table[
                                        i + 1].lower() \
                                            and "process" not in child_type.lower():
                                        G.add_edge(labels_table[i + 1], child_type)
                                        continue
                                else:
                                    if similar_words[labels_table[i + 1]] != child_type and "process" not in \
                                            labels_table[i + 1].lower() \
                                            and "process" not in child_type.lower():
                                        G.add_edge(similar_words[labels_table[i + 1]], child_type)
                                        break
"""

"""
    else:
        if row["class"] != " ":
            superclass = row["class"]
            classX = row["superclass"]
            all_nodes = {superclass, classX}
            all_nodes = all_nodes - set(G.nodes())
            G.add_nodes_from(all_nodes)
            G.add_edge(superclass,classX)
"""

target_path = os.path.join(os.getcwd(), "datasets/TabFact/")
with open(os.path.join(target_path, "graphGroundTruth2.pkl"), "wb") as file:
    pickle.dump(G, file)

# Setting graph attributes for top-to-bottom layout
"""
# Load the graph from the pickle file
"""

# Drawing the directed graph
"""
graph_layout = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
plt.figure(figsize=(12, 10))
nx.draw(G, pos=graph_layout, with_labels=True, node_size=1500, node_color="skyblue", arrowsize=20)
plt.show()"""
"""

# results = [(ground_truth_csv.iloc[i, 0], ground_truth_csv.iloc[i, 2]) for i in range(0, len(ground_truth_csv))]

#

"""  # Process the DataFrames as needed

# Below needs reconstruct and important
"""target_path = os.path.join(os.getcwd(), "datasets/TabFact/")
with open(os.path.join(target_path, "graphGroundTruth.pkl"), "rb") as file:
    G = pickle.load(file)

all_nodes = set(G.nodes())

all_child_nodes = set(child for _, child in G.edges())

top_nodes = all_nodes - all_child_nodes

print("PARENT:")
print(top_nodes, len(top_nodes))
indegree_zero_nodes = {node for node in G.nodes() if G.in_degree(node) == 0}
all_p = []
lowest_n = []
Top_column_label = {}
Label_Low = {}
for index, row in ground_truth_csv.iterrows():
    if row["class"] != ' ':
        table_name, classX = row["fileName"], row["class"]
        topmost_parent = []
        if table_name in labels:
            table_label = pd.read_csv(os.path.join(os.getcwd(), "datasets/TabFact/Label", table_name))[
                "classLabel"].unique()
            Label_Low[row["fileName"]] = list(table_label)
            for x_la in list(table_label):
                if x_la not in lowest_n:
                    lowest_n.append(x_la)
            # print(table_label)
            for loa in table_label:
                ancestors = list(nx.ancestors(G, loa))
                compare_parent = [i for i in ancestors if i in top_nodes]
                indegrees = [G.in_degree(parent) for parent in ancestors]
                min_element = min(indegrees)
                min_indices = [i for i, x in enumerate(indegrees) if x == min_element]
                for i in min_indices:
                    topmost_parent.append(ancestors[i])

        else:
            Label_Low[row["fileName"]] = [classX]
            if classX not in lowest_n:

                lowest_n.append(classX)
            ancestors = list(nx.ancestors(G, classX))

            if len(ancestors) == 0:
                topmost_parent = [classX]
            else:
                indegrees = [G.in_degree(parent) for parent in ancestors]
                min_element = min(indegrees)
                min_indices = [i for i, x in enumerate(indegrees) if x == min_element]
                topmost_parent = [ancestors[i] for i in min_indices]

            # print(classX, topmost_parent)
        for item in topmost_parent:
            if item not in all_p:
                all_p.append(item)
        Top_column_label[row["fileName"]] = sorted(topmost_parent)

data_path = os.path.join(os.getcwd(), "datasets/TabFact/02TableAttributes.csv")
multi = pd.read_csv(data_path, encoding='latin1')
print("all parent nodes and lowest nodes are",len(all_p), len(lowest_n))
ground_truth_csv['TopLabel'] = Top_column_label
ground_truth_csv['LowestLabel'] = Label_Low

# print("The topmost parent node of node '{}' is '{}'".format(classX, topmost_parent))
# for i in labels:


ground_truth_csv['LowestClass'] = ground_truth_csv['fileName'].map(Label_Low)
ground_truth_csv['TopClass'] = ground_truth_csv['fileName'].map(Top_column_label)

multi['LowestClass'] = multi['fileName'].map(Label_Low)
multi['TopClass'] = multi['fileName'].map(Top_column_label)

ground_truth_csv.to_csv(os.path.join(os.getcwd(), "datasets/TabFact/Try.csv"))

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
