import ast
import pickle
import networkx as nx
from openai import OpenAI
import pandas as pd
client = OpenAI(api_key="sk-proj-L19IhpKWkjF8KZfBjtAvT3BlbkFJ5a1Ixe0wl4YZm9AhCS1b")
def find_indices(lst, word):
    return [i for i, x in enumerate(lst) if x == word]
def find_keys_by_value(dictionary, value):
    keys = []
    for key, val in dictionary.items():
        if val == value:
            keys.append(key)
    return keys
def highLevelGroup(top_types_dict):
    list_t = list(top_types_dict.values())
    similar_semantics=[]
    cluster_messages = []
    cluster_messages.append({"role": "user", "content":
        "I am building a conceptual schema and now have some conceptual entity types inferred from different table clusters." +
        "Each table cluster indicates the mutual entity type of tables inside the tables. " +
        f"Now, I will provide a list of conceptual entity types: {list_t}, inferred from the table clusters, the entity type is separated by commas. " +
        f"Please tell me which types in the list that share the same semantics. The output should be a nested list"
        f"like the following: [['Type1', 'Type2', 'Type3'],['Type4', 'Type5']].Each type should be the same format as in the provided list."})


    response = client.chat.completions.create(
        model="gpt-3.5-turbo",messages=cluster_messages)
    similar = response.choices[0].message.content
    print(similar, type(similar))
    similar_semantics = ast.literal_eval(similar)
    similar_semantics_index = []

    for similar_tuple in similar_semantics:
        indexes_tuple = set()
        for type_per in similar_tuple:
            for index in find_keys_by_value(top_types_dict, type_per):
                indexes_tuple.add(index)
        similar_semantics_index.append(indexes_tuple)
    print(similar_semantics_index)
    return similar_semantics


def highLevelMatch(top_types_dict):
    similar_semantics =[]
    ancestor=[]
    for index_A in top_types_dict:
        index_A_similar = {index_A}
        messages = []
        messages.append({"role": "user", "content":
            f"I will provide a list of pairs of conceptual entity types. Please tell me if they match."})
        messages.append(
            {"role": "system", "content": "For all the questions, please only output in the format of: Yes/No"})
        for index_B in top_types_dict:
            if index_A != index_B:
                messages.append({"role": "user",
                                 "content": f"Do entity types {top_types_dict[index_A]} and {top_types_dict[index_B]} match?"})
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages)
                    isSame = response.choices[0].message.content
                    if isSame == 'Yes':
                        index_A_similar.add(index_B)
                        '''
                    else:
                        messages.append({"role": "user", "content": isSame})
                        messages.append({"role": "user",
                                         "content": f"Do entity type {top_types_dict[index_A]} and {top_types_dict[index_B]} exist generally-accepted ancestor-descendant relationship?"})
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages)
                        isParent = response.choices[0].message.content
                        if isParent == 'Yes':
                            messages.append({"role": "user", "content": isParent})
                            messages.append({"role": "user",
                                             "content": f"Is {top_types_dict[index_A]} a generally-accepted ancestor of {top_types_dict[index_B]}?"})
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=messages)
                            isAncestor = response.choices[0].message.content
                            if isAncestor == 'Yes':
                                ancestor.append((index_A, index_B ))
                            else:
                                ancestor.append((index_B, index_A))'''
                except Exception as e:
                    print(f"An error occurred: {e}")
        print([top_types_dict[i] for i in index_A_similar])
        similar_semantics.append(index_A_similar)
    print(similar_semantics, ancestor)
    return similar_semantics, ancestor




graph = nx.DiGraph()
with open(f'result/P1/WDC/All/cl_SCT6_lm_sbert_head_column_0_none_subCol/data_types_no_tp_mention.pickle',
          'rb') as file:
    loaded_data = pickle.load(file)
print(loaded_data)
df = pd.DataFrame(columns=['cluster id', 'cluster type', 'table name', 'table type', 'isChild'])
top_types = {}
for key, cluster_dict in loaded_data.items():
    cluster_type = cluster_dict['cluster type']
    graph.add_node(cluster_type)
    top_types[key]=cluster_type
highLevelGroup(top_types)

highLevelMatch(top_types)


"""

for key, cluster_dict in loaded_data.items():
    message_cluster = []
    cluster_type = cluster_dict['cluster type']
    message_cluster.append({"role": "user", "content":
        "I am building a conceptual schema and have some conceptual entity types inferred from different table clusters." +
        "Each table indicates an entity type, but entity types may have different names while sharing the same semantics. " +
        "Now, I will provide a pair of conceptual entity types. Please tell me if they have the semantics , considering their semantics."})
    for table_name, table_dict in cluster_dict['table types'].items():
        table_type = table_dict['specific type']
        isChild = table_dict['isSubType']
        df = df._append(
            {'cluster id': key, 'cluster type': cluster_type, 'table name': table_name, 'table type': table_type,
             'isChild': isChild},
            ignore_index=True)
print(df)
df.to_csv("result/P1/WDC/All/cl_SCT6_lm_sbert_head_column_0_none_subCol/specific.csv", index=False)
"""


"""
The following is for pair comparison between high level types
"""
'''
pairs = []
for index_A in range(0, len(top_types)):
    messages = []
    messages.append({"role": "user", "content":
        "I am building a conceptual schema and now have some conceptual entity types inferred from different table clusters." +
        "Each table cluster indicates the mutual entity type of tables inside the tables. " +
        f"Now, I will provide a list of pairs of conceptual entity types. While one of the type is always {top_types[index_A]}. Please tell me if they have similar semantics."})
    messages.append(
        {"role": "system", "content": "For all the questions, please only output  in the format of: Yes/No"})


    for index_B in range(index_A + 1, len(top_types)):
        messages.append({"role": "user",
                         "content": f"Does conceptual entity type '{top_types[index_A]} have the same semantics with {top_types[index_B]}?"})
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages)
            isSame = response.choices[0].message.content
            pairs.append(({top_types[index_A]}, {top_types[index_B]},isSame))
            if isSame == 'Yes':
                print({top_types[index_A]}, {top_types[index_B]}, "Similar semantics")
                similar_semantics.append((top_types[index_A], top_types[index_B]))
            else:
                messages.append({"role": "system", "content": isSame})
                messages.append({"role": "user", "content": f"Do entity type {top_types[index_A]} and {top_types[index_B]} exist generally-accepted ancestor-descendant relationship?"})
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages)
                isParent = response.choices[0].message.content
                print(f"{top_types[index_A]} and {top_types[index_B]}|  parent type existence: {isParent} ")
                dissimilar.append( (top_types[index_A],top_types[index_B], isParent ) )

        except Exception as e:
            print(f"An error occurred: {e}")
with open(f'result/P1/WDC/All/cl_SCT6_lm_sbert_head_column_0_none_subCol/toplevelRefine.pickle', 'wb') as file:
    pickle.dump((similar_semantics,dissimilar ), file)'''