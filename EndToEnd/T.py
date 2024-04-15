import pickle


def generate_uml(classes):
    """
    Generate a PlantUML class diagram from a dictionary of classes.

    :param classes: A dictionary where keys are class names and values are tuples containing
                    a list of attributes and a dictionary of relationships.
                    Format: {'ClassName': (['attribute1', 'attribute2'], {'RelatedClassName': '1..*'})}
    :return: A string containing the PlantUML code.
    """
    uml_code = '@startuml\n'

    # Define classes and attributes
    for class_name, class_info in classes.items():
        attributes, relationships = class_info
        uml_code += f'class {class_name} {{\n'
        for attr in attributes:
            uml_code += f'    +{attr}\n'
        uml_code += '}\n'

    # Define relationships
    for class_name, class_info in classes.items():
        _, relationships = class_info
        for related_class, multiplicity in relationships.items():
            uml_code += f'{class_name} --> "{multiplicity}" {related_class}\n'

    uml_code += '@enduml'
    return uml_code


# Example usage:
"""classes = {
    'Customer': (['id', 'name', 'email'], {'Order': '1..*'}),
    'Order': (['order_id', 'order_date', 'customer_id'], {})
}"""


# with open("result/WDCEndtoEnd.pkl", 'rb') as f:
#   cluster_dict_all = pickle.load(f)


def class_define(class_name, attributes):
    class_code = ""
    f'class {class_name} {{\n'
    for attr in attributes:
        class_code += f'    +{attr}\n'
        class_code += '}\n'
    return class_code


def relationship_define(ParentClass, ChildClass, relationship_type=0):
    relationship_code = ""
    if relationship_type == 0:
        relationship_code += f'{ParentClass} --> {ChildClass}\n'
    elif relationship_type == 1:
        relationship_code += f'{ParentClass} --> 1..* {ChildClass}\n'
    elif relationship_type == 2:
        relationship_code += f'{ParentClass}  --|> {ChildClass}\n'
    else:
        print("--Currently does not support other relationship type")
    return relationship_code


def subType(G, N, attribute_dict):
    subType_name = G.nodes[N].get('name')[0].title() if len(G.nodes[N]['name']) > 0 else '?'
    attributes_index = G.nodes[N].get("attributes")
    attributes_subType = []
    for attri_index in attributes_index:
        attri = attribute_dict[attri_index]["name"]
        if attri in attributes_subType:
            attributes_subType.append(attri)
    subtype_code = class_define(subType_name, attributes_subType)
    return subtype_code


"""
    relationships_code = ""
    if TopType is not None:
        relationships_code += relationship_define(TopType, subType_name, relationship_type=2)
    else:
        parents_type = G.predecessors(N)
        for parent in parents_type:
            Type_name = G.nodes[parent].get('name')[0].title() if len(G.nodes[parent]['name']) > 0 else '?'
            relationships_code += relationship_define(Type_name, subType_name, relationship_type=2)"""


def generateUML(TypeDict):
    """
    Generate a PlantUML class diagram from a dictionary of classes.

    :param classes: A dictionary where keys are class names and values are tuples containing
                    a list of attributes and a dictionary of relationships.
                    Format: {'ClassName': (['attribute1', 'attribute2'], {'RelatedClassName': '1..*'})}
    :return: A string containing the PlantUML code.
    """
    uml_code = '@startuml\n'
    for key, type_info in TypeDict.items():
        tree = type_info['tree']
        class_name = type_info["name"][0]
        attributes_info = type_info["attributes"]
        ###TODO find subject columns of top level types
        attributes = [info["name"] for key, info in attributes_info.items()]
        class_define(class_name, list(set(attributes)))

        if tree is not None:
            nodes = [i for i in tree.nodes() if
                     tree.nodes[i].get('type') != 'data' and tree.nodes[i].get('attributes') != []]
            for node in nodes:
                subType(tree, node, attributes_info)
            """Top_layer = [i for i in tree.nodes() if tree.in_degree(i) == 0]
            for node in Top_layer:
                if tree.nodes[node].get('type') != 'data' and  tree.nodes[node].get('attributes')!=[]:
                    subType(tree, node, type_info["attributes"], TopType=class_name)
                    successors = list(tree.successors(node))
                    successors = [i for i in successors if tree.nodes[i].get('type') != 'data']"""
            # no child conceptual model
            # if len(successors) != 0:
            # for successor in successors:
            # if tree.nodes[node].get('type') != 'data' and tree.nodes[node].get('attributes') != []:

    uml_code += '@enduml'
    return uml_code
