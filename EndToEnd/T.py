import os
import pickle

from EndToEnd.UML import savefig_uml
from Utils import mkdir


# from UML import savefig_uml,save_svg_from_plantuml


# generateU(classes)


def class_define(class_name, attributes):
    class_code = f'class {class_name} {{\n'
    # print("attributes:", attributes)
    for attr in attributes:
        class_code += f'    +{attr}\n'
    class_code += '}\n'
    return class_code


def relationship_define(ParentClass, ChildClass, relationship_type=0):
    relationship_code = ""
    if ParentClass == ChildClass:
        return relationship_code
    if relationship_type == 0:
        relationship_code += f'{ParentClass} --> {ChildClass}\n'
    elif relationship_type == 1:
        relationship_code += f'{ParentClass} --> 1..* {ChildClass}\n'
    elif relationship_type == 2:
        relationship_code += f'{ParentClass}  --|> {ChildClass}\n'
    else:
        print("--Currently does not support other relationship type")
    return relationship_code


def subType(G, N, attribute_dict, subAttribute_name):
    subType_name = G.nodes[N].get('name')[0].title() if len(G.nodes[N]['name']) > 0 else 'UnnamedClass'
    attributes_index = G.nodes[N].get("attributes")
    attributes_subType = [subAttribute_name]
    for attri_index in attributes_index:
        attris = attribute_dict[attri_index]["name"]
        if len(attris) > 0:
            if attris[0] not in attributes_subType:
                attributes_subType.append(attris[0])
    subtype_code = class_define(subType_name,  list(set(attributes_subType)))
    return subtype_code


def childRelationship(G, N, ParentNode):
    subType_name = G.nodes[N].get('name')[0].title() if len(G.nodes[N]['name']) > 0 else 'UnnamedClass'
    print(N, subType_name)
    Parent_name = G.nodes[ParentNode].get('name')[0].title() if len(G.nodes[ParentNode]['name']) > 0 else 'UnnamedClass'
    relationship = relationship_define(Parent_name, subType_name, relationship_type=2)
    return relationship


def generateUML(TypeDict, dataset, number):
    """
    Generate a PlantUML class diagram from a dictionary of classes.

    :param classes: A dictionary where keys are class names and values are tuples containing
                    a list of attributes and a dictionary of relationships.
                    Format: {'ClassName': (['attribute1', 'attribute2'], {'RelatedClassName': '1..*'})}
    :return: A string containing the PlantUML code.
    """
    uml_code = '@startuml\n'
    sub_umls = {}
    relationship_code = ""
    test_num = 0
    for key, type_info in TypeDict.items():
        tree = type_info['tree']
        class_name = type_info["name"][0]
        test_num += 1
        # print("class_name",class_name)
        subject_attribute_names = type_info["subjectAttribute"]['name']
        subAttri_name = subject_attribute_names[0] if subject_attribute_names else class_name + ' name'
        attributes_info = type_info["attributes"]
        attributes = [subAttri_name]
        for key, info in attributes_info.items():
            names = info["name"]
            if names:
                attributes.append(names[0])
        class_naming = class_name+"_P" # if tree is None else class_name + "_Parent"
        class_type = class_define(class_naming, list(set(attributes)))
        uml_code += class_type
        types_id = type_info["relationship"].keys()

        for type_id in types_id:
            class_name_other = TypeDict[type_id]["name"][0]+"_P"
            relationship = relationship_define(class_naming, class_name_other)
            if relationship not in relationship_code:
                relationship_code += relationship

        if tree is not None:
            nodes = [i for i in tree.nodes() if
                     tree.nodes[i].get('type') != 'data']
            uml_code_inside = '@startuml\n' + class_type
            relationship_inside = ""
            for node in nodes:
                uml_code_inside += subType(tree, node, attributes_info, subAttri_name)
                successors = tree.successors(node)
                successors = [i for i in successors if tree.nodes[i].get('type') != 'data']
                for successor in successors:
                    child_relationship = childRelationship(tree, successor, node)
                    if child_relationship not in relationship_inside:
                        relationship_inside += child_relationship
            top_nodes = [i for i in nodes if tree.in_degree(i) == 0]
            for node in top_nodes:
                Type_name = tree.nodes[node].get('name')[0].title() if len(
                    tree.nodes[node]['name']) > 0 else 'UnnamedClass'

                relationship = relationship_define(class_naming, Type_name, relationship_type=2)
                if relationship not in relationship_inside:
                    relationship_inside += relationship
            uml_code_inside += relationship_inside
            uml_code_inside += '@enduml'
            print(uml_code_inside)
            path_target = os.path.join(os.getcwd(), f"result/EndToEnd/Decomposition/{dataset}/{number}")
            mkdir(path_target)
            savefig_uml(uml_code_inside, path_target, fileName=f"{key}_inside.uml")
            sub_umls[key] = uml_code_inside

    uml_code += relationship_code
    uml_code += '@enduml'
    # print(relationship_code)
    print(uml_code)
    return uml_code, sub_umls

