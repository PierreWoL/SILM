import plantuml
import os
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
classes = {
    'Customer': (['id', 'name', 'email'], {'Order': '1..*'}),
    'Order': (['order_id', 'order_date', 'customer_id'], {})
}
uml_code = generate_uml(classes)
print(uml_code)

def generate_uml(UML_Code, target_path, fileName = "diagram.uml"):

    # 创建一个PlantUML对象，指定使用公共服务器
    plantuml_obj = plantuml.PlantUML(url='http://www.plantuml.com/plantuml/img/')

    # 使用直接定义的UML代码生成图片，并返回图片的URL
    generated_diagram_url = plantuml_obj.processes(UML_Code)
    print("Diagram URL:", generated_diagram_url)
    # 或者保存到文件并生成图
    target_file = os.path.join(target_path, fileName)
    with open(target_file, 'w') as file:
        file.write(UML_Code)
    plantuml_path = 'E:\Project\CurrentDataset\plantuml.jar'  # 修改为你的 plantuml.jar 的实际路径
    cmd = f'java -jar {plantuml_path} -tjpg diagram.uml'
    os.system(cmd)

generate_uml(uml_code, os.getcwd())