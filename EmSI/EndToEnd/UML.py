import plantuml
import os

def savefig_uml(UML_Code, target_path, fileName = "diagram.uml"):
    plantuml_obj = plantuml.PlantUML(url='http://www.plantuml.com/plantuml/img/')
    generated_diagram_url = plantuml_obj.processes(UML_Code)
    print("Diagram URL:", generated_diagram_url)
    # 或者保存到文件并生成图
    target_file = os.path.join(target_path, fileName)
    with open(target_file, 'w') as file:
        file.write(UML_Code)
    plantuml_path = 'E:\Project\CurrentDataset\plantuml.jar'  # 修改为你的 plantuml.jar 的实际路径
    cmd = f'java -jar {plantuml_path} -tjpg {target_file}'
    plantuml_obj = plantuml.PlantUML(url='http://www.plantuml.com/plantuml/svg/')
    # 生成图像并输出结果
    result = plantuml_obj.processes(UML_Code)
    os.system(cmd)


import requests


def save_svg_from_plantuml(uml_code, target_path, fileName = "diagram.svg"):
    """
    Generates an SVG from PlantUML code and saves it to a file.

    :param uml_code: A string containing PlantUML code.
    :param output_file: The path to the output file where the SVG will be saved.
    """
    # URL of the PlantUML server that generates the SVG
    plantuml_url = 'http://www.plantuml.com/plantuml/svg/'
    output_file = os.path.join(target_path, fileName)
    data = uml_code.encode('utf-8')
    print(data)
    # Encode the PlantUML code in a format that can be sent to the server
    # Create the full URL to request the SVG
    response = requests.post(plantuml_url, data=data)
    # Send a GET request to the server and get the SVG response
    if response.status_code == 200:
        # Write the SVG data to a file
        with open(output_file, 'wb') as file:
            file.write(response.content)
        print(f"SVG file saved to {output_file}")
    else:
        print("Failed to retrieve SVG:", response.status_code)

#generate_uml(uml_code, os.getcwd())