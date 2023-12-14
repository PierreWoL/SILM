
from dataclasses import dataclass
import matplotlib.pyplot as plt




import erdantic as erd


from dataclasses import dataclass
import networkx as nx
import textwrap
from textwrap import dedent

import matplotlib.patches as mpatches


# this has a problem
def draw_network_with_shapes_on_top(G, pos, ax):
    # Draw edges first so that shapes cover the edges
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False)

    for n in G:
        c = 'skyblue' if G.nodes[n]['type'] == 'entity' else 'lightgreen'
        if G.nodes[n]['type'] == 'entity':
            # Draw rectangles for entities
            box = mpatches.FancyBboxPatch(
                (pos[n][0]-0.15, pos[n][1]-0.06),
                0.3, 0.12,
                boxstyle="round,pad=0.02",
                ec="black",
                fc=c,
                zorder=1,  # Set zorder to 1 for shapes to be on top
            )
            ax.add_patch(box)
        else:
            # Draw ellipses for attributes
            ellipse = mpatches.Ellipse(
                (pos[n][0], pos[n][1]),
                0.3, 0.12,
                ec="black",
                fc=c,
                zorder=1  # Set zorder to 1 for shapes to be on top
            )
            ax.add_patch(ellipse)
        # Add node labels
        plt.text(pos[n][0], pos[n][1], n, ha='center', va='center', zorder=2)  # Set zorder to 2 for text








def dataclass_writing(class_ER:dict):
    print(class_ER.keys())
    created_classes = {}
    class_code_all =''
    for entity, details in class_ER.items():
        #print(entity)
        """attributes = "\n".join([f"    {name}: {type_}" for name, type_ in details['attributes']])
        foreign_keys = "\n".join([f"    {tuple_attri[0]}: {tuple_attri[1]}" for tuple_attri in details['foreign_keys']]) if 'foreign_keys' in details.keys() else ' '
        fields = attributes + '\n' + foreign_keys if details['attributes'] or details['foreign_keys'] else "    pass"
        class_code = f"@dataclass\nclass {entity}:\n{fields}"
        class_code_all +="\n"+ class_code"""
        attributes = "\n".join([f"    {name}: {type_}" for name, type_ in details['attributes']])
        foreign_keys = "\n".join(
            [f"    {name}: {type_}" for name, type_ in details['foreign_keys']]) if 'foreign_keys' in details else ''
        fields = (attributes + '\n' + foreign_keys).strip()
        fields = fields if fields else "pass"

        class_code = f"@dataclass\nclass {entity.capitalize()}:\n    {fields}\n"
        class_code_all += class_code
        class_code_all += class_code
    print(class_code_all)
    exec(class_code_all, globals())
    # Retrieve class definitions
    for entity in class_ER:
        created_classes[entity] = globals()[entity.capitalize()]
    return created_classes





def dataclass_writing2(class_ER: dict):
    created_dataclasses = {}
    for entity, details in class_ER.items():
        attributes = "\n".join([f"    {name}: {type_}" for name, type_ in details['attributes']])
        foreign_keys = "\n".join([f"    {tuple_attri[0]}: {tuple_attri[1]}" for tuple_attri in details['foreign_keys']]) if 'foreign_keys' in details.keys() else ''
        fields = attributes + ('\n' + foreign_keys if foreign_keys else '')

        class_code = f"@dataclass\nclass {entity}:\n{fields}"
        print(class_code)
        try:
            exec(class_code, globals()) #created_dataclasses
        except NameError as e:
            print(f"Error when creating class {entity}: {e}")

    return created_dataclasses


classER = {}

classER['Person'] = {}
classER['Person']['attributes'] = [('Name', 'str'),('country', 'str'),('telephone', 'int'),('Year', 'int'),('sameAs', 'str'),
                                  ('birthDate', 'str'),('birthPlace', 'str'),('jobTitle', 'str'),('affiliation', 'str')]



classER['Swimmer'] = {}
classER['Swimmer']['attributes'] = [('Name', 'str'),('Country', 'str'),('telephone', 'int'),('Year', 'int'),('sameAs', 'str'),
                                  ('birthDate', 'str'),('birthPlace', 'str'),('jobTitle', 'str'),('affiliation', 'str')]
classER['Scientist'] = {}
classER['Scientist']['attributes'] = [('Name', 'str'),('Year', 'str'),('Lifetime', 'int'),('Laureate', 'int'),('Contribution', 'str'),
                                  ('Country', 'str'),('Rationale', 'str')]
classER['Rulers'] = {}
classER['Rulers']['attributes'] = [('Name', 'str'),('Lifetime', 'str'),('reign', 'int'),("Anmerkungen","str")]
classER['Saints'] = {}
classER['Saints']['attributes'] = [('Saint', 'str'),('Year', 'int')]

classER['Animal'] = {}
classER['Animal']['attributes'] = [('Name', 'str'),('Class', 'str'),('animal', 'str'),('Distribution', 'str'),('figure', 'str')]

classER['Birdwatching'] = {}
classER['Birdwatching']['attributes'] = [('name', 'str')]


classER['Musicalbum'] = {}
classER['Musicalbum']['attributes'] = [('name', 'str'),('Genre', 'str'),('Price', 'str'),('url', 'str')]


classER['Book'] = {}
classER['Book']['attributes'] = [('Title', 'str'),('author', 'str'),('availability', 'str'),('bookFormat', 'str'),('datePublished', 'str'),
                                  ('description', 'str'),('inLanguage', 'str'),('isbn', 'int'),('numberOfPages', 'int'),('price', 'int'),
                                 ('priceCurrency', 'str'),('ratingValue', 'int'),('reviewCount', 'int'),('url', 'str')]
classER['Book']['foreign_keys'] = [('publisher', 'Person')] #publisher

classER['TVEpisode'] = {}
classER['TVEpisode']['attributes'] = [('Title', 'str'),('duration', 'str'),('partOfSeries', 'int'),
                                      ('url', 'str')]
classER['TVEpisode']['foreign_keys'] = [('director', 'Person')]

classER['Movie'] = {}
classER['Movie']['attributes'] = [('Rank', 'int'),('Title', 'str'),('Year', 'int'),('datePublished', 'str'),
                                  ('image', 'str'),('genre', 'str'),('ratingCount', 'int'),('actor', 'str'),('duration', 'str'),
                                  ('url', 'str'), ('Country', 'str'),('description', 'str')]
classER['Movie']['foreign_keys'] = [('director', 'Person')]

classER['Recipe'] = {}
classER['Recipe']['attributes'] = [('recipeYield', 'str'),('recipeIngredient', 'str'),('sugarContent', 'int'),('saturatedFatContent', 'str'),
                                  ('description', 'str'),('datePublished', 'str'),('prepTime', 'int'),('recipeCategory', 'str'),('url', 'str'),
                                  ('sodiumContent', 'int'), ('performTime', 'int'),('servingSize', 'int'),('carboHydrateContent', 'int')]
classER['Movie']['foreign_keys'] = [('author', 'Person')]

classER['VideoGame'] = {}
classER['VideoGame']['attributes'] = [('Title', 'str'),('Publisher', 'str'),('Year', 'str'),('Genre', 'int'),('Release', 'str'),
                                  ('rating', 'int'),('Developer', 'str'),('Publisher', 'int'),('Platform', 'str'),('#', 'int'),('Kommentar', 'str')]




G = nx.DiGraph()
G.add_node('Person', type="entity")
G.add_node('Organism', type="entity")
G.add_node('CreativeWork', type="entity")
for entity,attributes in classER.items():
    G.add_node(entity, type="entity")
    if entity in  ['Person', 'Swimmer', 'Scientist', 'Rulers', 'Saints'] and entity!='Person':
        G.add_edge(entity,'Person', type="relationship")
    elif entity in ['Animal', 'Birdwatching']:
        G.add_edge(entity, 'Organism', type="relationship")
    elif entity in ['Musicalbum', 'Book', 'TVEpisode', 'Movie', 'Recipe', 'VideoGame']:
        G.add_edge(entity, 'CreativeWork', type="relationship")
    #for tupe_att in attributes['attributes']:
        #G.add_node(tupe_att[0], type="attribute")
       # G.add_edge(tupe_att[0], entity, type="relationship")

    if 'foreign_keys' in attributes.keys():
        for tupe_att in attributes['foreign_keys']:
            G.add_node(tupe_att[0], type="attribute")
            G.add_edge(tupe_att[0], tupe_att[1], type="relationship")
pos = nx.shell_layout(G)

# Draw the graph
fig, ax = plt.subplots(figsize=(25, 25))
draw_network_with_shapes_on_top(G, pos, ax)
"""plt.axis('off')
plt.show()"""

pos = nx.spring_layout(G, iterations=50)
# Draw the graph with the generated layout
draw_network_with_shapes_on_top(G, pos, ax)
plt.axis('off')
plt.show()

# 确保在使用Person类之前已经定义了Person类
created_classes = dataclass_writing(classER)

dataclass_types = list(created_classes.values())
# Create the ER diagram
diagram = erd.create(*dataclass_types)

# Save the diagram to a file
diagram.draw("my_diagram.png")





"""
G = nx.DiGraph()
G.add_node("City", type="entity")
G.add_node("population", type="attribute", attributeType='int')
G.add_node("city", type="attribute", attributeType='str')
G.add_node("Country", type="entity")
G.add_node("country", type="attribute", attributeType='str')
# Add relationships
G.add_edge( "population", "City",type="relationship")
G.add_edge( "city", "City",type="relationship")
G.add_edge( "country","City", type="relationship")
G.add_edge( "country","Country", type="relationship")

G.add_edge( "country","Country", type="PKFK")"""

"""
@dataclass
class Person:
    Name: str
    country: str
    telephone: int
    Year: int
    sameAs: str
    birthDate: str
    birthPlace: str
    jobTitle: str
    affiliation: str


@dataclass
class Animal:
    Name: str
    Class: str
    animal: str
    Distribution: str
    figure: str


@dataclass
class Birdwatching:
    name: str


@dataclass
class Swimmer:
    Name: str
    Country: str
    telephone: int
    Year: int
    sameAs: str
    birthDate: str
    birthPlace: str
    jobTitle: str
    affiliation: str


@dataclass
class Scientist:
    Name: str
    Year: str
    Lifetime: int
    Laureate: int
    Contribution: str
    Country: str
    Rationale: str


@dataclass
class Rulers:
    Name: str
    Lifetime: str
    reign: int
    Anmerkungen: str


@dataclass
class Saints:
    Saint: str
    Year: int


@dataclass
class Musicalbum:
    name: str
    Genre: str
    Price: str
    url: str


@dataclass
class Book:
    Title: str
    author: str
    availability: str
    bookFormat: str
    datePublished: str
    description: str
    inLanguage: str
    isbn: int
    numberOfPages: int
    price: int
    priceCurrency: str
    ratingValue: int
    reviewCount: int
    url: str
    publisher: Person


@dataclass
class Tvepisode:
    Title: str
    duration: str
    partOfSeries: int
    url: str
    director: Person


@dataclass
class Movie:
    Rank: int
    Title: str
    Year: int
    datePublished: str
    image: str
    genre: str
    ratingCount: int
    actor: str
    duration: str
    url: str
    Country: str
    description: str
    author: Person


@dataclass
class Recipe:
    recipeYield: str
    recipeIngredient: str
    sugarContent: int
    saturatedFatContent: str
    description: str
    datePublished: str
    prepTime: int
    recipeCategory: str
    url: str
    sodiumContent: int
    performTime: int
    servingSize: int
    carboHydrateContent: int


@dataclass
class Videogame:
    Title: str
    Publisher: str
    Year: str
    Genre: int
    Release: str
    rating: int
    Developer: str
    Publisher: int
    Platform: str
    #: int
    Kommentar: str

diagram = erd.create(Person, Animal, Birdwatching, Swimmer, Scientist, Rulers,
                     Saints, Musicalbum, Book, Tvepisode, Movie, Recipe, Videogame)
diagram.draw("my_diagram.png")
#created_classes = create_dataclasses_from_digraph(G)


"""