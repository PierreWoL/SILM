from utils import load_template

_COL_TEMP_FULL = """
The following is a column, and its belonging table's entity type. Your task is to assign this column to \
 a suitable column semantic type hierarchy. \
A column type is a semantic label that describes the real-world meaning of the values in a column,\
 typically corresponding to either entity types (e.g., person, organization) or literal types \
 (e.g., date, currency, number).\
A column type hierarchy is represented by a collection of paths starting from the generic root type, \
 "Thing," \
to a specific type most appropriate for the column. As you move from the root to the leaf, \
 the topics in the hierarchy should progressively become more specific.


{% if examples|length > 0 %}
{% for example in examples %}
### EXAMPLE {{ loop.index }} ###
### COLUMN ###
Header: {{example['header']}}; Column values: {{example['col'] }}
### END COLUMN ###

### TABLE TYPE ###
Table most specific type: {{ example['type'] }}
### END TABLE TYPE ###

{{example['path']}}
### END EXAMPLE {{ loop.index }} ###
{% endfor %}
{% else %}
You must answer in the format of:
Thing -> Broad type 1 -> Subtype 1 -> ... -> Most specific type 1 
Thing -> Broad type 2 -> Subtype 2 -> ... -> Most specific type 2 
...
{% endif %}


### COLUMN ###
Header:{{ header}} ;  Column: {{ col}}
### END COLUMN ###

### TABLE TYPE ###
Table most specific type: {{ type }}
### END TABLE TYPE ###

Provide a column semantic type hierarchy for the above column.  \
{% if examples|length > 0 %}
Use the same format as the examples above. Please only output the result hierarchy.
{% else %}
Use the format described above. Please only output the result hierarchy.
{% endif %}"""
COL_TEMP_FULL = load_template(_COL_TEMP_FULL)
_COL_TEMP_TFULL = """
The following is a column, and its surrounding columns context and it table's entity type. \
Your task is to assign this column to \
 a suitable column semantic type hierarchy. \
A column type is a semantic label that describes the real-world meaning of the values in a column,\
 typically corresponding to either entity types (e.g., person, organization) or literal types \
 (e.g., date, currency, number).\
A column type hierarchy is represented by a collection of paths starting from the generic root type, \
 "Thing," \
to a specific type most appropriate for the column. As you move from the root to the leaf, \
 the topics in the hierarchy should progressively become more specific.


{% if examples|length > 0 %}
{% for example in examples %}
### EXAMPLE {{ loop.index }} ###
### COLUMN ###
Header: {{example['header']}}; Column values: {{example['col'] }}
### END COLUMN ###

### BELONGING TABLE ###
Column: {{ example['table'] }}
### END BELONGING TABLE ###

### TABLE TYPE ###
Table most specific type: {{ example['type'] }}
### END TABLE TYPE ###

{{example['path']}}
### END EXAMPLE {{ loop.index }} ###
{% endfor %}
{% else %}
You must answer in the format of:
Thing -> Broad type 1 -> Subtype 1 -> ... -> Most specific type 1 
Thing -> Broad type 2 -> Subtype 2 -> ... -> Most specific type 2 
...
{% endif %}


### COLUMN ###
Header:{{ header}} ;  Column: {{ col}}
### END COLUMN ###

### BELONGING TABLE ###
Table most specific type: : {{ table }}
### END BELONGING TABLE ###


### TABLE TYPE ###
Table most specific type: {{ type }}
### END TABLE TYPE ###

Provide a column semantic type hierarchy for the above column.  \
{% if examples|length > 0 %}
Use the same format as the examples above. Please only output the result hierarchy.
{% else %}
Use the format described above. Please only output the result hierarchy.
{% endif %}"""
COL_TEMP_TFULL = load_template(_COL_TEMP_TFULL)