from utils import load_template

_ATTRPROMPT_TEMPLATE_FULL = """
The following is a table and its most specific entity type. \
Your task is to infer attribute names for each column in the given table considering both column’s header, \
context and table’s most specific type. An entity type is typically defined as a word or a short phrase representing \
the overarching category or domain shared by the entities described in the table rows.   

An attribute name should be a concise and semantically meaningful label that reflects not only the type of data\
in the column but also its specific role in the table’s context. When multiple columns share similar value types,\
 attribute names should distinguish their roles by leveraging contextual cues from the table context \
 and its most specific entity type.


{% if examples|length > 0 %}
{% for example in examples %}
### EXAMPLE {{ loop.index }} ###
### TABLE ###
Table: {{ example['table'] }}
### END TABLE ###
### TYPE ###
Most specific type: {{ example['type'] }} 
### END TYPE ###
{{ example['attrs'] | join(',') }}

### END EXAMPLE {{ loop.index }} ###
{% endfor %}
{% else %}
Attribute1,Attribute2,Attribute3
...
{% endif %}

### TABLE ###
Table: {{ table }}
### END TABLE ###
### TYPE ###
Most specific type: {{ type }} 
### END TYPE ###

Provide attribute names for the above table columns.  
{% if examples|length > 0 %}
Use the same format as the examples above. Please ONLY output the ALL attribute names for each column in the given table, \
in order, separated by commas, with no additional text.

{% else %}
Please output only the inferred attribute names for each column in the given table, \
in order, separated by commas, with no additional text.
{% endif %}"""
ATTRPROMPT_TEMPLATE_FULL = load_template(_ATTRPROMPT_TEMPLATE_FULL)

_ATTRPROMPT_TEMPLATESPL_FULL = """
The following is a column, its belonging table and its most specific entity type. \
Your task is to infer attribute name for the given column in the given table considering both column’s header, \
context and table’s most specific type. An entity type is typically defined as a word or a short phrase representing \
the overarching category or domain shared by the entities described in the table rows.   

An attribute name should be a concise and semantically meaningful label that reflects not only the type of data\
in the column but also its specific role in the table’s context. When multiple columns share similar value types,\
 attribute names should distinguish their roles by leveraging contextual cues from the table context \
 and its most specific entity type.


{% if examples|length > 0 %}
{% for example in examples %}
### EXAMPLE {{ loop.index }} ###
### COLUMN ###

Header: {{example['header']}}; Column: {{example['col'] }}
### END COLUMN ###
### TABLE ###
Table: {{ example['table'] }}
### END TABLE ###
### TYPE ###
Most specific type: {{ example['type'] }} 
### END TYPE ###
{{example['attr']}}

### END EXAMPLE {{ loop.index }} ###
{% endfor %}
{% else %}
Attribute1,Attribute2,Attribute3
...
{% endif %}
### COLUMN ###
Header:{{ header}} ;  Column: {{ col}}
### END COLUMN ###
### TABLE ###
Table: {{ table }}
### END TABLE ###
### TYPE ###
Most specific type: {{ type }} 
### END TYPE ###

Provide attribute name for the given column.  
{% if examples|length > 0 %}
 Please ONLY output the attribute name for the given column, with no additional text.

{% else %}
 Please ONLY output the attribute name for the given column, with no additional text.
{% endif %}"""
ATTRPROMPT_TEMPLATESPL_FULL = load_template(_ATTRPROMPT_TEMPLATESPL_FULL)

# Abstract types refer to vague or generic categories that do not convey concrete meaning.


_ARPROMPT_TEMPLATE_FULL = """
You are given a list of attribute names inferred from table columns, \
 which may be semantically redundant, paraphrased, or differently phrased. \
 Your task is to group those names into unified conceptual attribute groups, and assign a clear,\
  canonical attribute name to each group. Please output python dictionary format, where: \
the key is the canonical attribute name of the group,\
the value is the list of original attribute names grouped under that attribute.\
Be careful to distinguish different roles (e.g., supplier vs. consumer) \
even if they share similar semantics (like company names).

### EXAMPLE 1 ###
### ATTRIBUTE NAMES ###
{{ea}}
### END ATTRIBUTE NAMES ###
Output:
{{eao}}
 ### END EXAMPLE 1 ###
 
### ATTRIBUTE NAMES ###
Attribute names: {{ attrs }} 
### END ATTRIBUTE NAMES ###
Provide ONLY output unified conceptual attribute names in the format of Python dictionary, with no additional text. \
You must ensure that **every** attribute name from the input is included in the output!
Use the same format as the examples above. \

"""
ARPROMPT_TEMPLATE_FULL = load_template(_ARPROMPT_TEMPLATE_FULL)
_REST_PROMPT = """
You are given a list of attribute names (i.e., unified conceptual groups),\
 and a separate list of attribute names that were not yet assigned to any group.

Your task is to assign each missing attribute to one of the **existing canonical attribute groups**. \
If an attribute does not match any existing group, create a new canonical group for it.

### EXAMPLE 1 ###
### EXISTING CANONICAL ATTRIBUTES ###
["Event Name", "Event Date"]
### END ATTRIBUTE NAMES ###

### MISSING ATTRIBUTE NAMES ###
["event_status","status", "venue_name", "venue", "event_time", "event_start_date"]
### END MISSING ATTRIBUTE NAMES ###

{
  "Event Date": ["event_start_date"],
  "Event Time": ["event_time"],
  "Event Status": ["event_status","status"],
  "Event Venue": ["venue_name","venue"]
}

### END EXAMPLE 1 ###
 
 
### EXISTING CANONICAL ATTRIBUTES ###
{{keys}}
### END EXISTING CANONICAL ATTRIBUTES ###

### MISSING ATTRIBUTE NAMES ###
{{missing}}
### END MISSING ATTRIBUTE NAMES ###
Output a Python dictionary, where:
– Each key is a canonical attribute name (either chosen from the provided list or newly created),
– Each value is a list of one or more of the provided missing attribute names grouped under that key.

Only include groups for the given missing attributes — do not repeat any previously grouped attributes or their canonical names.

Provide ONLY A VALID Python dictionary as output, with no additional text.
"""
REST_PROMPT = load_template(_REST_PROMPT)

_MERGE_PROMPT = """
You are given a list of attribute names collected from several types that share a common parent type in a type hierarchy.\
Your task is to group semantically similar attribute names together.\
For each group, assign a clear and meaningful canonical name. \
If an attribute does not belong to any groups, ignore it.

### EXAMPLE 1 ###
### ATTRIBUTE NAMES ###
Supplier, Vendor, Factory Name, Production Site, Buyer, Customer
### END ATTRIBUTE NAMES ###
### PARENT TYPE ###
Company 
### END PARENT TYPE ###

Output: {"Supplier Company":["Supplier", "Vendor"],\
 "Production Location":["Factory Name", "Production Site"], \
 "Consumer Company":["Buyer", "Customer"]}\
 
### END EXAMPLE 1 ###
### ATTRIBUTE NAMES ###
Attribute names: {{attrs}}
### END ATTRIBUTE NAMES ###
### PARENT TYPE ###
{{type}}
### END PARENT TYPE ###
Provide only output unified conceptual attributes. Use the same format as the examples above.

Output a dictionary where:
- Each key is a canonical attribute name to be assigned to the parent type;
- Each value is a list of original attribute names from the related types that are semantically similar. 

Provide ONLY A VALID Python dictionary as output, with no additional text.
"""
MERGE_PROMPT = load_template(_MERGE_PROMPT)
