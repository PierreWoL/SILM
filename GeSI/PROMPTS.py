from utils import load_template

_PROMPT_TEMPLATE = """Table: {{ table }}"""
PROMPT_TEMPLATE = load_template(_PROMPT_TEMPLATE)
_PROMPT_TEMPLATE_FULL = """
The following is a table. Your task is to assign this table to a suitable entity type hierarchy. \
An entity type is typically defined as a word or a short phrase representing the overarching category\
 or domain shared by the entities described in the table rows.  \
 An entity type hierarchy is represented by a collection of paths starting from the generic root type,  "Thing," \
to a specific type most appropriate for the table. As you move from the root to the leaf, \
 the topics in the hierarchy should progressively become more specific. \


{% if examples|length > 0 %}
{% for example in examples %}
### EXAMPLE {{ loop.index }} ###
### TABLE ###
Table: {{ example['table'] }}
### END TABLE ###
{% for path in example['paths'] %}
{{ path | join(" -> ") }}
{% endfor %}
### END EXAMPLE {{ loop.index }} ###
{% endfor %}
{% else %}
You must answer in the format of:
Thing -> Broad type 1 -> Subtype 1 -> ... -> Most specific type 1 
Thing -> Broad type 2 -> Subtype 2 -> ... -> Most specific type 2 
...
{% endif %}

### TABLE ###
Table: {{ table }}
### END TABLE ###

Provide an entity type hierarchy for the above table.  \
{% if examples|length > 0 %}
Use the same format as the examples above. Please only output the result hierarchy.
{% else %}
Use the format described above. Please only output the result hierarchy.
{% endif %}"""
PROMPT_TEMPLATE_FULL = load_template(_PROMPT_TEMPLATE_FULL)

#Abstract types refer to vague or generic categories that do not convey concrete meaning.

_PROMPT_TEMPLATEABS_FULL = """
The following is a table. Your task is to assign this table to a suitable entity type hierarchy. \
An entity type is typically defined as a word or a short phrase representing the overarching category\
 or domain shared by the entities described in the table rows.  \
 An entity type hierarchy is represented by a collection of paths starting from the generic root type,  "Thing," \
to a specific type most appropriate for the table. As you move from the root to the leaf, \
 the topics in the hierarchy should progressively become more specific. \
The top-level type directly under "Thing" must not be an abstract type, i.e. \
{{ abs_list | join(', ') }}.

{% if examples|length > 0 %}
{% for example in examples %}
### EXAMPLE {{ loop.index }} ###
### TABLE ###
Table: {{ example['table'] }}
### END TABLE ###
{% for path in example['paths'] %}
{{ path | join(" -> ") }}
{% endfor %}
### END EXAMPLE {{ loop.index }} ###
{% endfor %}
{% else %}
You must answer in the format of:
Thing -> Broad type 1 -> Subtype 1 -> ... -> Most specific type 1 
Thing -> Broad type 2 -> Subtype 2 -> ... -> Most specific type 2 
...
{% endif %}

### TABLE ###
Table: {{ table }}
### END TABLE ###

Provide an entity type hierarchy for the above table.  \
{% if examples|length > 0 %}
Use the same format as the examples above. Please only output the result hierarchy.
{% else %}
Use the format described above. Please only output the result hierarchy.
{% endif %}"""
PROMPT_TEMPLATEABS_FULL = load_template(_PROMPT_TEMPLATEABS_FULL)



_CONSTRAINT_FET = "**Lowest-level type is given**: The leaf node (lowest level) in each path" \
                 " must be the specific type(s) that is provided."
_CONSTRAINT_SET = "**Fixed second-level type if multiple paths**: " \
                 "If the hierarchy contains multiple paths, the second-level type in all paths " \
                 "must be the same fixed type."

_PROMPT_TEMPLATE_CONSTRAINTS_FULL = """
The following are a table and its most specific entity type(s). Your task is to assign this table to a suitable entity type hierarchy. \
An entity type is typically defined as a word or a short phrase representing the overarching category\
 or domain shared by the entities described in the table rows.  \
 An entity type hierarchy is represented by a collection of paths starting from the generic root type,  "Thing," \
to the given specific type(s) of the table. As you move from the root to the leaf, \
 the types in the hierarchy should progressively become more specific. \
 
**Lowest-level type is given**: The leaf type (lowest level) in each inferred path \
must be the specific type(s) that is provided.
**Fixed second-level type if multiple paths**: \
If the inferred hierarchy of the table contains multiple paths, the second-level type in all paths \
must be the same fixed type.

{% if examples|length > 0 %}
{% for example in examples %}
### EXAMPLE {{ loop.index }} ###
### TABLE ###
Table: {{ example['table'] }}
### END TABLE ###

### LEAF_TYPE ###
Most specific type: {{ example['paths'] | map('last') | join(', ') }}
### END LEAF_TYPE ###
{% for path in example['paths'] %}
{{ path | join(" -> ") }}
{% endfor %}
### END EXAMPLE {{ loop.index }} ###
{% endfor %}
{% else %}
You must answer in the format of:
Thing -> Broad type 1 -> Subtype 1 -> ... -> Most specific type 1 
Thing -> Broad type 1 -> Subtype 2 -> ... -> Most specific type 2 
...
{% endif %}

### TABLE ###
Table: {{ table }}
### END TABLE ###

### LEAF_TYPE ###
Most specific type: {{ fet }}
### END LEAF_TYPE ###

Provide an entity type hierarchy for the above table and its given leaf type.  \
{% if examples|length > 0 %}
Use the same format as the examples above. Please only output the result hierarchy.
{% else %}
Use the format described above. Please only output the result hierarchy.
{% endif %}"""
PROMPT_TEMPLATE_CONSTRAINT_FULL = load_template(_PROMPT_TEMPLATE_CONSTRAINTS_FULL)

_PROMPT_TEMPLATE_CONSTRAINTSTL_FULL = """
The following is a table. Your task is to assign this table to a suitable entity type hierarchy. \
An entity type is typically defined as a word or a short phrase representing the overarching category\
 or domain shared by the entities described in the table rows.  \
 An entity type hierarchy is represented by a collection of paths starting from the generic root type,  "Thing," \
to a specific type most appropriate for the table. As you move from the root to the leaf, \
 the topics in the hierarchy should progressively become more specific. \

**Fixed second-level type if multiple paths**: \
If the inferred hierarchy of the table contains multiple paths, the second-level type in all paths \
must be the same fixed type.

{% if examples|length > 0 %}
{% for example in examples %}
### EXAMPLE {{ loop.index }} ###
### TABLE ###
Table: {{ example['table'] }}
### END TABLE ###

{% for path in example['paths'] %}
{{ path | join(" -> ") }}
{% endfor %}
### END EXAMPLE {{ loop.index }} ###
{% endfor %}
{% else %}
You must answer in the format of:
Thing -> Broad type 1 -> Subtype 1 -> ... -> Most specific type 1 
Thing -> Broad type 1 -> Subtype 2 -> ... -> Most specific type 2 
...
{% endif %}

### TABLE ###
Table: {{ table }}
### END TABLE ###

Provide an entity type hierarchy for the above table.  \
{% if examples|length > 0 %}
Use the same format as the examples above. Please only output the result hierarchy.
{% else %}
Use the format described above. Please only output the result hierarchy.
{% endif %}"""
PROMPT_TEMPLATE_CONSTRAINTSTL_FULL = load_template(_PROMPT_TEMPLATE_CONSTRAINTSTL_FULL)




_PROMPT_TEMPLATE_CONSTRAINTSET_FULL = """
The following are a table and its most specific entity type(s). Your task is to assign this table to a suitable entity type hierarchy. \
An entity type is typically defined as a word or a short phrase representing the overarching category\
 or domain shared by the entities described in the table rows.  \
 An entity type hierarchy is represented by a collection of paths starting from the generic root type,  "Thing," \
to the given specific type(s) of the table. As you move from the root to the leaf, \
 the types in the hierarchy should progressively become more specific. \
 
**Lowest-level type is given**: The leaf type (lowest level) in each inferred path \
must be the specific type(s) that is provided.

{% if examples|length > 0 %}
{% for example in examples %}
### EXAMPLE {{ loop.index }} ###
### TABLE ###
Table: {{ example['table'] }}
### END TABLE ###

### LEAF_TYPE ###
Most specific type:  {{ example['paths'] | map('last') | join(', ') }}
### END LEAF_TYPE ###

{% for path in example['paths'] %}
{{ path | join(" -> ") }}
{% endfor %}
### END EXAMPLE {{ loop.index }} ###
{% endfor %}
{% else %}
You must answer in the format of:
Thing -> Broad type 1 -> Subtype 1 -> ... -> Most specific type 1 
Thing -> Broad type 2 -> Subtype 2 -> ... -> Most specific type 2 
...
{% endif %}

### TABLE ###
Table: {{ table }}
### END TABLE ###

### LEAF_TYPE ###
Most specific type: {{ fet }}
### END LEAF_TYPE ###

Provide an entity type hierarchy for the above table and its given leaf type.  \
{% if examples|length > 0 %}
Use the same format as the examples above. Please only output the result hierarchy. \
Please only output the result hierarchy. Please ONLY OUTPUT \
THE RESULT WITH NO ADDITIONAL TEXTS.
{% else %}
Use the format described above. Please only output the result hierarchy. Please ONLY OUTPUT \
THE RESULT WITH NO ADDITIONAL TEXTS.
{% endif %}"""
PROMPT_TEMPLATE_CONSTRAINTSET_FULL = load_template(_PROMPT_TEMPLATE_CONSTRAINTSET_FULL)

_COT_ROUND_ONE_TEMPLATE = """The following is a table. \
Briefly break down the entity types (both specific and general concepts) relevant to this table. Explain your reasoning step by step.

### TABLE ###
Table: {{ table }}
### END TABLE ###
"""
COT_ROUND_ONE_TEMPLATE = load_template(_COT_ROUND_ONE_TEMPLATE)
_COT_ROUND_TWO_TEMPLATE = """Your task is to assign this table to a suitable entity type hierarchy. \
An entity type is typically defined as a word or a short phrase representing the overarching category\
 or domain shared by the entities described in the table rows.  \
 An entity type hierarchy is represented by a collection of paths starting from the generic root type,  "Thing," \
to a specific type most appropriate for the table. As you move from the root to the leaf, \
 the topics in the hierarchy should progressively become more specific. \
 
You must answer in the format of:
Thing -> Broad type 1 -> Subtype 1 -> ... -> Most specific type 1 
Thing -> Broad type 2 -> Subtype 2 -> ... -> Most specific type 2 
...
"""
COT_ROUND_TWO_TEMPLATE = load_template(_COT_ROUND_TWO_TEMPLATE)

_PRED_CONCEPT_TEMPLATE = """The following is a table. \
Suggest the entity types that this table belongs to. \
An entity type is typically defined as a word or a short phrase representing the overarching category\
 or domain shared by the entities described in the table rows.  \
You should include both general and specific categories.

### TABLE ###
Table: {{ table }}
### END TABLE ###

Give your answer as a comma-separated string.
"""
PRED_CONCEPT_TEMPLATE = load_template(_PRED_CONCEPT_TEMPLATE)

_RESPONSE_TEMPLATE = """{% for path in paths %}
{{ path | join(" -> ") }}
{% endfor %}"""
RESPONSE_TEMPLATE = load_template(_RESPONSE_TEMPLATE)
RESPONSE_REGEX = r"(Thing( -> [\w()\-\–\—,.?!/\\&\"\'+=\[\]\{\} ]+)+\n)*(Thing( -> [\w()\-\–\—,.?!/\\&\"\'+=\[\]\{\} ]+)+)\n?"

_MISTRAL_TEMPLATE = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
MISTRAL_TEMPLATE = load_template(_MISTRAL_TEMPLATE)
