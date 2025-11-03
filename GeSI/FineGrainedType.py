from utils import load_template

_PROMPT_TEMPLATE = """Table: {{ table }}"""
PROMPT_TEMPLATE = load_template(_PROMPT_TEMPLATE)
_PROMPT_TEMPLATE_FULL = """
The following is a table. Your task is to infer the most fine-grained entity type
 that best describes the rows in this table. \
 
{% if examples|length > 0 %}
{% for example in examples %}
### EXAMPLE {{ loop.index }} ###
### TABLE ###
Table: {{ example['table'] }}
### END TABLE ###
{{ example['type'] }}
### END EXAMPLE {{ loop.index }} ###
{% endfor %}
{% else %}
...
{% endif %}
### TABLE ###
Table: {{ table }}
### END TABLE ###

Provide the most fine-grained entity type for the above table.  \
{% if examples|length > 0 %}
Use the same output format as the examples above. Provide only the entity type without explanation. \
The inferred type should be a single-word or short-phrase.

{% else %}
Provide only the entity type without explanation. The inferred type should be a single-word or short-phrase.
{% endif %}"""
PROMPT_TEMPLATE_FULL = load_template(_PROMPT_TEMPLATE_FULL)

#Use the same output format as the examples above.
