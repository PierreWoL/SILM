from utils import load_template
_TopKFilter = """
You are given an attribute of an entity type and a list of candidate entity types.\
 The attribute includes a name and a set of sample values.
Your task is to: based on the attribute name and the sample values, \
select the Top {{ topk }} most relevant types from the candidate list.\
Each selected type should represent a parent type that could encompass the possible types of all the given values.
Each sample value should be considered an instance of the selected type.
Criteria:



1. The selected types should represent different plausible parent categories that can generalize all the given values.
2. Do not select types that apply only to a subset of the values; \
each selected type must plausibly serve as a parent type for ALL values based on an is-a relationship.


### EXAMPLE 1 ###
Attribute name: founder
Sample values: Elon Musk, Jeff Bezos, Bill Gates
Candidate types: Organization, Place, Person, Event
Person
### END EXAMPLE 1 ###
### EXAMPLE 2 ###
Attribute name: published_in
Sample values: Nature, NeurIPS, ER, ICDE, VLDBJ
Candidate types: Journal, Conference, Organization, Event
Journal, Conference, Event
### END EXAMPLE 2 ###
### EXAMPLE 3 ###
Attribute name: restaurant_name
Sample values:  The Ritz, The Wolseley, Restaurant Gordon Ramsay, The Ivy
Candidate types: Organization, Person, Event
None
### END EXAMPLE 3 ###

Attribute name: {{name}}
Sample values: {{ attrs| join(', ') }}
Candidate types: {{ types | join(', ') }}

Use the same format as the examples above. \
Please ONLY output up to {{ topk }} type names from the candidate list, with no additional text. 
If fewer than {{ topk }} types are appropriate, output fewer than {{topk}} type names.
If none are suitable, output "None".
Do not explain your answer. Only ONLY output up to {{ topk }} top-level type names, separated by commas.
"""
delete = """
Conference, Event
"""
TopKFilter = load_template(_TopKFilter)

_MostSpecificTypes = """
You are given an attribute of an entity type and a list of candidate entity types.\
 The attribute includes a name and a set of sample values.
Your task is to: based on the attribute name and the semantics of the sample values,\
 infer the most specific entity type that accurately describes all of the sample values.
Each sample value should be a plausible instance of the selected type, 

Criteria:
1. The selected type must be the most specific parent type that generalizes all the given values. 
2. Do not return broad, general, or high-level types if a more specific type can precisely capture the meaning of the values.
3. Only return a type if it can plausibly serve as a parent type for all sample values.

### EXAMPLE 1 ###
Attribute name: employer  
Sample values: Google, Microsoft, OpenAI, Amazon, Meta
Candidate types:  Fruit, Organization, Company, University
Company
### END EXAMPLE 1 ###
### EXAMPLE 2 ###
Attribute name: medal
Sample values: Gold, Silver, Bronze
Candidate types: Person, Country, Event, Sport
None
### END EXAMPLE 2 ###
### EXAMPLE 3 ###
Attribute name: city
Sample values:  London, Tokyo, New York, Paris, Berlin
Candidate types: Country, City, Region
City
### END EXAMPLE 3 ###

Attribute name: {{name}}
Sample values: {{ attrs | join(', ') }}
Candidate types: {{ types | join(', ') }}

Answer with the name of the most specific matching candidate type, or "None" if the inferred type is not in the list.
Do not explain your answer. Only return with the name of the most specific matching candidate type.
"""
MostSpecificTypes = load_template(_MostSpecificTypes)

_RelatiosnhipPredicate = """
You are given two conceptual types: a source type and a target type.  
You are also given the name of an attribute from the source type, whose values are instances of the target type.  
Your task is to generate a semantic predicate that best describes the relationship from \
the source type to the target type, as expressed by the attribute.  

### EXAMPLE 1 ###
Source type: Movie
Target type: Person
Attribute name: director
directedBy
### END EXAMPLE 1 ###
### EXAMPLE 2 ###
Source type: Student
Target type: Professor
Attribute name: advisor
hasAdvisor
### END EXAMPLE 2 ###

### EXAMPLE 3 ###
Source type: Company
Target type: City
Attribute name: location
locatedIn
### END EXAMPLE 3 ###
Source type: {{source}}
Target type: {{target}}
Attribute name: {{attr}}

The output should be a single predicate name, with no additional text.  \
Do not explain your answer. Only return the predicate name.
"""
RelatiosnhipPredicate = load_template(_RelatiosnhipPredicate)

_NE_prompt = """
Given a list of values from a table column, determine whether the column is a named entity column.
A named entity is a clearly identifiable and named object, either concrete (e.g., a person, location, product) or abstract (e.g., a paper title, project, or event).
Do not consider abbreviations, codes, or purely descriptive phrases as named entities.
Answer "Yes" only if the majority of the values are clear named entities; otherwise, answer "No".
Values: {{values}}
"""
NE_prompt = load_template(_NE_prompt)
