from GeSI.utils import load_template
_Cardinality = """
You are an expert in conceptual schema inference. Given a source type Person that has a conceptual relationship with a \
 target type ‘Company’ through an attribute ‘worksfor’ belonging to ‘Person’, your task is to determine the most \
 plausible relationship cardinality pattern between ‘Person’ and ‘Company’ based on table samples. \
Candidate relationship: Person --(worksfor)--> Company
In a conceptual schema, the cardinality of a binary relationship summarizes how many instances on one side \
can be associated with instances on the other side. For reporting, use only the four canonical patterns: \
1:1 (one-to-one)
1:n (one-to-many, from Person to Company) 
n:1 (many-to-one, from Person to Company) 
m:n (many-to-many)
Note: Optionality (min=0 vs. min=1) is not included in the output label. \
We provide representative tables containing the attribute ‘worksfor’, \
along with sampled rows and per-table statistics of the corresponding ‘worksfor’ related columns.
The column statistics summary includes: 
- Completeness (non-null ratio)
- Multiplicity (multi-valued cell %)
- Uniqueness (unique values / total values)


### EXAMPLE 1 ###
 Movie --(director)--> Person
### TABLE 1 ###
Movie|Rel.|Director
Mission Impossible 3|2006|Abrams, J.J.
Super 8|2011|Abrams, J.J.
The Corporation|2003|Achbar, Mark
Gone Baby Gone|2007|Affleck, Ben
Lion of the Desert|1981|Akkad, Moustapha

### STATISTICS ###
Column director of Attribute director:
- non-null ratio: 1.0
- multi-valued cell %: 0.0625
- unique values/total values: 0.9375
### END STATISTICS ###
m:n
### END TABLE 1 ###
### END EXAMPLE 1 ###

### EXAMPLE 2 ###
 VideoGame --(Publisher)--> Company
### TABLE 1 ###
Rank|Title|Category|Publisher
1|Super Smash Bros. Melee|Fighting|Nintendo
2|Pikmin 2|Strategy/Sim|Nintendo
3|Legend of Zelda: Collector's Edition|RPG|Nintendo
4|Legend of Zelda: The Wind Waker|Action Adventure|Nintendo
5|Metal Gear Solid: Twin Snakes|Action Adventure|Konami
### STATISTICS ###
Column Publisher of Attribute Publisher:
- non-null ratio: 1.0
- multi-valued cell %: 0.0
- unique values/total values: 0.2
### END STATISTICS ###
n:1 
### END TABLE 2 ###

### EXAMPLE 3 ###
Person --(worksfor)--> Company
### TABLE 1 ###
Name|image|jobTitle|worksFor
Gregory Wyatt|https://nordcham.glueup.com/resources/public/...|Director for Business Intelligence|PSA Consultancy Philippines
Ruzel Van Jaarsveld|https://capechamber...|Business Development Manager|Bizfacility (Pty) Ltd
Anna Seefeldt|https://swedchamsg.g...||Pink Pineapple Consulting Pte Ltd
Paul Ryan Balon|https://iia-p.glueup.com/...|Chief Audit Executive|Makati Development Corporation
Todd E. Riddle, DC, CCSP, RKT, CSCS, FAKTR|https://app....|Director of Education|Southeast Sports Seminars
### STATISTICS ###
Column statistics summary of worksFor:
- non-null ratio: 1.0
- multi-valued cell %: 0.0625
- unique values/total values: 0.9375
### END STATISTICS ###
### END TABLE 3 ###


Relationship: {{relationship}}
### TABLE ###
{{table}}
### STATISTICS ###
Column statistics summary of worksFor :
- non-null ratio: {{nnr}}
- multi-valued cell %: {{mvc}} 
- unique values/total values: {{uvtv}}
### END STATISTICS ###
### END TABLE ###
PLEASE ONLY OUTPUT one label from this closed set, with no extra text:
1:1
1:n
n:1
m:n
"""

Cardinality = load_template(_Cardinality)


