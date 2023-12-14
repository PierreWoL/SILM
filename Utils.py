import os
import re

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from collections import Counter
import TableAnnotation as TA
from SubjectColumnDetection import ColumnType

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased',
         'sbert': 'sentence-transformers/all-mpnet-base-v2'}

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
def are_all_numbers(values):
    cleaned_values = [val for val in values if
                      pd.notna(val) and str(val).strip() != '' and val not in ["n / a", "n/a", "N/A"]]
    return all(val.isdigit() for val in cleaned_values)

def simplify_string(augment_op):
    string_split_list = augment_op.split(",")
    simplified_elements = [''.join([word[0].upper() for word in element.split("_")]) for element in string_split_list]
    if len(set(simplified_elements)) == 1:
        return f"{simplified_elements[0]}{len(simplified_elements)}"
    else:
        return ",".join(simplified_elements)


def aug(table: pd.DataFrame):
    exists = []
    for index in range(0, table.shape[1]):
        if are_all_numbers(table.iloc[:, index][0].split(",")) is False:
            exists.append(index)
    return table.iloc[:, exists]

def split(column:pd.Series):
        if "," in column:
            return column.split(",")
        elif "|" in column:
            return column.split("|")
        else:
            return column.split(" ")
            #return column.tolist()
def subjectCol(table: pd.DataFrame, combine=False):
    sub_cols_header = []
    anno = TA.TableColumnAnnotation(table, isCombine=combine)
    types = anno.annotation
    for key, type in types.items():
        if type == ColumnType.named_entity:
            sub_cols_header = [table.columns[key]]
            break
    return sub_cols_header
def most_frequent(list1, isFirst=True):
    """
    count the most frequent occurring annotated label in the cluster
    """

    count = Counter(list1)
    if isFirst is True:
        return count.most_common(1)[0][0]
    else:
        most_common_elements = count.most_common()
        max_frequency = most_common_elements[0][1]
        most_common_elements_list = [element for element, frequency in most_common_elements if
                                     frequency == max_frequency]
        return most_common_elements_list
example_clusters = {112: ['2-1180976-1.html.name', '2-1180976-2.html.model'],
                    4: ['2-1180976-1.html.power', '2-1180976-1.html.torque', '2-1415652-1.html.power'],
                    5: ['2-1180976-1.html.size', '1-1212189-1.html.capacity', '2-1212189-1.html.capacity', '2-1530048-1.html.displacement'],
                    84: ['2-1180976-1.html.engine code', '2-1285530-1.html.engine'], 48: ['2-1180976-1.html.0 - 100 km / h , s', '2-1867790-1.html.no built'],
                    36: ['2-1180976-1.html.top speed', '2-1361602-1.html.torque', '2-17941111-1.html.top speed', '2-1118731-1.html.torque', '1-17941111-2.html.top speed', '2-17941111-2.html.top speed', '2-1139122-1.html.top speed'], 88: ['2-1180976-1.html.years', '2-17941111-1.html.years'], 38: ['1-1147705-1.html.model', '1-11167610-1.html.trim', '2-1773703-1.html.model', '1-20595642-2.html.original no', '2-14410430-1.html.model'], 25: ['1-1147705-1.html.max motive power', '1-1147705-1.html.max torque at rpm', '2-1773703-1.html.power'], 27: ['1-1147705-1.html.engine displacement', '2-1176162-6.html.displacement', '1-229917-2.html.tail number', '1-21021796-1.html.displacement', '1-1373768-1.html.displacement', '2-1373768-1.html.displacement', '2-14410430-1.html.displ'], 28: ['1-1147705-1.html.engine type', '2-1361602-1.html.power', '1-11167610-1.html.engine', '2-1176162-5.html.engine code (s)'], 131: ['1-1147705-1.html.engine configuration & notes 0 - 100 km / h'], 81: ['1-2079664-3.html.number', '2-13964981-1.html.cr'], 12: ['1-2079664-3.html.previous br no', '1-2079664-3.html.disposal', '2-1654827-1.html.years', '1-229917-2.html.fatalities'],
                    46: ['1-2079664-3.html.taken into deptal stock', '1-2079664-3.html.withdrawn'], 31: ['2-1654827-1.html.model', '1-229917-2.html.location'],
                    54: ['2-1654827-1.html.engine', '2-1361602-1.html.trim', '2-1361602-1.html.engine'], 33: ['2-1654827-1.html.displacement', '1-21021796-1.html.torque', '2-14410430-1.html.torque'],
                    50: ['2-1654827-1.html.power', '2-1285530-1.html.torque', '1-1373768-1.html.torque', '2-1848560-1.html.power', '2-1373768-1.html.torque', '1-1444201-1.html.power'],
                    74: ['2-1654827-1.html.fuel system', '2-1848560-1.html.fuel system'], 69: ['2-10040530-1.html.year', '2-17748094-1.html.year', '2-1379652-1.html.order'],
                    90: ['2-10040530-1.html.team / chassis', '2-10040530-1.html.engine'], 15: ['2-10040530-1.html.tyres', '2-10040530-1.html.points', '2-17748094-1.html.tyres', '2-17748094-1.html.pts'],
                    78: ['2-1180976-2.html.years', '1-21795986-1.html.original number'], 116: ['2-1180976-2.html.engine code', '2-18964684-1.html.engine code', '2-11867198-1.html.locomotive'],
                    62: ['2-1180976-2.html.power', '2-1180976-2.html.torque'], 108: ['2-1176162-6.html.engine name', '2-1415821-6.html.model'],
                    16: ['2-1176162-6.html.engine configuration', '2-1285530-1.html.displacement', '2-12444110-1.html.lot no', '1-1444201-1.html.year model'],
                    14: ['2-1176162-6.html.max power at rpm', '2-1415821-6.html.max power at rpm', '2-14410430-1.html.power'],
                    6: ['2-1176162-6.html.max torque at rpm', '2-18964684-1.html.torque', '2-1415652-1.html.torque', '2-17941111-1.html.torque', '1-11167610-1.html.torque', '2-1871071-2.html.max torque at rpm', '2-1530048-1.html.max torque at rpm', '2-1139122-1.html.torque'], 29: ['2-1361602-1.html.year', '1-21795986-1.html.year', '2-12992801-1.html.lot no'], 30: ['2-1361602-1.html.epa (2008) city', '1-19332157-1.html.Unnamed: 0', '2-1139122-1.html.combined consumption'],
                    68: ['2-1176162-3.html.model', '1-1176162-3.html.model', '2-17941111-1.html.name', '1-17941111-2.html.name', '2-1176162-5.html.engine name', '2-17941111-2.html.name', '2-1139122-1.html.name'], 137: ['2-1176162-3.html.cylinders / valves', '1-1176162-3.html.cylinders / valves'],
                    132: ['2-1176162-3.html.displacement cc', '1-1176162-3.html.displacement cc'], 8: ['2-1176162-3.html.max power kw (ps) at rpm', '1-1176162-3.html.max power kw (ps) at rpm', '2-1118731-1.html.engine', '1-221315-3.html.max mach', '1-11167610-1.html.transmission', '2-1379652-1.html.built'], 96: ['2-1176162-3.html.max torque (nm) at rpm', '1-1176162-3.html.max torque (nm) at rpm'],
                    7: ['2-1176162-3.html.engine code', '1-1176162-3.html.engine code', '2-17941111-1.html.engine id code (s)', '1-17941111-2.html.engine id code (s)', '2-17941111-2.html.engine id code (s)'], 126: ['2-1176162-3.html.top speed (km / h)', '1-1176162-3.html.top speed (km / h)'], 121: ['2-1176162-3.html.production period', '1-1176162-3.html.production period'], 11: ['2-1415821-6.html.engine', '2-1415821-6.html.fuel system'], 57: ['2-1415821-6.html.displacement', '2-1415821-6.html.max torque at rpm'], 21: ['2-1415821-6.html.valvetrain', '2-1176162-5.html.valvetrain', '2-1871071-2.html.valvetrain', '2-1530048-1.html.valvetrain', '2-1139122-1.html.type'], 59: ['2-11373937-1.html.locomotive', '2-11373937-1.html.entered service', '2-11373937-1.html.gauge', '2-11373937-1.html.livery'], 77: ['2-11373937-1.html.serial no', '2-11867198-1.html.serial no'], 64: ['2-1285530-1.html.power', '2-12992801-1.html.diagram'], 3: ['2-1285530-1.html.redline', '2-1285530-1.html.year', '2-10103151-1.html.weight'], 34: ['2-15177130-1.html.date', '2-15177130-1.html.circuit', '2-15177130-1.html.driver', '2-15177130-1.html.race'], 43: ['2-15177130-1.html.event', '2-12992801-1.html.notes'], 18: ['2-18964684-1.html.model', '2-13964981-1.html.code', '2-13964981-1.html.vehicle'],
                    75: ['2-18964684-1.html.years', '2-14410430-1.html.years'], 120: ['2-18964684-1.html.power', '1-11167610-1.html.power'], 104: ['2-1415652-1.html.engine', '2-1371853-2.html.name', '2-12992801-1.html.mark', '1-1444201-1.html.model'], 117: ['2-1415652-1.html.type', '2-1415652-1.html.displacement'], 37: ['1-229917-2.html.date (ddmmyyyy)', '1-12113888-1.html.built', '1-12113888-1.html.rebuilt'], 26: ['1-229917-2.html.brief description', '2-1530048-1.html.engine'], 89: ['1-1886270-1.html.hr no', '2-1886270-1.html.hr no'], 107: ['1-1886270-1.html.hr name', '2-1886270-1.html.hr name'], 85: ['1-1886270-1.html.cr no', '1-1886270-1.html.lms no', '2-1886270-1.html.cr no', '2-1886270-1.html.lms no'], 113: ['1-1886270-1.html.built', '2-1886270-1.html.built'], 79: ['1-1886270-1.html.works', '2-1886270-1.html.works'], 102: ['1-1886270-1.html.withdrawn', '2-1886270-1.html.withdrawn'], 20: ['2-1371853-2.html.capacity', '2-1871071-2.html.displacement', '2-1371853-1.html.capacity', '2-1139122-1.html.capacity', '1-1444201-1.html.cylinder volume'], 92: ['2-1371853-2.html.power', '2-1371853-1.html.type'], 47: ['2-1371853-2.html.type', '2-17941111-1.html.output', '1-17941111-2.html.output', '2-17941111-2.html.output', '2-1371853-1.html.power'], 35: ['2-1371853-2.html.torque', '1-17941111-2.html.torque', '2-17941111-2.html.torque', '2-1371853-1.html.torque'], 49: ['1-21795986-1.html.uic number', '1-21795986-1.html.name'], 23: ['1-21795986-1.html.constructer', '2-1139122-1.html.power'], 13: ['1-21795986-1.html.constructor number', '2-10103151-1.html.name'], 39: ['1-21795986-1.html.withdrawn', '2-1112993-1.html.year built', '2-13964981-1.html.year'], 91: ['1-21795986-1.html.kilometers worked', '1-20391799-1.html.br no'], 32: ['2-1112993-1.html.class', '2-1112993-1.html.operator', '2-1112993-1.html.no built'], 66: ['2-1112993-1.html.cars per set', '1-1444201-1.html.engine'], 61: ['2-1112993-1.html.unit nos', '2-1118731-1.html.power'], 136: ['1-1212189-1.html.model / engine', '2-1212189-1.html.model / engine'], 0: ['1-1212189-1.html.cylinders / valves', '1-1212189-1.html.torque (nm) / rpm', '2-1212189-1.html.cylinders / valves', '2-1212189-1.html.torque (nm) / rpm'], 97: ['1-1212189-1.html.power / rpm', '2-1212189-1.html.power / rpm'],
                    44: ['2-17941111-1.html.volume', '1-17941111-2.html.volume', '2-1176162-5.html.displacement', '2-17941111-2.html.volume'],
                    118: ['2-17941111-1.html.engine', '1-17941111-2.html.engine', '2-17941111-2.html.engine'], 133: ['2-17941111-1.html.fuel', '1-17941111-2.html.fuel', '2-17941111-2.html.fuel'],
                    72: ['2-17941111-1.html.0 - 100 km / h , s', '2-12992801-1.html.fleet numbers'], 71: ['2-1118731-1.html.generation', '2-1848560-1.html.engine', '2-14410430-1.html.engine'],
                    17: ['2-1118731-1.html.years', '2-1773703-1.html.years', '2-1867790-1.html.model years'], 24: ['2-1118731-1.html.induction', '2-13964981-1.html.displacement', '2-1437522-2.html.width', '2-1773703-1.html.engine', '2-1867790-1.html.engine'], 124: ['2-11867198-1.html.name', '2-1437522-2.html.city'],
                    1: ['2-11867198-1.html.entered service', '2-11867198-1.html.owner', '1-11167610-1.html.performance'], 103: ['2-17557270-1.html.owner', '2-17557270-1.html.class', '2-17557270-1.html.road numbers'],
                    119: ['2-17557270-1.html.number in class', '2-1437522-2.html.number of vehicles'], 95: ['2-17557270-1.html.built', '1-221315-3.html.total flights'], 53: ['2-1759889-1.html.br no', '1-20391799-1.html.builder'],
                    130: ['2-1759889-1.html.lot no', '1-20391799-1.html.built'], 65: ['2-1759889-1.html.date', '1-20391799-1.html.withdrawn', '1-12113888-1.html.scrapped / sold'],
                    58: ['2-1759889-1.html.built at', '1-20391799-1.html.ltsr no', '1-20391799-1.html.ltsr name'], 129: ['2-1759889-1.html.boiler type'], 134: ['1-21021796-1.html.bore', '1-21021796-1.html.stroke'],
                    101: ['1-21021796-1.html.cylinders', '1-21021796-1.html.valves', '1-21021796-1.html.power'], 40: ['1-21021796-1.html.applications', '2-10103151-1.html.capacity', '2-10103151-1.html.power'],
                    2: ['1-221315-3.html.pilot', '1-221315-3.html.organization'], 114: ['1-221315-3.html.usaf space flights', '1-221315-3.html.fai space flights'],
                    67: ['1-221315-3.html.max speed (mph)', '1-221315-3.html.max altitude (miles)', '2-1176162-5.html.max torque at rpm'], 105: ['2-12992801-1.html.builder'],
                    122: ['1-20391799-1.html.mr no', '1-20391799-1.html.lms 1930 no'], 109: ['1-19332157-1.html.type a', '1-19332157-1.html.type b', '1-19332157-1.html.type c'],
                    110: ['1-19332157-1.html.type d', '1-19332157-1.html.type e'], 55: ['1-12113888-1.html.number', '2-1773703-1.html.torque'], 76: ['1-12113888-1.html.builder', '2-12444110-1.html.builder'],
                    127: ['1-12113888-1.html.name as rebuilt'], 60: ['2-13964981-1.html.bore', '2-13964981-1.html.stroke'], 45: ['2-13964981-1.html.max power', '2-12898181-1.html.gwr / br nos'],
                    135: ['1-11167610-1.html.turbo'], 86: ['1-11167610-1.html.fuel delivery'], 128: ['1-17941111-2.html.0 - 100 km / h , s', '2-17941111-2.html.0 - 100 km / h , s'],
                    83: ['1-17941111-2.html.co 2', '2-17941111-2.html.co 2'], 115: ['1-17941111-2.html.years', '2-17941111-2.html.years'], 138: ['1-1373768-1.html.trim', '2-1373768-1.html.trim'],
                    93: ['1-1373768-1.html.engine', '2-1373768-1.html.engine'], 56: ['1-1373768-1.html.power', '2-1848560-1.html.displacement', '2-1373768-1.html.power'],
                    94: ['1-1373768-1.html.transmission', '2-1373768-1.html.transmission'], 87: ['1-1373768-1.html.fuel mileage (latest epa mpg - us )', '2-1373768-1.html.fuel mileage (latest epa mpg - us )'],
                    106: ['2-10103151-1.html.cyl', '2-10103151-1.html.bore'], 52: ['2-1848560-1.html.model', '2-1848560-1.html.years'], 19: ['1-22481967-1.html.date', '1-22481967-1.html.withdrawn'],
                    10: ['1-22481967-1.html.builder', '2-17748094-1.html.team / chassis', '2-17748094-1.html.engine'], 70: ['1-22481967-1.html.type', '1-22481967-1.html.status'],
                    100: ['1-22481967-1.html.operator', '1-22481967-1.html.number'], 123: ['2-1176162-5.html.max power at rpm', '2-1871071-2.html.max power at rpm', '2-1530048-1.html.max power at rpm'],
                    9: ['2-12444110-1.html.diagram', '2-12444110-1.html.built', '2-12444110-1.html.fleet numbers'], 51: ['2-1437522-2.html.operator', '2-1437522-2.html.type designation'],
                    82: ['2-1871071-2.html.model', '2-1530048-1.html.model', '2-1371853-1.html.name'], 22: ['2-1871071-2.html.engine', '2-1139122-1.html.code'],
                    99: ['2-1871071-2.html.fuel system', '2-1530048-1.html.fuel system'], 63: ['2-1871071-2.html.years', '2-1530048-1.html.years'],
                    73: ['2-1773703-1.html.chassis code', '2-1867790-1.html.chassis code'],
                    80: ['1-20595642-2.html.lner no (intermediate no)', '1-20595642-2.html.br no', '1-20595642-2.html.name'],
                    98: ['1-20595642-2.html.rebuild date', '1-20595642-2.html.withdrawn'],
                    41: ['2-1379652-1.html.serial numbers', '2-1379652-1.html.quantity', '2-1379652-1.html.1st no'],
                    111: ['1-1444201-1.html.torque', '1-1444201-1.html.fuel system'],
                    42: ['2-12898181-1.html.year', '2-12898181-1.html.rr class', '2-12898181-1.html.rr nos', '2-12898181-1.html.builder', '2-12898181-1.html.builder nos'],
                    125: ['2-1867790-1.html.model']}
import nltk
from nltk.corpus import wordnet
from collections import defaultdict

# Make sure you've downloaded the necessary resources
#nltk.download('wordnet')
#nltk.download('punkt')


# Tokenize and Simple Named Entity Recognition

# Tokenize and Named Entity Recognition (simple version)
def namedEntityRecognition(tokens):
    # Here, we consider every token as an entity for simplicity
    return set(tokens)

# Get synonyms (using WordNet)
def synonyms(entity):
    syns = set()
    for syn in wordnet.synsets(entity):
        for lemma in syn.lemmas():
            syns.add(lemma.name())
    return syns

def naming(InitialNames, threshold= 0):
    # Given data
    GivenEntities = {}

    for name in InitialNames:
        name = re.sub(r'\d', '', name)
        tokens = nltk.word_tokenize(name)
        Entities = namedEntityRecognition(tokens)
        Synonyms = set()
        for entity in Entities:
            Synonyms.update(synonyms(entity))
        GivenEntities[name] = (Entities, Synonyms)

    # Find the most frequently occurring values in Entities and Synonyms
    frequency = defaultdict(int)
    for name, (Entities, Synonyms) in GivenEntities.items():
        for entity in Entities:
            frequency[entity] += 1
        for syn in Synonyms:
            frequency[syn] += 1

    sorted_frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

    # Select terms that appear in more than 50% of the values
    if threshold ==0:
        threshold = len(InitialNames) / 2
    most_frequent_terms = [term for term, freq in sorted_frequency if freq > threshold]
    if len(most_frequent_terms)==0:
        most_frequent_terms = [term for term, freq in sorted_frequency if freq > 0]
        if len(most_frequent_terms)==0:
            most_frequent_terms =[""]
            #
    #print("Most Frequent Terms:", most_frequent_terms)
    # Given data
    return most_frequent_terms

names = ["Premier League", "Scottish Premiership", "Division 1", "Champions League"]
for index, cluster in example_clusters.items():
  attribute_cluster =[i.split("html.")[1] for i in cluster]
  name_attribute = naming(attribute_cluster)
