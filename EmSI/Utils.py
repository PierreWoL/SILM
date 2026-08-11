import random
import numpy as np
import SCDection.TableAnnotation as TA
import math
from datetime import datetime
import os
import pickle
import string
from dateutil.parser import parse
import pandas as pd
from typing import Iterable, Any

from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse
import re
import unicodedata

from nltk.corpus import stopwords
STOPWORDS = list(set(stopwords.words('english')))


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


def aug(table: pd.DataFrame):
    exists = []
    for index in range(0, table.shape[1]):
        if are_all_numbers(table.iloc[:, index][0].split(",")) is False:
            exists.append(index)
    return table.iloc[:, exists]


def split(column: pd.Series):
    if "," in column:
        return column.split(",")
    elif "|" in column:
        return column.split("|")
    else:
        return column.split(" ")
        # return column.tolist()


from EmSI.SCDection.TableAnnotation import TableColumnAnnotation as TA


def subjectColDetection(DATA_PATH, RESULT_PATH=None):
    if RESULT_PATH is None:
        folder_path = os.path.abspath(DATA_PATH)
        RESULT_PATH = os.path.dirname(folder_path)
    if 'dict.pickle' in os.listdir(RESULT_PATH):
        with open(os.path.join(RESULT_PATH, 'dict.pickle'), "rb") as f:
            table_dict = pickle.load(f)
    else:
        table_dict = {}
        table_names = [i for i in os.listdir(DATA_PATH) if i.endswith(".csv")]
        for tableName in table_names:
            table_dict[tableName] = []
            table_ori = pd.read_csv(os.path.join(DATA_PATH, tableName), encoding="latin1")
            annotation_table = TA(table_ori, SearchingWeb=False)
            annotation_table.subcol_Tjs()
            table_dict[tableName].append(annotation_table.annotation)
            table_dict[tableName].append(annotation_table.column_score)

        with open(os.path.join(RESULT_PATH, 'dict.pickle'), "wb") as save_file:
            pickle.dump(table_dict, save_file)
    return table_dict


def childList(cut_list, select_length):
    if len(cut_list) < select_length:
        return cut_list
    return random.sample(cut_list, select_length)


def subjectCol(table: pd.DataFrame):
    sub_cols_header = []
    annotation_table = TA(table, SearchingWeb=False)
    annotation_table.subcol_Tjs()
    NE_column_score = annotation_table.column_score
    if len(NE_column_score) > 0:
        max_score = max(NE_column_score.values())
        subcol_index = [key for key, value in NE_column_score.items() if value == max_score]
        for index in subcol_index:
            sub_cols_header.append(table.columns[index])
    return sub_cols_header


def findSubCol(table_path: str, table: str):
    sub_cols_header = []
    df_table = pd.read_csv(os.path.join(table_path, table))
    SubjectCol_dict = subjectColDetection(table_path)
    annotation, NE_column_score = SubjectCol_dict[table]
    if len(NE_column_score) > 0:
        max_score = max(NE_column_score.values())
        subcol_index = [key for key, value in NE_column_score.items() if value == max_score]
        for index in subcol_index:
            sub_cols_header.append(df_table.columns[index])
    del df_table,SubjectCol_dict,annotation, NE_column_score
    return sub_cols_header


from nltk.corpus import wordnet
from collections import defaultdict, Counter


def calculate_similarity(e1, e2, Euclidean=False):
    if Euclidean is False:
        dot_product = np.dot(e1, e2)
        norm_e1 = np.linalg.norm(e1)
        norm_e2 = np.linalg.norm(e2)
        similarity = dot_product / (norm_e1 * norm_e2)
    else:
        similarity = np.linalg.norm(e1 - e2)
    return similarity


# Make sure you've downloaded the necessary resources
# nltk.download('wordnet')
# nltk.download('punkt')


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


def name_type(cluster, name_dict=None):
    if name_dict is None:
        name_i = naming(cluster, threshold=5)
    else:
        names = [str(name_dict[i + ".csv"]) for i in cluster]
        name_i = naming(names, threshold=5)
    return name_i


def naming(InitialNames, threshold=0):
    # Given data
    ### TODO top K name
    GivenEntities = {}
    for name in InitialNames:
        # tokens = nltk.word_tokenize(name)
        # Entities = namedEntityRecognition(tokens)
        Entities = token_stop_word(name)
        if Entities == ['']:
            continue
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
    if threshold == 0:
        threshold = len(InitialNames) / 2
    most_frequent_terms = [term for term, freq in sorted_frequency if freq > threshold]
    if len(most_frequent_terms) == 0:
        most_frequent_terms = [term for term, freq in sorted_frequency if freq > 0][:5]
    # Given data
    return most_frequent_terms



def shingles(value: str) -> Iterable[str]:
    """
    Generate multi-word tokens delimited by punctuation.
        Parameters
        ----------
        value : str
            The value to shingle.

        Returns
        -------
        Iterable[str]
            A generator of shingles.
    """

    delimiterPattern = re.compile(r"[^\w\s\-_@&]+")
    for shingle in delimiterPattern.split(value):
        yield re.sub(r"\s+", " ", shingle.strip().lower())


def is_empty(text) -> bool:
    if isinstance(text, float):
        if str(text) == "nan":
            return True
        if math.isnan(text):
            return True
    empty_representation = ['-', 'NA', 'na', 'nan', 'n/a', 'NULL', 'null', 'nil', 'empty', ' ', '']
    if text in empty_representation:
        return True
    if pd.isna(text):
        return True
    return False


def is_numeric(values: Iterable[Any]) -> bool:
    """
    Check if a given column contains only numeric values.

    Parameters
    ----------
    values :  Iterable[Any]
        A collection of values.

    Returns
    -------
    bool
        All non-null values are numeric or not (True/False).
    """
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    return pd.api.types.is_numeric_dtype(values.dropna())


'''
check if the string is in date expression 
'''


def strftime_format(str_format):
    def func(value):
        try:
            datetime.strptime(value, str_format)
        except ValueError:
            return False
        return True

    func.__doc__ = f'should use date format {str_format}'
    return func


def is_number(s):
    """
    Return whether the string can be numeric.
    :param s: str, string to check for number
    REFERENCE: https://www.runoob.com/python3/python3-check-is-number.html
    """
    if type(s) is bool:
        return False
    if "%" in s:
        s = s.replace("%", "")
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def is_date_expression(text, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param text: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    REFERENCE: https://stackoverflow.com/questions/25341945/check-if-string-has-date-any-format
    """
    try:

        parse(text, fuzzy=fuzzy)
        return True

    except:
        return False


def is_acronym(text: str) -> bool:
    """
    todo: I don't think this cover all kinds of cases, so need to update later
    Return whether the string can be a acronym.
    :param text: str, string to check for acronym
    REFERENCE: https://stackoverflow.com/questions/47734900/detect-abbreviations-in-the-text-in-python
    NOTE: the expression -- r"\b[A-Z\.]{2,}\b" tests if this string contain constant upper case characters
    \b(?:[a-z]*[A-Z][a-z]*){2,} at least two upper/lower case

    Parameters
    ----------
    text
    """
    if text.islower() is False and len(text) < 3:
        return True
    if len(text) < 6:
        rmoveUpper = re.sub(r"\b[A-Z\\.]{2,}\b", "", text)
        removePunc = rmoveUpper.translate(str.maketrans('', '', string.punctuation))
        if removePunc == "":
            return True
    else:
        return False


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def is_id(text: str) -> bool:
    """
    todo: I don't think this cover all kinds of cases, so need to update later
    Return whether the string can be a acronym.
    :param text: str, string to check for acronym
    REFERENCE: https://stackoverflow.com/questions/47734900/detect-abbreviations-in-the-text-in-python
    NOTE: the expression -- r"\b[A-Z\.]{2,}\b" tests if this string contain constant upper case characters
    \b(?:[a-z]*[A-Z][a-z]*){2,} at least two upper/lower case
    """
    if is_number(text) is False:
        removeCharacter = re.sub(r"[a-zA-Z]+", "", text)
        removePunc = removeCharacter.translate(str.maketrans('', '', string.punctuation))
        if is_number(removePunc) is True:
            return True
        else:
            return False
    else:
        # print("this is number!")
        return False


def tokenize_str(text: str) -> str:
    re.compile(r"[^\w\s\-_@&]+")
    textRemovePuc = str(text).translate(str.maketrans('', '', string.punctuation)).strip()
    textRemovenumber = textRemovePuc.translate(str.maketrans('', '', string.digits)).strip()
    ele = re.sub(r"\s+", " ", textRemovenumber)
    return ele


from country_list import countries_for_language


def is_country(string):
    token = tokenize_str(string)
    countries = [x[1] for x in countries_for_language('en')]


def token_stop_word(text) -> list:
    elements = []
    if not is_empty(text):
        lemmatizer = WordNetLemmatizer()
        ele = tokenize_str(text).lower()
        ele_origin = lemmatizer.lemmatize(ele)
        elements = [i for i in ele_origin.split(" ") if i not in STOPWORDS]
    return elements


def remove_stopword(text: str) -> list:
    ele = tokenize_str(text)
    lemmatizer = WordNetLemmatizer()
    elements = [i for i in ele.split(" ") if lemmatizer.lemmatize(i) not in STOPWORDS]
    return elements


def tokenize_with_number(text: str) -> str:
    delimiterPattern = re.compile(r"[^\w\s\-_@&]+")
    textRemovePuc = text.translate(str.maketrans('', '', string.punctuation)).strip()
    ele = re.sub(r"\s+", " ", textRemovePuc)
    return ele


def has_numbers(input_string):
    return bool(re.search(r'\d', input_string))


def bow(sentence):
    bows = {}
    vectorizer = CountVectorizer()
    bow_i = vectorizer.fit_transform([sentence.lower()])
    # Get feature name (vocabulary)
    feature_names = vectorizer.get_feature_names_out()
    for word, index in zip(feature_names, bow_i.toarray()[0]):
        bows[word] = index
    return bows


def nltk_tokenize(text):
    # Use NLTK's tokenizer
    tokens = word_tokenize(text)
    return tokens


def remove_blank(column):
    index_list = []
    if type(column) != pd.Series:
        column = pd.Series(column)
    for index, item in column.items():
        if is_empty(item) is True:
            index_list.append(index)
    if len(index_list) < 1:
        return list(column)
    return column.drop(index_list)


def token_list(column: list):
    list_column_tokens = []
    for item in column:
        tokens = token_stop_word(item)
        if len(tokens) > 0:
            list_column_tokens.append(' '.join(tokens))
    is_blank = True
    for element in list_column_tokens:
        if element != '':
            is_blank = False
            break
    if is_blank is True:
        return None
    else:
        return list_column_tokens


def remove_blanked_token(column):
    column_no_empty = remove_blank(column)
    return token_list(column_no_empty)


def pickle_python_object(obj: Any, object_path: str):
    """
    Save the given Python object to the given path.

    Parameters
    ----------
    obj : Any
        Any *picklable* Python object.
    object_path : str
        The path where the object will be saved.

    Returns
    -------

    """
    parent_dir = "/".join(object_path.split("/")[:-1])
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    try:
        with open(object_path, "wb") as save_file:
            pickle.dump(obj, save_file)
    except Exception:
        raise pickle.PicklingError(
            "Failed to save object {} to {}!".format(str(obj), object_path)
        )


def unpickle_python_object(object_path: str) -> Any:
    """
    Load the Python object from the given path.

    Parameters
    ----------
    object_path : str
        The path where the object is saved.

    Returns
    -------
    Any
        The object existing at the provided location.

    """
    if not os.path.isfile(object_path):
        raise FileNotFoundError("File {} does not exist locally!".format(object_path))
    try:
        with open(object_path, "rb") as save_file:
            obj = pickle.load(save_file)
    except Exception:
        raise pickle.UnpicklingError(
            "Failed to load object from {}!".format(object_path)
        )
    return obj



#change to list and retry in glove_embedding


SYMBPATT = r"\@" + re.escape(
    u"".join(chr(i) for i in range(0xFFFF) if unicodedata.category(chr(i)) == "Sc")
)
PUNCTPATT = r"\!\"\#\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\[\\\]\^\_\`\{\|\}\~"

LOWALPHA = re.compile(r"[a-z]([a-z\-])*")
UPPALPHA = re.compile(r"[A-Z]([A-Z\-\.])*")
CAPALPHA = re.compile(r"[A-Z][a-z]([a-z\-])*")

POSDEC = re.compile(r"\+?[0-9]+(,[0-9]+)*\.[0-9]+")
NEGDEC = re.compile(r"\-[0-9]+(,[0-9]+)*\.[0-9]+")
POSINT = re.compile(r"\+?[0-9]+(,[0-9]+)*")
NEGINT = re.compile(r"\-[0-9]+(,[0-9]+)*")

PUNCT = re.compile(r"[" + PUNCTPATT + r"]+")
SYMB = re.compile(r"[" + SYMBPATT + r"]+", re.UNICODE)
WHITE = re.compile(r"\s+")

ALPHANUM = re.compile(r"(?:[0-9]+[a-zA-Z]|[a-zA-Z]+[0-9])[a-zA-Z0-9]*")
NUMSYMB = re.compile(
    r"(?=.*[0-9,\.])(?=.*[" + SYMBPATT + r"]+)([0-9" + SYMBPATT + r"]+)", re.UNICODE
)

FASTTEXTURL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/"
GLOVEURL = "http://nlp.stanford.edu/data/"