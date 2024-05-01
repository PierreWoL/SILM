from collections import defaultdict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.tokenize import word_tokenize
import math
from datetime import datetime
import re
import string
from dateutil.parser import parse
import pandas as pd
from typing import Iterable, Any
from d3l.utils.constants import STOPWORDS
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse
from country_list import countries_for_language


def convergence(previousState, currentState, threshold=0.05, **kwargs):
    """
        Check if the algorithm has converged based on the entropy difference.
        """
    if previousState is None:
        return False
    return abs(entropy(currentState) - entropy(previousState)) < threshold


def entropy(key_value_pairs):
    """Calculate the entropy of key-value pairs."""

    def calculate_probability(value, total_v):
        """Calculate the probability of a given value."""
        return value / total_v if total > 0 else 0

    total = sum(key_value_pairs.values())
    entropy_value = -sum(calculate_probability(v, total) * np.log2(calculate_probability(v, total))
                         for v in key_value_pairs.values() if v > 0)
    return entropy_value


def I_inf(dataset,
          current_state,
          process,
          update,
          **kwargs):
    """
    Implement the i-inf algorithm in the TableMiner+
    Parameters
    ----------
    dataset : D in the paper representing datasets
    process :
    update :
    and <key, value> if the list has convergence

    Returns
    -------
    the collections of key value pairs ranked by v
    :param dataset:
    :param current_state:
    """
    i = 0
    previous_state = {}
    for index, data_item in enumerate(dataset):
        i += 1
        previous_state = current_state.copy() if current_state is not None else {}
        new_pairs = process(data_item, index, **kwargs)
        current_state = update(current_state, new_pairs, **kwargs)
        # if key in the pairs, update this key value pair, if not, add into key value pairs list
        if previous_state and convergence(current_state, previous_state, **kwargs):
            print("converged!")
            break
    return current_state


"""

from keras.preprocessing.text import Tokenizer


def bow(content):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([content])
    sequences = tokenizer.texts_to_sequences([content])
    word_index = tokenizer.word_index
    bow = {}
    for key in word_index:
        bow[key] = sequences[0].count(word_index[key])
    return bow
    
    

"""


def bow(sentence):
    bows = {}
    vectorizer = CountVectorizer()
    try:
        bow_i = vectorizer.fit_transform([sentence.lower()])
        # Get feature name (vocabulary)
        feature_names = vectorizer.get_feature_names_out()
        for word, index in zip(feature_names, bow_i.toarray()[0]):
            bows[word] = index
    except:
        # Tokenize the sentence into words
        bows = defaultdict(int)
        words = tokenize_str(sentence).split(" ")
        # Normalize words and remove stop words
        words = [word for word in words if word.isalpha()]
        words = [word.lower() for word in words]
        # Count the words
        for word in words:
            bows[word] += 1
        bows = dict(bows)
    return bows


def nltk_tokenize(text):
    # Use NLTK's tokenizer
    tokens = word_tokenize(text)
    return tokens


def def_bow(definitional_sentences):
    # Initialize a default dictionary to hold the word counts
    bow_representation = defaultdict(int)
    # Get the set of English stop words
    stop_words = set(stopwords.words('english'))

    # Process each definitional sentence
    for sentence in definitional_sentences:
        # Tokenize the sentence into words
        words = tokenize_str(sentence).split(" ")
        # Normalize words and remove stop words
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

        # Count the words
        for word in words:
            bow_representation[word] += 1

    return dict(bow_representation)


def keys_with_max_value(d):
    if len(d) == 0:
        return {}
    # find max value
    max_value = max(d.values())
    # collect the keys of which value equal to max value
    max_keys = {k for k, v in d.items() if v == max_value}
    return max_keys


def dice_coefficient(text1, text2):
    """
        Calculate the standard Dice coefficient(entity name score) between two texts.

        :param text1: The first text.
        :param text2: The second text.
        :return: The Dice coefficient score.
        """
    # Create bag-of-words for each text
    if isinstance(text1, str):
        bow1 = bow(text1)
    elif isinstance(text1, dict):
        bow1 = text1
    else:
        print("invalid input")
        return None

    if isinstance(text2, str):
        bow2 = bow(text2)
    elif isinstance(text2, dict):
        bow2 = text2
    else:
        print("invalid input")
        return None
    # Calculate the intersection of the two bags-of-words
    intersection = set(bow1) & set(bow2)
    # Calculate the sum of frequencies in the context for the intersection words
    sum_freq_intersection = sum(bow1[word] + bow2[word] for word in intersection)
    # Calculate the Dice coefficient
    dice_score = (2.0 * sum_freq_intersection) / (sum(bow1.values()) + sum(bow2.values()))
    return dice_score


def union_bags_of_words(bag1, bag2):
    union_bag = {}
    for word, count in bag1.items():
        union_bag[word] = count

    for word, count in bag2.items():
        if word in union_bag:

            union_bag[word] = max(union_bag[word], count)
        else:
            union_bag[word] = count
    return union_bag


def stabilized(current_collection, previous_collection):
    if current_collection == previous_collection:
        return True
    else:
        print(current_collection, previous_collection)
        return False


def is_country(string):
    token = tokenize_str(string)
    countries = [x[1] for x in countries_for_language('en')]


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



