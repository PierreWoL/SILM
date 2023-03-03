import math
from datetime import datetime
import re
import os
import pickle
import string
from dateutil.parser import parse
import pandas as pd
from typing import Iterable, Any
from d3l.utils.constants import STOPWORDS
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse


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

    except ValueError:
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
            list_column_tokens.append(' '.join(token_stop_word(item)))
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
