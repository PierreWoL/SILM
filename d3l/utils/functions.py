from datetime import datetime
import re
import os
import pickle
import string
from dateutil.parser import parse
import pandas as pd
from typing import Iterable, Any


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

def is_empty(text: str) -> bool:
    empty_representation = ['-', 'NA', 'na', 'n/a', 'NULL', 'null', 'nil', 'empty', ' ', '']
    if text in empty_representation:
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
def strftime_format(format):
    def func(value):
        try:
            datetime.strptime(value, format)
        except ValueError:
            return False
        return True
    func.__doc__ = f'should use date format {format}'
    return func

def is_number(s):
    """
    Return whether the string can be numeric.
    :param string: str, string to check for number
    :param fuzzy: bool, ignore unknown tokens in string if True
    REFERENCE: https://www.runoob.com/python3/python3-check-is-number.html
    """
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
def is_date_expression(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    REFERENCE: https://stackoverflow.com/questions/25341945/check-if-string-has-date-any-format
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

def tokenize(text: str)-> str:
    delimiterPattern = re.compile(r"[^\w\s\-_@&]+")
    ele = re.sub(r"\s+", " ", text.translate(str.maketrans('', '', string.punctuation)).strip().lower())
    return ele



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
