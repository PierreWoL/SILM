import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import math
from typing import Iterable, Any
import re
import string

def tokenize(text: str) -> str:
    re.compile(r"[^\w\s\-_@&]+")
    textRemovePuc = str(text).translate(str.maketrans('', '', string.punctuation)).strip()
    textRemovenumber = textRemovePuc.translate(str.maketrans('', '', string.digits)).strip()
    ele = re.sub(r"\s+", " ", textRemovenumber)
    return ele


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