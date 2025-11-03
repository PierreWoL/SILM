import pandas as pd
from scipy.stats import normaltest, skew


def derive_meta_features(col):
    features = {}
    if not col.astype(str).apply(str.isnumeric).all():
        return {"std": round(col.astype(str).str.len().std(), 2), "mean": round(col.astype(str).str.len().mean(), 2),
                "mode": col.astype(str).str.len().mode().iloc[0].item(), "median": col.astype(str).str.len().median(),
                "max": col.astype(str).str.len().max(), "min": col.astype(str).str.len().min(),
                "rolling-mean-window-4": [0.0]}
    col = col.dropna().astype(float)
    if col.apply(float.is_integer).all():
        col = col.astype(int)
    # print(f"Collecting metafeatures for column {col} \n")
    features['std'] = round(col.std(), 2)
    features['mean'] = round(col.mean(), 2)
    features['mode'] = col.mode().iloc[0].item()
    features['median'] = col.median()
    features['max'] = col.max()
    features['min'] = col.min()
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=4)
    return features


def fix_mode(d):
    if isinstance(d['mode'], pd.Series):
        d['mode'] = d['mode'].loc[0].item()
    return d


def split_meta_features(d):
    return pd.Series(
        [d.get('std', "N/A"), d.get('mean', "N/A"), d.get('median', "N/A"), d.get('mode', "N/A"), d.get('max', "N/A"),
         d.get('min', "N/A")])


def analyze_column_distribution(column):
    std = round(column.std(), 2)
    mean = round(column.mean(), 2)
    median = round(column.median(), 2)
    min = round(column.min(), 2)
    max = round(column.max(), 2)
    normal_test_p = normaltest(column).pvalue
    skewness = skew(column)
    return {
        "std": std,
        "min": min,
        "max": max,
        "mean": mean,
        "median": median,
        "skewness": skewness,
        "normal Test p-value": normal_test_p
    }


def numeric_column_features(col: pd.Series):
    """
    We may need to know the specific features for the numeric column
    """
    return analyze_column_distribution(col)
