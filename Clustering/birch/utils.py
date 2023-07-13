# Authors: Roman Yurchak
#
# License: BSD 3 clause

import numpy as np
from sklearn.utils.validation import check_array


def _binary_linkage2clusters(linkage, n_samples):
    """ Given a list of elements of size n_sample and a linkage matrix
    linking some of those samples, compute the cluster_id of each element

    Parameters
    ----------

    linkage : array (n_pairs, 2)
       arrays indicating binary links between elements
    n_samples : int
       total number of elements

    Returns
    -------
    labels : array (n_samples)
    """

    if (linkage > n_samples).any():
        raise ValueError
    if (linkage < 0).any():
        raise ValueError
    if n_samples < 0:
        raise ValueError

    dmap = {}
    idx = 0
    for a, b in linkage:
        if a in dmap:
            cid = dmap[a]
        elif b in dmap:
            cid = dmap[b]
        else:
            cid = idx
            idx += 1
        dmap[a] = cid
        dmap[b] = cid

    labels = np.zeros(n_samples, dtype=np.int)
    cid = 0
    for idx in range(n_samples):
        if idx in dmap:
            labels[idx] = n_samples + dmap[idx]
        else:
            labels[idx] = cid
            cid += 1
    _, labels_renamed = np.unique(labels, return_inverse=True)
    return labels_renamed


def _merge_clusters(X, rename=False):
    """
    Compute a union of all clusters

    Used to determine which cluster_id a document should belong to
    if at least one of it's lexicons suggest that it's a duplicate

    Approximate time complexity O(n_samples*n_features)

    Parameters
    ----------
     X: array (n_samples, n_features)
       input arrays with the cluster id's to merge
     rename : binary
       make sure the output array is between 0 and len(unique(cluster_id))

    Parameters
    ----------
      cluster_id: array (n_samples)
    """
    X = check_array(X, ensure_2d=True)

    n_samples, n_features = X.shape

    y = np.zeros(n_samples, dtype=X.dtype)

    out = {}

    for (i_idx, X_row) in enumerate(X):
        for j_idx, X_el in enumerate(X_row):
            if (j_idx, X_el) in out:
                res = out[(j_idx, X_el)]
                break
        else:
            res = X_row[0]  # use the 1st columnt index for this cluster id

        for (j_idx, X_el) in enumerate(X_row):
            out[(j_idx, X_el)] = res

        y[i_idx] = res
    if rename:
        _, labels_renamed = np.unique(y, return_inverse=True)
        return labels_renamed
    else:
        return y



def centroid_similarity(X, internal_ids, nn_metric='cosine'):
    """ Given a list of documents in a cluster, compute the cluster centroid,
    intertia and individual distances

    Parameters
    ----------
    internal_ids : list
      a list of internal ids
    nn_metric : str
      a rescaling of the metric if needed
    """

    from sklearn.metrics.pairwise import pairwise_distances

    X_sl = X[internal_ids, :]
    centroid = X_sl.mean(axis=0)

    if centroid.ndim == 1:
        centroid = centroid[None, :]

    S_cos = 1 - pairwise_distances(X_sl, centroid, metric='cosine')
    S_sim = _scale_cosine_similarity(S_cos, metric=nn_metric)
    S_sim_mean = np.mean(S_sim)
    return float(S_sim_mean), S_sim[:, 0]




def _scale_cosine_similarity(x, metric='cosine', inverse=False):
    """ Given a cosine similarity on L2 normalized data,
    appriximately convert it to Jaccard similarity, and/or
    normalize it to the [0, 1] interval

    Parameters
    ----------
    x : {float, ndarray}
      the cosine similarity value
    metric : str
      the conversion to apply one of ['cosine', 'jaccard']
    inverse : bool
      perform the inverse de-normalization operation
    """
    valid_metrics = ['cosine', 'jaccard', 'cosine_norm', 'jaccard_norm',
                     'cosine-positive']
    if metric not in valid_metrics:
        raise ValueError('metric {} not supported, must be in {}'
                         .format(metric, valid_metrics))
    if metric == 'cosine':
        return x
    elif metric == 'cosine-positive':
        if isinstance(x, (int, float)):
            return max(x, 0.0)
        else:
            return np.fmax(x, 0.0)

    if metric.startswith('jaccard'):
        if not inverse:
            x = cosine2jaccard_similarity(x)
        else:
            x = jaccard2cosine_similarity(x)

    if metric.endswith('norm'):
        x = _normalize_similarity(x, metric=metric.split('_')[0],
                                  inverse=inverse)

    return x

def jaccard2cosine_similarity(s_jac):
    """ Given a cosine similarity on L2 normalized data,
    compute the jaccard similarity

    Parameters
    ----------
    s_jac : {float, ndarray}
      the cosine similarity

    Returns
    -------
    s_cos : {float, ndarray}
      the Jaccard similarity
    """
    return 2*s_jac / (1 + s_jac)


def _normalize_similarity(x, metric='cosine', inverse=False):
    """Given a similarity score, normalize it to the
    [0, 1] range

    Parameters
    ----------
    x : {float, ndarray}
      the similarity score
    metric : str
      the metric used (one of 'cosine', 'jaccard')
    inverse : bool
      perform the inverse de-normalization operation
    """
    if metric == 'cosine':
        # cosine similarity can be in the [-1, 1] range
        if not inverse:
            return (x + 1)/2
        else:
            return 2*x - 1
    elif metric == 'jaccard':
        # jaccard similarity could potenitally be in the [-1/3, 1] range
        # when using the cosine2jaccard_similarity function
        if not inverse:
            return (3*x + 1)/4.
        else:
            return (4*x - 1)/3.
    else:
        raise ValueError