"""
This module exposes the LSH indexing functionality.
Based on https://github.com/ekzhu/datasketch and D3L
and the LSH Forest paper <http://ilpubs.stanford.edu:8090/678/1/2005-14.pdf>.
"""
from collections import Counter, defaultdict
import struct
from index.Hash.MinHash import MinHash
from typing import Any, ByteString, Iterable, List, Optional, Tuple, Union
import numpy as np
from index.LSH.LSHAbstract import LSHAbstract


class LSH:
    """
    The :ref:`minhash_lsh` index.
    It supports query with `Jaccard similarity`_ threshold.
    Reference: `Chapter 3, Mining of Massive Datasets
    <http://www.mmds.org/>`_.

    Args:
        threshold (float): The Jaccard similarity threshold between 0.0 and
            1.0. The initialized MinHash LSH will be optimized for the threshold by
            minizing the false positive and false negative.
        num_perm (int, optional): The number of permutation functions used
            by the MinHash to be indexed. For weighted MinHash, this
            is the sample size (`sample_size`).
        weights (tuple, optional): Used to adjust the relative importance of
            minimizing false positive and false negative when optimizing
            for the Jaccard similarity threshold.
            `weights` is a tuple in the format of
            :code:`(false_positive_weight, false_negative_weight)`.
        params (tuple, optional): The LSH parameters (i.e., number of bands and size
            of each bands). This is used to bypass the parameter optimization
            step in the constructor. `threshold` and `weights` will be ignored
            if this is given.


    Note:
        `weights` must sum to 1.0, and the format is
        (false positive weight, false negative weight).
        For example, if minimizing false negative (or maintaining high recall) is more
        important, assign more weight toward false negative: weights=(0.4, 0.6).
        Try to live with a small difference between weights (i.e. < 0.5).
    """

    def __init__(self,
                 hash_size,
                 threshold=0.85,
                 num_perm=128,
                 weights=(0.5, 0.5),
                 params=None,
                 dimension: [int] = None
                 ):

        self._hash_size = hash_size
        self._dimension = dimension
        self._buffer_size = 50000
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.h = num_perm
        if params is not None:
            self._b, self._r = params
            if self._b * self._r > num_perm:
                raise ValueError("The product of b and r in params is "
                                 "{} * {} = {} -- it must be less than num_perm {}. "
                                 "Did you forget to specify num_perm?"
                                 .format(self._b, self._r, self._b * self._r, num_perm))
        else:
            false_positive_weight, false_negative_weight = weights
            LSH_abstract = LSHAbstract()
            self._fp_fn_weights = weights
            self._threshold = threshold
            self._b, self._r = LSH_abstract.optimal_param(threshold, num_perm,
                                                          false_positive_weight, false_negative_weight)
            self._hashTables = [defaultdict(set) for _ in range(self._b)]
            self._hashRanges = [(i * self._r, (i + 1) * self._r) for i in range(self._b)]
            print("self._hashTables", self._hashTables)
            print("self._hashRanges", self._hashRanges)
            self._keys = defaultdict(list)

        self._hashfunc = self._hash_generator = MinHash()

    @property
    def hash_generator(self):
        return self._hashfunc

    @property
    def hash_size(self) -> int:
        return self._hash_size

    @property
    def dimension(self) -> [int]:
        return self._dimension

    @property
    def fp_fn_weights(self):
        return self._fp_fn_weights

    @property
    def keys(self) -> dict:
        return self._keys

    @property
    def similarity_threshold(self) -> float:
        return self._threshold

    @property
    def lsh_parameters(self) -> [float, float]:
        return self._b, self._r

    @property
    def hashtables(self) -> List[defaultdict]:
        return self._hashTables

    def _get_lsh_keys(self, input_hash: Iterable[int]) -> List[ByteString]:
        """
        Transform a given hashcode into a collection of index keys.

        Parameters
        ----------
        input_hash : List[int]
            The hashcode to transform

        Returns
        -------
        ByteString
            A collection of LSH index keys represented as a bytestring.
        """

        hash_chunks = [
            bytes(np.array(input_hash[start:end]).byteswap().data)
            for start, end in self._hashRanges
        ]
        return hash_chunks

    def _get_hash(self, input_id: str) -> Optional[np.ndarray]:
        """
        Reconstruct the hash back from the index keys.
        This should be used only if the index keys originate from hash hashcode.

        Parameters
        ----------
        input_id : str
            The item identifier that is already indexed.

        Returns
        -------
        Optional[np.ndarray]
            The item hashcode as a Numpy array or None if the item has not been indexed.

        """

        hashcode = self._keys.get(input_id, None)
        if hashcode is None:
            return None

        hashcode = b"".join(hashcode)
        original_hash = []
        for counter in range(0, len(hashcode), 8):
            chunk = struct.unpack("=Q", hashcode[counter: counter + 8])[0]
            original_hash.append(np.array(chunk).byteswap())

        return np.array(original_hash, dtype=np.uint64)

    def get_similarity_score(
            self,
            left_element: Union[Iterable[Any], str],
            right_element: Union[Iterable[Any], str],
    ) -> float:
        """
        Estimate the similarity between the two sets.

        Parameters
        ----------
        left_element : Union[Iterable[Any], str]
            The id of an already indexed element or a hashcode.
        right_element : Union[Iterable[Any], str]
            The id of an already indexed element or a hashcode.

        Returns
        -------
        np.float16
            The estimated similarity score.

        """

        if isinstance(left_element, str):
            left_hashcode = self._get_hash(left_element)
        else:
            left_hashcode = left_element

        if isinstance(right_element, str):
            right_hashcode = self._get_hash(right_element)
        else:
            right_hashcode = right_element

        if left_hashcode is None or right_hashcode is None:
            return 0.0

        max_size = min([left_hashcode.size, right_hashcode.size])
        return np.float16(
            np.count_nonzero(left_hashcode[:max_size] == right_hashcode[:max_size])
        ) / np.float16(max_size)

    def add(self, input_id: str, input_set: Iterable) -> bool:
        """
        Add a new item to the index.

        Parameters
        ----------
        input_id : str
            The id that will identify the input item as a string.
            Only the ids are stored in the index buckets.
        input_set : Iterable
            Since this is a set-based index, the *input* has to be an iterable.
            It will be chunked into multiple keys that will be added to the index.

        Returns
        -------
        bool
            True if the item has been successfully added, False otherwise.

        """

        if input_id in self._keys:
            raise ValueError("Input identifier already used: {}".format(input_id))

        input_hash = self._hash_generator.hash(input_set, hashvalues=None)

        if len(input_hash) != self._hash_size:
            raise ValueError(
                "The resulting input hash has inconsistent length. Expected {} but got {}".format(
                    self._hash_size, len(input_hash)
                )
            )

        hash_chunks = self._get_lsh_keys(input_hash)
        self._keys[input_id] = hash_chunks
        for hash_entry, hash_table in zip(hash_chunks, self._hashTables):
            hash_table[hash_entry].add(input_id)
        return True

    def query(
            self,
            query_id: Optional[str] = None,
            query: Optional[Iterable] = None,
            k: Optional[int] = None,
            with_scores: bool = False,
    ) -> Union[List[Any], List[Tuple[Any, float]]]:
        """
        Search for the nearest neighbours of the given query.

        Parameters
        ----------
        query_id : Optional[str]
            The the id of the query_engine.
            If defined then *query* is ignored.
            If None then it is assumed that the item has not been indexed and *query_hash* must be defined.
        query: Optional[Iterable]
            Since this is a set-based index, the *query* has to be an iterable.
            If None then it is assumed that the item has been indexed and *query_id* must be defined.
            If *query_id* is defined then this is ignored.
        k : int
            The number of neighbours to return.
        with_scores : bool
            Whether or not to return the estimated similarity scores associated with each result.
        Returns
        -------
        Union[List[Any], List[Tuple[Any, float]]]:
            A list of the nearest neighbours ids with or without associated similarity scores,
             depending on the values of *with_scores*.
        """

        query_hash = None
        if query_id is None:
            query_hash = self._hash_generator.hash(query, hashvalues=None)
            hash_chunks = self._get_lsh_keys(query_hash)
        else:
            hash_chunks = self._keys.get(query_id, None)
            if hash_chunks is None:
                raise ValueError(
                    "query_id must be an existing identifier in the index. Item with id {} not found!".format(
                        query_id
                    )
                )

        neighbours = [
            n
            for hash_entry, hash_table in zip(hash_chunks, self._hashTables)
            for n in hash_table.get(hash_entry, [])
        ]

        neighbour_counter = Counter(neighbours)
        neighbours = [w for w, _ in neighbour_counter.most_common(k) if w != query_id]
        if with_scores:
            similarity_scores = [
                self.get_similarity_score(query_hash, n)
                if query_id is None
                else self.get_similarity_score(query_id, n)
                for n in neighbours
            ]
            return list(zip(neighbours, similarity_scores))
        return neighbours
