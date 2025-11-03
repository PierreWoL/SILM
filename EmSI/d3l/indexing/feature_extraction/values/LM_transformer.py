import os
from typing import Iterable, Optional, Set

import torch
from transformers import AutoModel, AutoTokenizer

from d3l.utils.functions import remove_blanked_token  # remove_blank,
from collections import Counter
from d3l.utils.functions import token_stop_word as tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from d3l.utils.constants import STOPWORDS
from d3l.utils.functions import shingles, is_empty, is_number

lm_mp = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'sbert': 'sentence-transformers/all-mpnet-base-v2'}


class LM_Transformer:
    def __init__(
            self,
            token_pattern: str = r"(?u)\b\w\w+\b",
            max_df: float = 1,
            min_df: float = 0,
            stop_words: Iterable[str] = STOPWORDS,
            model_name: str = "bert",  # download pretrained model
            cache_dir: Optional[str] = None,
    ):
        """
        Instantiate a new embedding-based transformer
        Parameters
        ----------
        token_pattern : str
            The regex used to identify tokens.
            The default value is scikit-learn's Tfidf Vectorizer default.
        max_df : float
            Percentage of values the token can appear in before it is ignored.
        stop_words : Iterable[str]
            A collection of stopwords to ignore that defaults to NLTK's English stopwords.
        model_name : str
            The embedding model name to download from hugging face model hub website.
            It does not have to include to *.zip* extension.
            By default, the *Common Crawl 42B* model will be used.
        cache_dir : Optional[str]
            An exising directory path where the model will be stored.
            If not given, the current working directory will be used.
        """

        self._token_pattern = token_pattern
        self._max_df = max_df
        self._min_df = min_df
        self._stop_words = stop_words
        self._tokenizer = self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[model_name],
                                                                        selectable_pos=1)
        self._model_name = model_name
        if cache_dir is not None:
            folder = os.path.exists(cache_dir)
            if not folder:
                os.makedirs(cache_dir)
        self._cache_dir = (
            cache_dir if cache_dir is not None and os.path.isdir(cache_dir) else None
        )

        self._embedding_model = self.get_embedding_model(model_name
        )

        self._embedding_dimension = self.get_embedding_dimension()

    def __getstate__(self):
        d = self.__dict__
        self_dict = {k: d[k] for k in d if k != "bert"}
        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state
        self._embedding_model = self.get_embedding_model(self._model_name)

    @property
    def cache_dir(self) -> Optional[str]:
        return self._cache_dir

    def get_embedding_model(
            self,
            model_name: str = "bert",
    ):
        """
        Download, if not exists, and load the pretrained GloVe embedding model in the working directory.
        Parameters
        ----------
        model_name : str
            The embedding model name to download from Stanford's website.
            It does not have to include to *.zip* extension.
            By default, the *Common Crawl 42B* model will be used.
        overwrite : bool
            If True overwrites the model if exists.
        Returns
        -------
        """

        print("Loading embeddings. This may take a few minutes ...")
        url = lm_mp[model_name]
        print("Downloading %s" % url)
        model = AutoModel.from_pretrained(url) if not model_name == "sbert" else SentenceTransformer(url)
        return model

    def get_embedding_dimension(self) -> int:
        """
        Retrieve the embedding dimensions of the underlying model.
        Returns
        -------
        int
            The dimensions of each embedding
        """
        dimension = self._embedding_model.config.hidden_size if not self._model_name == "sbert" \
            else self._embedding_model._first_module().auto_model.config.hidden_size
        return dimension

    def get_vector(self, word: str) -> np.ndarray:
        """
        Retrieve the embedding of the given word.
        If the word is out of vocabulary a zero vector is returned.
        Parameters
        ----------
        word : str
            The word to retrieve the vector for.

        Returns
        -------
        np.ndarray
            A vector of float numbers.
        """
        if self._model_name =='sbert':
            vector = self._embedding_model.encode(
            ' '.join(tokenize(word))
            )

        else:
            #print(' '.join(tokenize(word)))
            tokens = self.tokenizer.encode_plus(' '.join(tokenize(word)), add_special_tokens=True,
                                                truncation=True, return_tensors="pt")
            # Perform the encoding using the model
            #print(tokens)
            with torch.no_grad():
                outputs = self._embedding_model(**tokens)
            # Extract the last hidden state (embedding) from the outputs
            vector = outputs.last_hidden_state.mean(dim=1)[0]

        return np.array(vector)


    # todo: modify here
    def get_tokens(self, input_values: Iterable[str]) -> Set[str]:
        """
        Extract the most representative tokens of each value and return the token set.
        Here, the most representative tokens are the ones with the lowest TF/IDF scores -
        tokens that describe what the values are about.
        Parameters
        ----------
        input_values : Iterable[str]
            The collection of values to extract tokens from.

        Returns
        -------
        Set[str]
            A set of representative tokens
        """

        if input_values is None:
            return set()
        elif len(input_values) < 1:
            return set()

        try:
            vectorizer = TfidfVectorizer(
                decode_error="ignore",
                strip_accents="unicode",
                lowercase=True,
                analyzer="word",
                stop_words=self._stop_words,
                token_pattern=self._token_pattern,
                max_df=self._max_df,
                min_df=self._min_df,
                use_idf=True,
            )
            term_counts = Counter(input_values)
            # preprocessed_data = ['{} ({})'.format(term, count) for term, count in term_counts.items()]
            preprocessed_data = ['{}'.format(term) for term, count in term_counts.items()]
            vectorizer.fit_transform(preprocessed_data)
        except ValueError as e:
            #  print(input_values, e)
            return set()
        weight_map = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
        tokenset = set()
        tokenizer = vectorizer.build_tokenizer()
        for value in input_values:
            if is_number(value) is True:
                continue
            if type(value) is bool:
                value = str(value)
            value = str(value).lower().replace("\n", " ").strip()
            for shingle in shingles(value):
                tokens = [t for t in tokenizer(shingle)]
                if len(tokens) < 1:
                    continue
                token_weights = [weight_map.get(t, 0.0) for t in tokens]
                min_tok_id = np.argmin(token_weights)
                tokenset.add(tokens[min_tok_id])

        term_counts = Counter(input_values)
        # preprocessed_data = ['{} ({})'.format(term, count) for term, count in term_counts.items()]
        tokenset = set(['{}'.format(term) for term, count in term_counts.items()])
        # print(tokenset)
        return tokenset

    def transform(self, input_values: Iterable[str]) -> np.ndarray:
        """
         Extract the embeddings of the most representative tokens of each value and return their **mean** embedding.
         Here, the most representative tokens are the ones with the lowest TF/IDF scores -
         tokens that describe what the values are about.
         Given that the underlying embedding model is a n-gram based one,
         the number of out-of-vocabulary tokens should be relatively small or zero.
         Parameters
         ----------
        input_values : Iterable[str]
             The collection of values to extract tokens from.

         Returns
         -------
         np.ndarray
             A Numpy vector representing the mean of all token embeddings.
        """
        embeddings = [self.get_vector(token) for token in self.get_tokens(remove_blanked_token(input_values))]
        if len(embeddings) == 0:
            return np.empty(0)
        return np.mean(np.array(embeddings), axis=0)
