import os
from typing import Iterable, Optional, Set
from d3l.utils.functions import remove_blanked_token
from d3l.utils.functions import token_stop_word as tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from d3l.utils.constants import STOPWORDS
from d3l.utils.functions import shingles


class SBERTTransformer:
    def __init__(
            self,
            token_pattern: str = r"(?u)\b\w\w+\b",
            max_df: float = 0.5,
            stop_words: Iterable[str] = STOPWORDS,
            model_name: str = "all-MiniLM-L6-v2",  # download pretrained model
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
        self._stop_words = stop_words
        self._model_name = model_name
        if cache_dir is not None:
            folder = os.path.exists(cache_dir)
            if not folder:
                os.makedirs(cache_dir)
        self._cache_dir = (
            cache_dir if cache_dir  is not None and os.path.isdir(cache_dir) else None
        )

        self._embedding_model = self.get_embedding_model(
            overwrite=False,
        )
        self._embedding_dimension = self.get_embedding_dimension()

    def __getstate__(self):
        d = self.__dict__
        self_dict = {k: d[k] for k in d if k != "_embedding_model"}
        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state
        self._embedding_model = self.get_embedding_model(overwrite=False)

    @property
    def cache_dir(self) -> Optional[str]:
        return self._cache_dir

    def _download_SBERT(self,
                        model_name: str = "all-MiniLM-L6-v2"):
        """
        Download pre-trained GloVe vectors from hugging face

        Parameters
        ----------
         model_name : str
            The embedding model name to download from hugging face website.
            It does not have to include to *.zip* extension.
            By default, the most popular model "all-MiniLM-L6-v2" will be used.
        Returns
        -------
        """
        url = model_name
        print("Downloading %s" % url)
        model = SentenceTransformer(url)

        write_file_name = (
            os.path.join(self._cache_dir, model_name)
            if self._cache_dir is not None
            else os.getcwd()+"/"+url
        )
        folder = os.path.exists(write_file_name)
        if not folder:
            os.makedirs(write_file_name)
        model.save(write_file_name)

    def _download_model(self,
                        model_name: str = "all-MiniLM-L6-v2",
                        if_exists: str = "strict"):
        """
        Download the pre-trained model file.
        Parameters
        ----------
        model_name : str
            The embedding model name to download from hugging face website.
            By default, the *"all-MiniLM-L6-v2"* model will be used.
        if_exists : str
            Supported values:
                - *ignore*: The model will not be downloaded
                - *strict*: This is the default. The model will be downloaded only if it does not exist at local path
                - *overwrite*: The model will be downloaded even if it already exists at the local path.

        Returns
        -------
        """

        base_file_name = "sentence-transformers/%s" % model_name
        file_name = (
            os.path.join(self._cache_dir, base_file_name)
            if self._cache_dir is not None
            else base_file_name)

        if os.path.isfile(file_name):
            if if_exists == "ignore":
                return file_name
            elif if_exists == "strict":
                print("File exists. Use --*overwrite* to download anyway.")
                return file_name
            elif if_exists == "overwrite":
                pass

        absolute_gz_file_name = (
            os.path.join(self._cache_dir, base_file_name)
            if self._cache_dir is not None
            else base_file_name
        )

        if not os.path.isfile(absolute_gz_file_name):
            self._download_SBERT(base_file_name)

        """Cleanup"""
        if os.path.isfile(absolute_gz_file_name):
            os.remove(absolute_gz_file_name)

        return file_name

    def get_embedding_model(
            self,
            model_name: str = "all-MiniLM-L6-v2",
            overwrite: bool = False
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
        if_exists = "strict" if not overwrite else "overwrite"

        model_file = self._download_model(model_name=model_name, if_exists=if_exists)
        print("Loading embeddings. This may take a few minutes ...")
        print(model_file)
        model = SentenceTransformer(model_file)
        return model

    def get_embedding_dimension(self) -> int:
        """
        Retrieve the embedding dimensions of the underlying model.
        Returns
        -------
        int
            The dimensions of each embedding
        """

        return self._embedding_model.get_sentence_embedding_dimension()

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
        vector = self._embedding_model.encode(
            ' '.join(tokenize(word))
        )
        return vector

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

        if len(input_values) < 1:
            return set()
        """
        try:
            vectorizer = TfidfVectorizer(
                decode_error="ignore",
                strip_accents="unicode",
                lowercase=True,
                analyzer="word",
                stop_words=self._stop_words,
                token_pattern=self._token_pattern,
                max_df=self._max_df,
                use_idf=True,
            )
            vectorizer.fit_transform(input_values)
        except ValueError:
            return set()

        weight_map = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
        tokenset = set()
        tokenizer = vectorizer.build_tokenizer()
        for value in input_values:
            value = value.lower().replace("\n", " ").strip()
            for shingle in shingles(value):
                tokens = [t for t in tokenizer(shingle)]

                if len(tokens) < 1:
                    continue

                token_weights = [weight_map.get(t, 0.0) for t in tokens]
                min_tok_id = np.argmin(token_weights)
                tokenset.add(tokens[min_tok_id])
        print(tokenset)
        """
        tokenset = remove_blanked_token(input_values)
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

        embeddings = [self.get_vector(token) for token in self.get_tokens(input_values)]
        # print(embeddings)
        if len(embeddings) == 0:
            return np.empty(0)
        return np.mean(np.array(embeddings), axis=0)
