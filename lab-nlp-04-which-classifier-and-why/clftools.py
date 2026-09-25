"""clftools: the data and the embeddings every Lab 04 notebook shares.

    load_sentences()   the UCI Sentiment Labelled Sentences as a DataFrame
    split(df)          the one train/test split every notebook and the checkpoint use
    load_reviews()     Kittiwake's 20 labelled reviews and the unlabelled rest
    embed(texts)       sentence embeddings from the small model in the image,
                       cached under out/.embeddings, one file per text, and
                       warmed at session start by warm_embeddings.py
    Embed()            the same, as a scikit-learn step, so a Pipeline can embed

It lives in a file rather than a notebook cell for two reasons: every notebook
uses the same split, so their numbers agree, and a saved Pipeline that embeds
must be able to find Embed again by name.
"""
import hashlib
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def load_sentences():
    """3,000 one-sentence reviews from Amazon, IMDB and Yelp; label 1 positive, 0 negative."""
    return pd.read_csv(os.path.join(DATA, "sentiment_sentences", "sentences.csv"))


def split(df):
    """70 percent to train, 30 to test, with the same share of each label in both."""
    from sklearn.model_selection import train_test_split
    return train_test_split(df["text"], df["label"], test_size=0.3,
                            random_state=42, stratify=df["label"])


def load_reviews():
    """(labelled, unlabelled): 20 Kittiwake reviews with a label, and the rest without."""
    return (pd.read_csv(os.path.join(DATA, "kittiwake_reviews_labelled.csv")),
            pd.read_csv(os.path.join(DATA, "kittiwake_reviews_unlabelled.csv")))


# ------------------------------------------------------------ embeddings ---
# The model is in the image (HF_HOME=/opt/nlplab/hf, offline), so nothing
# downloads. It is loaded once per Python process and never pickled.
MODEL = "ibm-granite/granite-embedding-small-english-r2"
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL)
    return _MODEL


def _cache_path(text):
    key = hashlib.sha1(f"{MODEL}\x1f{text}".encode()).hexdigest()
    return os.path.join(HERE, "out", ".embeddings", key[:2], f"{key}.npy")


def cached(texts):
    """How many of these texts already have a cached vector."""
    return sum(os.path.exists(_cache_path(str(t))) for t in texts)


def embed(texts):
    """One 384-number vector per text, each of length 1.

    Every vector is cached under out/.embeddings/, one file per text.
    warm_embeddings.py fills the cache in the background when the session
    starts; whatever is not there yet is computed here, and cached. If every
    text is cached, the model is not even loaded.
    """
    import numpy as np
    texts = [str(t) for t in texts]
    out = [None] * len(texts)
    missing = []
    for i, t in enumerate(texts):
        p = _cache_path(t)
        if os.path.exists(p):
            try:
                out[i] = np.load(p)
                continue
            except (OSError, ValueError):
                pass
        missing.append(i)
    if missing:
        todo = list(dict.fromkeys(texts[i] for i in missing))
        vecs = _model().encode(todo, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        got = dict(zip(todo, vecs))
        for t, v in got.items():
            p = _cache_path(t)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            # A temporary name, then a rename, so a reader never sees half a
            # file while the background warm-up is still writing.
            tmp = f"{p}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                np.save(f, v)
            os.replace(tmp, p)
        for i in missing:
            out[i] = got[texts[i]]
    return np.vstack(out) if out else np.zeros((0, 384), dtype="float32")


from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402


class Embed(BaseEstimator, TransformerMixin):
    """A Pipeline step that turns raw texts into sentence embeddings."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return embed(X)
