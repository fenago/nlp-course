"""spamtools: the pieces of Lab 03 that a saved model has to be able to find again.

A scikit-learn pipeline saved with joblib stores its functions by NAME, not by
value: "call spamtools.clean". A function typed into a notebook cell lives in
a module called __main__ that only exists while that notebook's kernel runs,
so a pipeline built on it cannot be loaded anywhere else, including by the
checkpoint. That is why clean() lives here, in a file, and the notebooks
import it.

    load_sms()      the SMS Spam Collection as a DataFrame, read correctly
    split(df)       the one train/test split every notebook and the checkpoint use
    clean(text)     the book's cleaning: letters only, lowercase, stop words out, Porter stems
    embed(texts)    sentence embeddings from the small embedding model in the image,
                    cached under out/.embeddings, one file per text, warmed at session start
    Embed()         the same, as a scikit-learn step, so a Pipeline can embed
    describe_classes(descriptions)   zero-shot: route by the nearest label description
"""
import csv
import os
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

HERE = os.path.dirname(os.path.abspath(__file__))
SMS = os.path.join(HERE, "data", "sms_spam", "SMSSpamCollection")

_STOP = set(stopwords.words("english"))
_PS = PorterStemmer()


def load_sms(path=SMS):
    """One message per line, 'ham' or 'spam', a tab, then the text.

    quoting=csv.QUOTE_NONE matters: some messages start with a double quote
    and never close it, and pandas' default reading treats everything up to the
    next quote, newlines included, as one field.
    """
    return pd.read_csv(path, sep="\t", names=["label", "message"], quoting=csv.QUOTE_NONE)


def split(df):
    """70 percent to train, 30 to test, with the same share of spam in each."""
    from sklearn.model_selection import train_test_split
    return train_test_split(df["message"], df["label"], test_size=0.3,
                            random_state=42, stratify=df["label"])


def clean(text):
    """The cleaning step from the book's Chapter 3 notebook."""
    words = re.sub("[^a-zA-Z]", " ", text).lower().split()
    return " ".join(_PS.stem(w) for w in words if w not in _STOP)


# ------------------------------------------------------------ embeddings ---
# The model is in the image (HF_HOME=/opt/nlplab/hf, offline), so nothing
# downloads. It is loaded once per Python process and never pickled: a saved
# Pipeline stores the class name Embed, and the model is loaded again on use.
MODEL = "ibm-granite/granite-embedding-small-english-r2"
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MODEL)
    return _MODEL


def _cache_path(text):
    import hashlib
    key = hashlib.sha1(f"{MODEL}\x1f{text}".encode()).hexdigest()
    return os.path.join(HERE, "out", ".embeddings", key[:2], f"{key}.npy")


def embed(texts):
    """One 384-number vector per text, each of length 1.

    Embedding runs a neural network on every text, which takes minutes on a
    session's CPU, so every vector is cached under out/.embeddings/, one file
    per text. warm_embeddings.py fills the cache in the background when the
    session starts; whatever is not there yet is computed here and cached.
    If every text is cached, the model is not even loaded.
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
            # Written to a temporary name and renamed, so a reader never sees
            # half a file even while the background warm-up is running.
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


def describe_classes(descriptions, texts):
    """Zero-shot routing: each text goes to the label whose description is nearest.

    descriptions maps a label to one sentence describing it. No labelled
    examples are used at all; the labels are chosen by meaning alone.
    """
    import numpy as np
    labels = list(descriptions)
    L = embed([descriptions[k] for k in labels])
    E = embed(texts)
    return np.array(labels)[(E @ L.T).argmax(axis=1)]
