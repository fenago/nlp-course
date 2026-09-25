"""neurontools: the one helper Lab 06's Part 3 needs, sentence embeddings with a cache.

    from neurontools import embed
    X = embed(list_of_texts)       # one row of 384 numbers per text, each of length 1

The embedding model is in the image (HF_HOME=/opt/nlplab/hf, offline), so
nothing downloads. Embedding runs a neural network over every text, which takes
about two minutes of a session's CPU for 800 tickets, so every vector is cached
under out/.embeddings/, one file per text. warm_embeddings.py fills the cache in
the background when the session starts; whatever is not there yet is computed
here and cached. If every text is cached the model is not even loaded.

The same cache code as Lab 03's spamtools.embed(), so it behaves the same way.
"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
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


def embed(texts):
    """One 384-number vector per text, each of length 1, from the cache when possible."""
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


def cached_fraction(texts):
    """How many of these texts are already in the cache, as a fraction."""
    texts = [str(t) for t in texts]
    return sum(os.path.exists(_cache_path(t)) for t in texts) / max(len(texts), 1)
