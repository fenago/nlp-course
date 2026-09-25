"""warm_embeddings: embed Kittiwake's 400 reviews once, and reuse the vectors.

A real search system embeds its documents once, when they are added, and at
search time embeds only the query. This lab does the same. At session start
workshop/setup.d/10-prepare.sh runs this file in the background:

    python3 warm_embeddings.py        # writes out/review_embeddings.npz

and the notebooks call review_vectors(), which loads that file when it holds
exactly the reviews you are searching and embeds them there and then (about
forty seconds) when it does not.

The file is written to a temporary name and renamed into place, so a notebook
never reads a half-written cache.
"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "ibm-granite/granite-embedding-small-english-r2"
CACHE = os.path.join(HERE, "out", "review_embeddings.npz")


def _encode(model, texts):
    return model.encode(list(texts), normalize_embeddings=True, batch_size=32)


def review_vectors(model, texts):
    """Return (vectors, how): cached vectors when they match texts, else fresh ones."""
    texts = list(texts)
    try:
        with np.load(CACHE, allow_pickle=False) as z:
            if str(z["model"]) == MODEL and list(z["texts"]) == texts:
                return z["vectors"], "loaded from out/review_embeddings.npz, computed when your session started"
    except (OSError, KeyError, ValueError):
        pass
    start = time.time()
    vectors = _encode(model, texts)
    return vectors, f"computed now, in {time.time() - start:.1f} s (the session-start copy was missing, unfinished, or for other texts)"


def main():
    with open(os.path.join(HERE, "data", "kittiwake_reviews.csv")) as f:
        texts = [r["text"] for r in csv.DictReader(f)]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    start = time.time()
    vectors = _encode(model, texts)
    tmp = CACHE + ".tmp.npz"
    np.savez(tmp, model=np.array(MODEL), texts=np.array(texts), vectors=vectors)
    os.replace(tmp, CACHE)
    print(f"embedded {len(texts)} reviews in {time.time() - start:.1f} s into {CACHE}")


if __name__ == "__main__":
    sys.exit(main())
