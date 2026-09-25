"""w2vtools: the helpers the Lab 05 notebooks share.

    load_glove()           the GloVe vectors in data/glove, as unit vectors
    nearest(...)           the k nearest words (or products) to a vector
    kittiwake_sentences()  Kittiwake's tickets and reviews as lists of words
    train_skipgram(...)    skip-gram with negative sampling, in PyTorch: the same
                           algorithm 05_02 builds line by line, packaged so 05_03
                           and Part 3 can reuse it on products
    load_baskets()         Online Retail invoices as baskets of stock codes,
                           split in time: before 1 November 2011 to learn from,
                           from then on to test on
    cooccurrence(...)      for every product, its partners in the same invoices,
                           most frequent first (ties broken by stock code)
    product_vectors(...)   05_03's product vectors: loaded from the copy trained
                           in the background when your session started, or
                           trained now if that copy is missing
    part3_queries()        the 500 products Part 3 asks you to recommend for

Nothing here downloads anything; everything reads files under data/.
"""
import collections
import csv
import functools
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CUTOFF = "2011-11-01"


# ----------------------------------------------------------------- GloVe ---

def load_glove():
    """Return (words, index, vectors): 40,000 words, word -> row, unit vectors."""
    words = open(os.path.join(DATA, "glove", "vocab.txt")).read().split("\n")[:-1]
    vectors = np.load(os.path.join(DATA, "glove", "glove.6B.50d.top40k.npy")).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return words, {w: i for i, w in enumerate(words)}, vectors


def nearest(vectors, names, vector, k=5, exclude=()):
    """The k rows of `vectors` most similar to `vector`, as (name, cosine) pairs.

    `vectors` must already be unit length, so a dot product is a cosine."""
    vector = vector / np.linalg.norm(vector)
    scores = vectors @ vector
    out = []
    for i in np.argsort(-scores):
        if names[i] in exclude:
            continue
        out.append((names[i], round(float(scores[i]), 3)))
        if len(out) == k:
            break
    return out


# ------------------------------------------------------------- Kittiwake ---

def kittiwake_sentences():
    """Every Kittiwake ticket and review, lowercased, as a list of word lists."""
    texts = []
    for name in ("kittiwake_tickets.csv", "kittiwake_tickets_unlabeled.csv", "kittiwake_reviews.csv"):
        with open(os.path.join(DATA, name)) as f:
            texts += [r["text"] for r in csv.DictReader(f)]
    return [re.findall(r"[a-z0-9]+", t.lower()) for t in texts]


# ------------------------------------------------------------- skip-gram ---

def skipgram_pairs(sequences, index, window):
    """Every (centre, context) pair of ids within `window` places of each other,
    in the order 05_02's make_pairs builds them."""
    pairs = []
    for seq in sequences:
        ids = [index[t] for t in seq if t in index]
        for i, centre in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    pairs.append((centre, ids[j]))
    return pairs


def train_skipgram(sequences, dim=32, window=3, negatives=5, epochs=5, min_count=3,
                   batch=1024, lr=0.01, seed=0, verbose=True, threads=None):
    """Skip-gram with negative sampling. Returns (vocab, index, unit_vectors, losses).

    The same model 05_02 writes out step by step: two embedding tables, a dot
    product as the score, and a loss that pulls each real (centre, context) pair
    together and pushes `negatives` random pairs apart."""
    import torch
    # A session has 4 CPUs; PyTorch may count the whole machine and start more
    # threads than it can use, which only slows it down.
    torch.set_num_threads(threads or min(4, os.cpu_count() or 1))
    torch.manual_seed(seed)
    counts = collections.Counter(t for s in sequences for t in s)
    vocab = [w for w, c in counts.most_common() if c >= min_count]
    index = {w: i for i, w in enumerate(vocab)}
    pairs = torch.tensor(skipgram_pairs(sequences, index, window))
    noise = torch.tensor([counts[w] for w in vocab], dtype=torch.float) ** 0.75
    centre_emb = torch.nn.Embedding(len(vocab), dim)
    context_emb = torch.nn.Embedding(len(vocab), dim)
    torch.nn.init.uniform_(centre_emb.weight, -0.5 / dim, 0.5 / dim)
    torch.nn.init.zeros_(context_emb.weight)
    opt = torch.optim.Adam(list(centre_emb.parameters()) + list(context_emb.parameters()), lr=lr)
    losses = []
    for epoch in range(epochs):
        order = torch.randperm(len(pairs))
        total = 0.0
        for b in range(0, len(pairs), batch):
            p = pairs[order[b:b + batch]]
            c, o = p[:, 0], p[:, 1]
            neg = torch.multinomial(noise, len(c) * negatives, replacement=True).view(len(c), negatives)
            vc = centre_emb(c)
            pos_score = (vc * context_emb(o)).sum(1)
            neg_score = torch.bmm(context_emb(neg), vc.unsqueeze(2)).squeeze(2)
            loss = -(torch.nn.functional.logsigmoid(pos_score).mean()
                     + torch.nn.functional.logsigmoid(-neg_score).sum(1).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(c)
        losses.append(round(total / len(pairs), 3))
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}  loss {losses[-1]}")
    vectors = centre_emb.weight.detach().numpy().copy()
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vocab, index, vectors, losses


# --------------------------------------------------------- Online Retail ---

@functools.lru_cache(maxsize=1)
def load_baskets():
    """Return (train_baskets, test_baskets, descriptions).

    A basket is one invoice's distinct stock codes, in the order they were
    scanned. Cancelled invoices (numbers starting C), returns (negative
    quantities) and rows with no customer are removed. Invoices before
    CUTOFF train; the rest test, so the test is always the future. The result
    is cached for the life of the kernel, so a second call is free; do not
    change the lists it returns."""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA, "online_retail", "online_retail.csv.gz"),
                     dtype={"InvoiceNo": str, "StockCode": str})
    df = df.dropna(subset=["CustomerID", "Description"])
    df = df[(df.Quantity > 0) & (~df.InvoiceNo.str.startswith("C"))]
    df["InvoiceDate"] = pd.to_datetime(df.InvoiceDate)
    descriptions = (df.drop_duplicates("StockCode", keep="last")
                      .set_index("StockCode").Description.str.strip().to_dict())
    cut = pd.Timestamp(CUTOFF)
    # One row per product per invoice, in scanning order; then each invoice's
    # codes as a list.
    df = df.drop_duplicates(["InvoiceNo", "StockCode"])

    def baskets(part):
        return part.groupby("InvoiceNo", sort=True).StockCode.agg(list).tolist()

    return baskets(df[df.InvoiceDate < cut]), baskets(df[df.InvoiceDate >= cut]), descriptions


_COOC = {}


def cooccurrence(baskets, top=50):
    """For every product, its `top` partners by number of shared invoices.

    The same count as this loop, which is easy to read and slow in Python:

        for b in baskets:
            for a in b:
                for c in b:
                    if a != c:
                        together[a][c] += 1

    Here it is one sparse matrix product: X is invoices by products with a 1
    where the invoice holds the product, and X.T @ X counts, for every pair of
    products, the invoices they share. Ties are broken by stock code, so every
    run gives the same list."""
    key = (id(baskets), len(baskets), top)
    if key in _COOC:
        return _COOC[key]
    import scipy.sparse as sp
    codes = sorted({c for b in baskets for c in b})
    ix = {c: i for i, c in enumerate(codes)}
    rows = np.repeat(np.arange(len(baskets)), [len(b) for b in baskets])
    cols = np.array([ix[c] for b in baskets for c in b])
    X = sp.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(baskets), len(codes)))
    C = (X.T @ X).tocsr()
    out = {}
    for i, c in enumerate(codes):
        row = C.getrow(i)
        keep = row.indices != i
        idx, cnt = row.indices[keep], row.data[keep]
        order = np.lexsort((idx, -cnt))[:top]
        out[c] = [codes[j] for j in idx[order]]
    _COOC[key] = out
    return out


PRODUCT_SETTINGS = dict(dim=48, window=3, epochs=3, min_count=5, batch=16384, lr=0.03, seed=0)
PRODUCT_CACHE = os.path.join(HERE, "out", "product_vectors.npz")


def product_vectors(train_baskets, verbose=True):
    """Return (products, index, vectors) for 05_03, trained with PRODUCT_SETTINGS.

    Training takes a minute or two of a session's CPU, so setup.d starts it in
    the background when the session starts, and saves the result to
    out/product_vectors.npz (written to a temporary name, then renamed, so a
    half-written file is never read). If that file is here, this loads it in
    under a second; if not, it trains now and saves it."""
    if os.path.exists(PRODUCT_CACHE):
        z = np.load(PRODUCT_CACHE)
        if verbose:
            print("loaded out/product_vectors.npz, trained in the background when your session started")
        products = list(z["products"])
        return products, {p: i for i, p in enumerate(products)}, z["vectors"]
    if verbose:
        print("out/product_vectors.npz is not ready yet, so training now; this takes a minute or two")
    products, index, vectors, _ = train_skipgram(train_baskets, verbose=verbose, **PRODUCT_SETTINGS)
    os.makedirs(os.path.dirname(PRODUCT_CACHE), exist_ok=True)
    tmp = PRODUCT_CACHE + ".tmp.npz"
    np.savez(tmp, products=np.array(products), vectors=vectors)
    os.replace(tmp, PRODUCT_CACHE)
    return products, index, vectors


def part3_queries(test_baskets, known, n=500, seed=7):
    """The Part 3 queries: `n` test invoices with at least two known products,
    and one product from each. Returns a list of (query_id, stock_code).

    `known` is the set of products seen in training; a query about a product
    nobody bought before November cannot be answered by any method here."""
    rng = np.random.default_rng(seed)
    eligible = [b for b in test_baskets if len([c for c in b if c in known]) >= 2]
    picks = rng.choice(len(eligible), size=n, replace=False)
    out = []
    for q, i in enumerate(sorted(picks)):
        b = [c for c in eligible[i] if c in known]
        out.append((f"Q{q:03d}", b[rng.integers(len(b))]))
    return out


def part3_truth(test_baskets, known, n=500, seed=7):
    """For each Part 3 query, the other known products in the same invoice."""
    rng = np.random.default_rng(seed)
    eligible = [b for b in test_baskets if len([c for c in b if c in known]) >= 2]
    picks = rng.choice(len(eligible), size=n, replace=False)
    truth = {}
    for q, i in enumerate(sorted(picks)):
        b = [c for c in eligible[i] if c in known]
        product = b[rng.integers(len(b))]
        truth[f"Q{q:03d}"] = set(b) - {product}
    return truth
