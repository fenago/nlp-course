"""attnlab: the transformer pieces and the headline data of Lab 10, in one importable file.

Why a file and not notebook cells: a saved PyTorch model is a set of numbers
(its state_dict). To use it again, whoever loads it (you tomorrow, or the
checkpoint today) has to rebuild the same network first, which means importing
the class that defines it. A class defined in a notebook cell exists only in
that notebook's kernel. A class defined here can be imported by anything.

    sinusoidal_positions(T, d)  the original transformer's position signal
    TransformerBlock            attention and a feed-forward layer, each with a
                                residual connection and layer normalisation
    ReverseTransformer          a tiny encoder that learns to reverse digits
    reverse_batch(n)            random digit sequences and their reverses
    load_split()                the news headlines, as tensors
    HeadlineTransformer         a small transformer classifier for headlines
    BagClassifier               the average of the word embeddings, no order
    train, evaluate             mini-batch training; accuracy and macro-F1
    save_model, load_model      save a model with what it needs to be rebuilt

The headlines are the UCI News Aggregator dataset (Gasparetti, 2016), already in
the image: https://archive.ics.uci.edu/dataset/359/news+aggregator (CC BY 4.0).
Nothing downloads.

You may add your own model class to this file for Part 3. save_model records
the class by name, so load_model can find it here.
"""
import collections
import csv
import gzip
import math
import os
import random
import re

import torch
import torch.nn as nn

DATA = os.path.join(os.environ.get("NLPLAB_DATA", "/opt/nlplab/data"), "news_headlines", "headlines.csv.gz")
CATEGORIES = ["b", "e", "m", "t"]
NAMES = ["business", "entertainment", "health", "science and technology"]
MAX_LEN = 20          # the longest headline in the data is 20 words
VOCAB_SIZE = 20000    # the 20,000 most frequent training words
TEST_SIZE = 10000
PAD, UNK = 0, 1

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "out", ".headlines.pt")


# ------------------------------------------------------------ positions ---

def sinusoidal_positions(T, d):
    """The position signal of Vaswani et al. (2017): a (T, d) tensor.

    Column 2i is sin(pos / 10000^(2i/d)), column 2i+1 is cos of the same angle,
    so every position gets a unique pattern of waves of many wavelengths.
    """
    pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d, 2, dtype=torch.float32)
    angle = pos / (10000 ** (i / d))
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


# ---------------------------------------------------------------- block ---

class TransformerBlock(nn.Module):
    """One transformer block, in the pre-norm arrangement modern models use.

        x = x + attention(norm(x))       every position looks at every other
        x = x + feedforward(norm(x))     then each position is processed alone

    The attention weights of the last call are kept in self.last_attention,
    shape (batch, heads, T, T), so you can look at what each head attended to.
    """

    def __init__(self, d=64, heads=4, ff=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)
        self.last_attention = None

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                         need_weights=True, average_attn_weights=False)
        self.last_attention = w.detach()
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.norm2(x)))


# ------------------------------------------------------ the reversal task ---

DIGITS, SEQ_LEN = 10, 8


def reverse_batch(n, T=SEQ_LEN, seed=None):
    """n random sequences of T digits, and the same sequences reversed."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.randint(0, DIGITS, (n, T), generator=g)
    return x, x.flip(1)


class ReverseTransformer(nn.Module):
    """Embed each digit, optionally add its position, two blocks, a digit out."""

    def __init__(self, d=64, heads=4, layers=2, positions=True, T=SEQ_LEN):
        super().__init__()
        self.emb = nn.Embedding(DIGITS, d)
        self.positions = positions
        self.register_buffer("pe", sinusoidal_positions(T, d))
        self.blocks = nn.ModuleList(TransformerBlock(d, heads, 2 * d, dropout=0.0) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, DIGITS)

    def forward(self, x):
        h = self.emb(x)
        if self.positions:
            h = h + self.pe[: x.size(1)]
        for b in self.blocks:
            h = b(h)
        return self.out(self.norm(h))


def train_reverse(model, steps=600, batch_size=128, lr=3e-3, seed=0):
    """Train on fresh random sequences; returns accuracy per digit on 2,000 new ones."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        x, y = reverse_batch(batch_size)
        loss = nn.functional.cross_entropy(model(x).reshape(-1, DIGITS), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    x, y = reverse_batch(2000, seed=12345)
    with torch.no_grad():
        return round(float((model(x).argmax(-1) == y).float().mean()), 4)


# ------------------------------------------------------------ headlines ---

def tokens(text):
    """Lowercase runs of letters: the tokenizer of Lab 01's regex line."""
    return re.findall(r"[a-z]+", text.lower())


def read_headlines():
    """(title, category) pairs, exact duplicate titles removed, in a fixed shuffle."""
    seen, rows = set(), []
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = r["TITLE"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append((r["TITLE"], r["CATEGORY"]))
    random.Random(10).shuffle(rows)
    return rows


def _build():
    rows = read_headlines()
    train, test = rows[:-TEST_SIZE], rows[-TEST_SIZE:]
    counts = collections.Counter(w for t, _ in train for w in tokens(t))
    vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(VOCAB_SIZE))}

    def encode(t):
        ids = [vocab.get(w, UNK) for w in tokens(t)][:MAX_LEN]
        return ids + [PAD] * (MAX_LEN - len(ids))

    return {
        "vocab": vocab,
        "X_train": torch.tensor([encode(t) for t, _ in train]),
        "y_train": torch.tensor([CATEGORIES.index(c) for _, c in train]),
        "X_test": torch.tensor([encode(t) for t, _ in test]),
        "y_test": torch.tensor([CATEGORIES.index(c) for _, c in test]),
        "raw_test": [t for t, _ in test[:20]],
    }


def load_split():
    """The headline data, built once and cached in out/ (a few seconds)."""
    if os.path.exists(CACHE):
        try:
            return torch.load(CACHE, weights_only=False)
        except Exception:
            pass
    data = _build()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = CACHE + ".tmp%d" % os.getpid()
    torch.save(data, tmp)
    os.replace(tmp, CACHE)      # atomic: a half-written cache is never read
    return data


def encode_text(text, vocab):
    """One new headline as a (1, MAX_LEN) tensor, the way the data was encoded."""
    ids = [vocab.get(w, UNK) for w in tokens(text)][:MAX_LEN]
    return torch.tensor([ids + [PAD] * (MAX_LEN - len(ids))])


class HeadlineTransformer(nn.Module):
    """Word embeddings plus positions, a stack of blocks, the mean of the words, a class.

    Padding is masked twice: attention never looks at a padding position, and
    the average at the end is taken over the real words only.
    """

    def __init__(self, vocab_size=VOCAB_SIZE + 2, d=64, heads=4, layers=1, ff=128,
                 dropout=0.1, classes=len(CATEGORIES)):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.register_buffer("pe", sinusoidal_positions(MAX_LEN, d))
        self.blocks = nn.ModuleList(TransformerBlock(d, heads, ff, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, classes)

    def forward(self, x):
        pad = x == PAD
        pad_attn = pad.clone()
        pad_attn[:, 0] = False          # an empty headline still has one position to attend to
        h = self.emb(x) + self.pe[: x.size(1)]
        for b in self.blocks:
            h = b(h, key_padding_mask=pad_attn)
        keep = (~pad).float().unsqueeze(-1)
        pooled = (self.norm(h) * keep).sum(1) / keep.sum(1).clamp(min=1)
        return self.out(pooled)


class BagClassifier(nn.Module):
    """Average the word embeddings, ignoring order, then one linear layer."""

    def __init__(self, vocab_size=VOCAB_SIZE + 2, d=64, classes=len(CATEGORIES)):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, d, mode="mean", padding_idx=PAD)
        self.out = nn.Linear(d, classes)

    def forward(self, x):
        return self.out(self.emb(x))


def train(model, X, y, epochs=2, optimizer=None, batch_size=128, seed=0, clip=1.0):
    """Mini-batch training with cross-entropy; returns the loss of each epoch."""
    torch.manual_seed(seed)
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=2e-3)
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed + epoch))
        total = 0.0
        for i in range(0, len(X), batch_size):
            batch = order[i:i + batch_size]
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(X[batch]), y[batch])
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        history.append(total / len(X))
    return history


def evaluate(model, X, y):
    """Accuracy and macro-F1 (the mean of each category's F1)."""
    from sklearn.metrics import f1_score
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(X[i:i + 1000]).argmax(1) for i in range(0, len(X), 1000)])
    return {"accuracy": round(float((pred == y).float().mean()), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 4)}


def save_model(model, path, **kwargs):
    """Save the weights with the class name and the arguments that rebuild it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"class": type(model).__name__, "kwargs": kwargs,
                "state_dict": model.state_dict()}, path)


def load_model(path):
    """Rebuild a model saved with save_model: the class must be in this file."""
    saved = torch.load(path, weights_only=False)
    cls = globals().get(saved["class"])
    if cls is None:
        raise ValueError(f"{saved['class']} is not defined in attnlab.py; define the class "
                         "here, not in a notebook cell, so it can be rebuilt")
    model = cls(**saved.get("kwargs", {}))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model
