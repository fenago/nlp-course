"""newswire: the Reuters data and the models of Lab 07, in one importable file.

Why a file and not notebook cells: a saved PyTorch model is a set of numbers
(its state_dict). To use it again, whoever loads it (you tomorrow, or the
checkpoint today) has to rebuild the same network first, which means importing
the class that defines it. A class defined in a notebook cell exists only in
that notebook's kernel. A class defined here can be imported by anything.

    load_split()               the six-topic Reuters data, as tensors
    RNNClassifier              nn.RNN over word embeddings, last hidden state
    RNNMeanClassifier          the same, averaging every hidden state instead
    BagClassifier              the average of the word embeddings, no order
    save_model(model, path)    save a model with what it needs to be rebuilt
    load_model(path)           rebuild and load it
    evaluate(model, X, y)      accuracy and macro-F1

The data is Reuters-21578 (the ApteMod split as packaged by NLTK), already in
the image: https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection
(CC BY 4.0). Nothing downloads.

You may add your own model class to this file for Part 3. save_model records
the class by name, so load_model can find it here.
"""
import collections
import os
import re

import torch
import torch.nn as nn

TOPICS = ["earn", "acq", "crude", "trade", "money-fx", "interest"]
MAX_LEN = 50          # the first 50 words of each newswire: headline and lead
VOCAB_SIZE = 10000    # the 10,000 most frequent training words
PAD, UNK = 0, 1

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "out", ".newswire.pt")


def tokens(text):
    """Lowercase runs of letters: the tokenizer of Lab 01's regex line."""
    return re.findall(r"[a-z]+", text.lower())


def _build():
    from nltk.corpus import reuters
    docs = [f for f in reuters.fileids()
            if len(reuters.categories(f)) == 1 and reuters.categories(f)[0] in TOPICS]
    train = [f for f in docs if f.startswith("training/")]
    test = [f for f in docs if f.startswith("test/")]
    text = {f: tokens(reuters.raw(f)) for f in docs}
    counts = collections.Counter(w for f in train for w in text[f])
    vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(VOCAB_SIZE))}

    def encode(f):
        ids = [vocab.get(w, UNK) for w in text[f]][:MAX_LEN]
        return ids + [PAD] * (MAX_LEN - len(ids))

    def label(f):
        return TOPICS.index(reuters.categories(f)[0])

    return {
        "vocab": vocab,
        "X_train": torch.tensor([encode(f) for f in train]),
        "y_train": torch.tensor([label(f) for f in train]),
        "X_test": torch.tensor([encode(f) for f in test]),
        "y_test": torch.tensor([label(f) for f in test]),
        "raw_test": [reuters.raw(f) for f in test[:20]],
    }


def load_split():
    """The six-topic data, built once and cached in out/ (a few seconds)."""
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
    """One new newswire as a (1, MAX_LEN) tensor, the way the data was encoded."""
    ids = [vocab.get(w, UNK) for w in tokens(text)][:MAX_LEN]
    return torch.tensor([ids + [PAD] * (MAX_LEN - len(ids))])


class RNNClassifier(nn.Module):
    """Embed each word, run nn.RNN left to right, classify the last real state."""

    def __init__(self, vocab_size=VOCAB_SIZE + 2, embed_dim=64, hidden=64, classes=len(TOPICS)):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.rnn = nn.RNN(embed_dim, hidden, batch_first=True)
        self.out = nn.Linear(hidden, classes)

    def forward(self, x):
        lengths = (x != PAD).sum(1).clamp(min=1)
        states, _ = self.rnn(self.emb(x))
        last = states[torch.arange(x.size(0)), lengths - 1]
        return self.out(last)


class RNNMeanClassifier(RNNClassifier):
    """The same network, but it averages the hidden state at every real word."""

    def forward(self, x):
        mask = (x != PAD).float().unsqueeze(-1)
        states, _ = self.rnn(self.emb(x))
        return self.out((states * mask).sum(1) / mask.sum(1).clamp(min=1))


class BagClassifier(nn.Module):
    """Average the word embeddings, ignoring order, then one linear layer."""

    def __init__(self, vocab_size=VOCAB_SIZE + 2, embed_dim=64, classes=len(TOPICS)):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean", padding_idx=PAD)
        self.out = nn.Linear(embed_dim, classes)

    def forward(self, x):
        return self.out(self.emb(x))


class ScalarIdRNN(nn.Module):
    """The book's Chapter 7 network: each word's id fed in as one number."""

    def __init__(self, hidden=50, classes=len(TOPICS)):
        super().__init__()
        self.rnn = nn.RNN(1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, classes)

    def forward(self, x):
        states, _ = self.rnn(x.float().unsqueeze(-1))
        return self.out(states[:, -1])


def train(model, X, y, epochs=5, optimizer=None, batch_size=64, seed=0, clip=5.0):
    """Mini-batch training with cross-entropy; returns the loss of each epoch."""
    torch.manual_seed(seed)
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=2e-3)
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
    """Accuracy and macro-F1 (the mean of each topic's F1, so small topics count)."""
    from sklearn.metrics import f1_score
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(1)
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
        raise ValueError(f"{saved['class']} is not defined in newswire.py; define the class "
                         "here, not in a notebook cell, so it can be rebuilt")
    model = cls(**saved.get("kwargs", {}))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model
