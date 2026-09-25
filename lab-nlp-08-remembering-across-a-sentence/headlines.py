"""headlines: the news headlines and the recurrent models of Lab 08, in one file.

Why a file and not notebook cells: a saved PyTorch model is a set of numbers
(its state_dict). To use it again, whoever loads it (you tomorrow, or the
checkpoint today) has to rebuild the same network first, which means importing
the class that defines it. A class defined in a notebook cell exists only in
that notebook's kernel. A class defined here can be imported by anything.

    load_split()                 the four-category headlines, as tensors
    SequenceClassifier           nn.RNN or nn.LSTM over word embeddings,
                                 one or both directions, packed or padded
    BagClassifier                the average of the word embeddings, no order
    first_word_task(n, length)   a made-up task whose answer is the first word
    open_forget_gate(model, b)   start every LSTM forget gate at sigmoid(b)
    position_gradients(model, X, y)  the gradient's size at every position
    train(model, X, y)           mini-batch AdamW with gradient clipping
    save_model / load_model      save a model with what rebuilds it, and back
    evaluate(model, X, y)        accuracy and macro-F1

The data is the UCI News Aggregator dataset (Gasparetti, 2016, CC BY 4.0),
15,000 headlines from each of four categories, already in the image:
https://archive.ics.uci.edu/dataset/359/news+aggregator . Nothing downloads.

You may add your own model class to this file for Part 3. save_model records
the class by name, so load_model can find it here.
"""
import collections
import csv
import gzip
import os
import re

import torch
import torch.nn as nn

DATA = os.path.join(os.environ.get("NLPLAB_DATA", "/opt/nlplab/data"), "news_headlines", "headlines.csv.gz")
CATEGORIES = ["b", "e", "m", "t"]
NAMES = {"b": "business", "e": "entertainment", "m": "health", "t": "science and technology"}
MAX_LEN = 20          # no headline in the set is longer than 19 words
VOCAB_SIZE = 16000    # about every word seen at least twice in training
PAD, UNK = 0, 1

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "out", ".headlines.pt")


def tokens(text):
    """Lowercase words and numbers, keeping an apostrophe ending like 's."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def read_rows():
    """(title, category) pairs, exact duplicate titles dropped, file order kept."""
    seen, rows = set(), []
    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = r["TITLE"].strip().lower()
            if key not in seen:
                seen.add(key)
                rows.append((r["TITLE"], r["CATEGORY"]))
    return rows


def _build():
    rows = read_rows()
    test = [r for i, r in enumerate(rows) if i % 5 == 0]      # every fifth headline is held out
    train = [r for i, r in enumerate(rows) if i % 5 != 0]
    counts = collections.Counter(w for t, _ in train for w in tokens(t))
    vocab = {w: i + 2 for i, (w, _) in enumerate(counts.most_common(VOCAB_SIZE))}

    def enc(rs):
        return torch.stack([encode_text(t, vocab)[0] for t, _ in rs])

    def lab(rs):
        return torch.tensor([CATEGORIES.index(c) for _, c in rs])

    return {"vocab": vocab, "X_train": enc(train), "y_train": lab(train),
            "X_test": enc(test), "y_test": lab(test), "raw_test": [t for t, _ in test[:20]]}


def load_split():
    """The four-category data, built once and cached in out/ (a few seconds)."""
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
    """One headline as a (1, MAX_LEN) tensor of ids, padded on the right."""
    ids = [vocab.get(w, UNK) for w in tokens(text)][:MAX_LEN]
    return torch.tensor([ids + [PAD] * (MAX_LEN - len(ids))])


def left_pad(X):
    """Move each row's padding to the front, the way the book's Keras code padded."""
    out = torch.zeros_like(X)
    for i, row in enumerate(X):
        ids = row[row != PAD]
        if len(ids):
            out[i, -len(ids):] = ids
    return out


class SequenceClassifier(nn.Module):
    """Embed each word, read the headline with a recurrent layer, classify its final state.

    cell           "rnn" (Lab 07's nn.RNN) or "lstm" (nn.LSTM)
    bidirectional  also read the headline right to left, and join the two final states
    pack           True: pack the batch, so the layer stops at each headline's last real
                   word. False: it reads the padding too, as the book's Keras code did.
    """

    def __init__(self, cell="lstm", bidirectional=False, pack=True, vocab_size=VOCAB_SIZE + 2,
                 embed_dim=64, hidden=64, layers=1, dropout=0.0, classes=len(CATEGORIES)):
        super().__init__()
        self.pack = pack
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        layer = {"rnn": nn.RNN, "lstm": nn.LSTM}[cell]
        self.rnn = layer(embed_dim, hidden, num_layers=layers, batch_first=True,
                         bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * (2 if bidirectional else 1), classes)

    def forward_embedded(self, e, lengths):
        """Everything after the embedding lookup, so a gradient can be asked of e."""
        if self.pack:
            e = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(e)
        h = h[0] if isinstance(h, tuple) else h          # an LSTM returns (h_n, c_n)
        if self.rnn.bidirectional:
            final = torch.cat([h[-2], h[-1]], dim=1)     # last layer, forward and backward
        else:
            final = h[-1]
        return self.out(self.drop(final))

    def forward(self, x):
        lengths = (x != PAD).sum(1).clamp(min=1)
        return self.forward_embedded(self.emb(x), lengths)


class BagClassifier(nn.Module):
    """Average the word embeddings, ignoring order, then one linear layer (Lab 07)."""

    def __init__(self, vocab_size=VOCAB_SIZE + 2, embed_dim=64, classes=len(CATEGORIES)):
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean", padding_idx=PAD)
        self.out = nn.Linear(embed_dim, classes)

    def forward(self, x):
        return self.out(self.emb(x))


TASK_VOCAB = 26       # ids for first_word_task: 0 and 1 unused, 2 to 5 the signal, 6 to 25 noise


def first_word_task(n, length, seed=0):
    """n made-up sequences whose label is decided by the first word alone.

    The first word is one of four signal words (ids 2 to 5, labels 0 to 3); every
    word after it is noise, drawn at random from twenty others. Nothing after the
    first word carries any information, so a model can only score above 0.25 by
    carrying the first word all the way to the end.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(6, TASK_VOCAB, (n, length), generator=g)
    X[:, 0] = torch.randint(2, 6, (n,), generator=g)
    return X, X[:, 0] - 2


def open_forget_gate(model, bias):
    """Set every LSTM forget gate's bias, so the gates start at sigmoid(bias).

    PyTorch stores each layer's four gates stacked in the order input, forget,
    candidate, output, with two bias vectors that are added together; the forget
    gate is the second quarter of each.
    """
    with torch.no_grad():
        for name, p in model.rnn.named_parameters():
            if name.startswith("bias"):
                h = p.shape[0] // 4
                p[h:2 * h] = bias if name.startswith("bias_ih") else 0.0
    return model


def position_gradients(model, X, y):
    """The size of the loss's gradient with respect to the word at each position.

    Averaged over the rows of X. Position 0 is the first word, -1 the last.
    """
    model.eval()
    e = model.emb(X).detach().requires_grad_(True)       # the embeddings, as a leaf we can ask about
    lengths = (X != PAD).sum(1).clamp(min=1)
    nn.functional.cross_entropy(model.forward_embedded(e, lengths), y).backward()
    return e.grad.norm(dim=2).mean(0)


def train(model, X, y, epochs=3, optimizer=None, batch_size=64, seed=0, clip=1.0):
    """Mini-batch training with cross-entropy and gradient clipping; returns each epoch's loss."""
    torch.manual_seed(seed)
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=3e-3)
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
        history.append(round(total / len(X), 4))
    return history


def evaluate(model, X, y):
    """Accuracy and macro-F1 (the mean of each category's F1)."""
    from sklearn.metrics import f1_score
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(X[i:i + 2048]).argmax(1) for i in range(0, len(X), 2048)])
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
        raise ValueError(f"{saved['class']} is not defined in headlines.py; define the class "
                         "here, not in a notebook cell, so it can be rebuilt")
    model = cls(**saved.get("kwargs", {}))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model
