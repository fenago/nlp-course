"""translate: the English-Spanish data, the encoder-decoder and its scores, in one file.

Why a file and not notebook cells: a saved PyTorch model is only its numbers
(its state_dict). To use it again, whoever loads it (you tomorrow, or the
checkpoint today) has to rebuild the same network first, by importing the
class that defines it. A class defined in a notebook cell exists only in that
notebook's kernel; a class defined here can be imported by anything.

    load_split()                 the pairs, tokenized, split by English sentence
    Seq2Seq                      a GRU encoder, one context vector, a GRU decoder
    train(model, data, ...)      teacher-forced training (or not, to compare)
    greedy(model, sentences)     translate by always taking the likeliest word
    beam(model, sentence, k)     translate by keeping the k likeliest prefixes
    eval_split(data)             the held-out sentences every score in the lab uses
    bleu(hyps, refs)             corpus BLEU, pure Python
    chrf(hyps, refs)             corpus chrF, pure Python
    save_model / load_model      save with what rebuilds it (and its steps_trained), and rebuild it
    lab_model()                  the model the session trained in the background

The pairs are from the Tatoeba Project (https://tatoeba.org), as packaged by
manythings.org (https://www.manythings.org/anki/), under Creative Commons
Attribution 2.0 France. They are already in the image; nothing downloads.

You may add your own model class to this file for Part 3. save_model records
the class by name, so load_model can find it here.
"""
import collections
import gzip
import hashlib
import math
import os
import re
import time

import torch
import torch.nn as nn

PAIRS = os.path.join(os.environ.get("NLPLAB_DATA", "/opt/nlplab/data"), "eng_spa", "pairs.tsv.gz")
PAD, UNK, SOS, EOS = 0, 1, 2, 3
SPECIALS = ["<pad>", "<unk>", "<sos>", "<eos>"]
SRC_VOCAB = 6000      # the 6,000 commonest English training words
TGT_VOCAB = 8000      # the 8,000 commonest Spanish training words
MAX_OUT = 16          # the longest translation the decoder may write, in tokens
TEST_SHARE = 20       # one English sentence in 20 is held out
MAX_BEAM = 5          # the widest beam Part 3 allows
LAB_STEPS = 1392      # the lab model's training: 2 epochs of 696 batches of 128
BUDGET_STEPS = 700    # Part 3: training steps allowed beyond LAB_STEPS, about one more epoch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
CACHE = os.path.join(OUT, ".pairs.pt")
LAB_MODEL = os.path.join(OUT, ".lab_model.pt")


def tokens(text):
    """Lowercase words, with each punctuation mark as its own token (keeps accents, ¿ and ¡)."""
    return re.findall(r"[\w']+|[^\w\s]", text.lower())


def held_out(english):
    """Whether an English sentence is in the test split: a fixed hash, so everyone gets the same split."""
    return int(hashlib.md5(english.lower().encode()).hexdigest(), 16) % TEST_SHARE == 0


def _build():
    with gzip.open(PAIRS, "rt", encoding="utf-8") as f:
        rows = [line.rstrip("\n").split("\t") for line in f][1:]
    rows = [(e, s) for e, s in rows if e and s]
    train = [(tokens(e), tokens(s)) for e, s in rows if not held_out(e)]
    src_counts = collections.Counter(w for e, _ in train for w in e)
    tgt_counts = collections.Counter(w for _, s in train for w in s)
    src_itos = SPECIALS + [w for w, _ in src_counts.most_common(SRC_VOCAB)]
    tgt_itos = SPECIALS + [w for w, _ in tgt_counts.most_common(TGT_VOCAB)]
    tgt_stoi = {w: i for i, w in enumerate(tgt_itos)}
    # train only on pairs whose Spanish is all in the vocabulary, so the decoder never learns to write <unk>
    train = [(e, s) for e, s in train if all(w in tgt_stoi for w in s)]
    refs = collections.defaultdict(list)
    for e, s in rows:
        if held_out(e):
            refs[e.lower()].append(tokens(s))
    test = sorted(refs)       # each held-out English sentence once, with all its translations
    return {"src_itos": src_itos, "tgt_itos": tgt_itos, "train": train,
            "test_src": [tokens(e) for e in test], "test_refs": [refs[e] for e in test]}


def load_split():
    """The data, built once and cached in out/ (about ten seconds the first time)."""
    if os.path.exists(CACHE):
        try:
            return torch.load(CACHE, weights_only=False)
        except Exception:
            pass
    data = _build()
    os.makedirs(OUT, exist_ok=True)
    tmp = CACHE + ".tmp%d" % os.getpid()
    torch.save(data, tmp)
    os.replace(tmp, CACHE)      # atomic: a half-written cache is never read
    return data


def encode(words, itos, eos=False, sos=False):
    stoi = {w: i for i, w in enumerate(itos)}
    ids = [stoi.get(w, UNK) for w in words]
    return ([SOS] if sos else []) + ids + ([EOS] if eos else [])


def batchify(seqs):
    """A list of id lists as one padded (batch, longest) tensor."""
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (n - len(s)) for s in seqs])


class Seq2Seq(nn.Module):
    """The book's encoder-decoder, with GRUs: the encoder's last state is the only thing the decoder sees."""

    def __init__(self, src_vocab=SRC_VOCAB + 4, tgt_vocab=TGT_VOCAB + 4, emb=128, hidden=256,
                 reverse_source=False, dropout=0.0):
        super().__init__()
        self.reverse_source = reverse_source
        self.src_emb = nn.Embedding(src_vocab, emb, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, emb, padding_idx=PAD)
        self.encoder = nn.GRU(emb, hidden, batch_first=True)
        self.decoder = nn.GRU(emb, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, tgt_vocab)

    def encode(self, src):
        """(batch, length) source ids to the context vector, shape (1, batch, hidden)."""
        lengths = (src != PAD).sum(1).clamp(min=1)
        if self.reverse_source:        # Sutskever et al.'s trick: read the sentence backwards
            src = torch.stack([torch.cat([s[:n].flip(0), s[n:]]) for s, n in zip(src, lengths)])
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.src_emb(src)), lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        _, h = self.encoder(packed)
        return h

    def decode_step(self, prev, h):
        """One decoder step: the previous word (batch,) and state to next-word scores and the new state."""
        o, h = self.decoder(self.tgt_emb(prev).unsqueeze(1), h)
        return self.out(o.squeeze(1)), h

    def forward(self, src, tgt_in):
        """Teacher forcing: the decoder reads the true previous words, all positions in one call."""
        o, _ = self.decoder(self.drop(self.tgt_emb(tgt_in)), self.encode(src))
        return self.out(self.drop(o))


def make_batches(model_data, pairs, batch_size, seed):
    src_stoi = {w: i for i, w in enumerate(model_data["src_itos"])}
    tgt_stoi = {w: i for i, w in enumerate(model_data["tgt_itos"])}
    order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed)).tolist()
    for i in range(0, len(order), batch_size):
        chunk = [pairs[j] for j in order[i:i + batch_size]]
        src = batchify([[src_stoi.get(w, UNK) for w in e] for e, _ in chunk])
        tgt = batchify([[SOS] + [tgt_stoi.get(w, UNK) for w in s] + [EOS] for _, s in chunk])
        yield src, tgt


def train(model, data, epochs=1, lr=2e-3, batch_size=128, pairs=None, seed=0, teacher_forcing=True,
          max_seconds=None, max_steps=None, clip=1.0, log_every=0, optimizer=None):
    """Train with cross-entropy on every target word. Returns the mean loss of each epoch.

    teacher_forcing=False feeds the decoder its own previous prediction instead of the
    true word, one step at a time, to show why nobody trains that way.
    max_seconds stops training when the time is up, whatever the epoch; max_steps stops it
    after that many optimizer steps. Every step is added to model.steps_trained, which
    save_model records.
    """
    torch.manual_seed(seed)
    pairs = data["train"] if pairs is None else pairs
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    history, start, step = [], time.time(), 0
    for epoch in range(epochs):
        model.train()
        total, words = 0.0, 0
        for src, tgt in make_batches(data, pairs, batch_size, seed + epoch):
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            if teacher_forcing:
                logits = model(src, tgt_in)
            else:
                h, prev, steps = model.encode(src), tgt_in[:, 0], []
                for t in range(tgt_out.size(1)):
                    step_logits, h = model.decode_step(prev, h)
                    steps.append(step_logits)
                    prev = step_logits.argmax(1)        # its own guess, not the true word
                logits = torch.stack(steps, 1)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
                                               ignore_index=PAD)
            optimizer.zero_grad()
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            n = int((tgt_out != PAD).sum())
            total, words, step = total + float(loss.detach()) * n, words + n, step + 1
            if log_every and step % log_every == 0:
                print(f"  step {step}: loss {total / words:.3f} ({time.time() - start:.0f} s)", flush=True)
            if (max_seconds and time.time() - start > max_seconds) or (max_steps and step >= max_steps):
                history.append(total / max(words, 1))
                model.steps_trained = getattr(model, "steps_trained", 0) + step
                print(f"stopped at the {'step' if max_steps and step >= max_steps else 'time'} limit: "
                      f"{step} steps, {step * batch_size:,} pairs, {time.time() - start:.0f} s", flush=True)
                return history
        history.append(total / max(words, 1))
    model.steps_trained = getattr(model, "steps_trained", 0) + step
    return history


def _words(ids, itos):
    out = []
    for i in ids:
        if i == EOS:
            break
        out.append(itos[i])
    return out


@torch.no_grad()
def greedy(model, sentences, data, batch_size=256, max_len=MAX_OUT):
    """Translate a list of token lists, taking the single likeliest word at every step."""
    model.eval()
    src_stoi = {w: i for i, w in enumerate(data["src_itos"])}
    result = []
    for i in range(0, len(sentences), batch_size):
        src = batchify([[src_stoi.get(w, UNK) for w in s] or [UNK] for s in sentences[i:i + batch_size]])
        h = model.encode(src)
        prev = torch.full((src.size(0),), SOS)
        steps = []
        for _ in range(max_len):
            logits, h = model.decode_step(prev, h)
            prev = logits.argmax(1)
            steps.append(prev)
        ids = torch.stack(steps, 1).tolist()
        result += [_words(r, data["tgt_itos"]) for r in ids]
    return result


@torch.no_grad()
def beam(model, sentence, data, k=5, max_len=MAX_OUT, alpha=0.7):
    """Translate one token list, keeping the k likeliest prefixes at every step.

    Finished translations are compared by log-probability divided by length ** alpha,
    because a raw sum of log-probabilities always prefers the shorter sentence.
    """
    model.eval()
    src_stoi = {w: i for i, w in enumerate(data["src_itos"])}
    h = model.encode(torch.tensor([[src_stoi.get(w, UNK) for w in sentence] or [UNK]]))
    live = [(0.0, [SOS], h)]
    done = []
    for _ in range(max_len):
        prev = torch.tensor([seq[-1] for _, seq, _ in live])
        hs = torch.cat([h for _, _, h in live], 1)
        logits, hs = model.decode_step(prev, hs)
        logp = torch.log_softmax(logits, -1)
        cands = []
        for b, (score, seq, _) in enumerate(live):
            top = logp[b].topk(k)
            for lp, w in zip(top.values.tolist(), top.indices.tolist()):
                cands.append((score + lp, seq + [w], hs[:, b:b + 1]))
        cands.sort(key=lambda c: -c[0])
        live = []
        for score, seq, hb in cands:
            if seq[-1] == EOS:
                done.append((score / (len(seq) - 1) ** alpha, seq))
            else:
                live.append((score, seq, hb))
            if len(live) == k:
                break
        if not live or (len(done) >= k and max(d[0] for d in done) > live[0][0] / max_len ** alpha):
            break
    done += [(s / (len(q) - 1) ** alpha, q) for s, q, _ in live]
    best = max(done, key=lambda d: d[0])[1]
    return _words(best[1:], data["tgt_itos"])


def translate_all(model, sentences, data, k=1):
    """Translate a list of token lists: batched greedy for k = 1, beam search one by one otherwise."""
    return greedy(model, sentences, data) if k == 1 else [beam(model, s, data, k=k) for s in sentences]


def eval_split(data):
    """Every fourth held-out English sentence, with all its references: the set every score in this lab uses."""
    return data["test_src"][::4], data["test_refs"][::4]


def translate(model, text, data, k=1):
    """One English string to one Spanish string."""
    words = tokens(text)
    out = greedy(model, [words], data)[0] if k == 1 else beam(model, words, data, k=k)
    return " ".join(out)


# ------------------------------------------------------------- scores -----

def ngrams(seq, n):
    return collections.Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))


def bleu(hyps, refs, max_n=4):
    """Corpus BLEU-4 on tokens, with several references per sentence (Papineni et al., 2002).

    For each n, clipped n-gram matches over all sentences divided by n-grams written;
    the geometric mean of the four precisions; times a brevity penalty for writing
    less than the closest reference length. Between 0 and 100.
    """
    match, total = [0] * max_n, [0] * max_n
    hyp_len = ref_len = 0
    for h, rs in zip(hyps, refs):
        hyp_len += len(h)
        ref_len += min((abs(len(r) - len(h)), len(r)) for r in rs)[1]
        for n in range(1, max_n + 1):
            hc = ngrams(h, n)
            most = collections.Counter()
            for r in rs:
                most |= ngrams(r, n)        # the most times each n-gram appears in any one reference
            match[n - 1] += sum(min(c, most[g]) for g, c in hc.items())
            total[n - 1] += max(len(h) - n + 1, 0)
    if min(match) == 0:
        return 0.0
    log_p = sum(math.log(m / t) for m, t in zip(match, total)) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100 * bp * math.exp(log_p)


def _chr_stats(h, r, max_n):
    h, r = "".join(h), "".join(r)        # characters, spaces removed, as sacreBLEU's chrF does
    stats = []
    for n in range(1, max_n + 1):
        hc, rc = ngrams(h, n), ngrams(r, n)
        stats.append((sum((hc & rc).values()), sum(hc.values()), sum(rc.values())))
    return stats


def _chr_f(stats, beta):
    ps, rs = [], []
    for m, hn, rn in stats:
        if hn and rn:
            ps.append(m / hn)
            rs.append(m / rn)
    if not ps:
        return 0.0
    p, r = sum(ps) / len(ps), sum(rs) / len(rs)
    return 0.0 if p + r == 0 else 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def chrf(hyps, refs, max_n=6, beta=2):
    """Corpus chrF (Popovic, 2015): character n-gram F-score, n from 1 to 6, recall weighted twice.

    Each sentence is compared with whichever of its references it matches best, and the
    counts are summed over the corpus before the score is computed. Between 0 and 100.
    """
    total = [[0, 0, 0] for _ in range(max_n)]
    for h, rs in zip(hyps, refs):
        best = max((_chr_stats(h, r, max_n) for r in rs), key=lambda s: _chr_f(s, beta))
        for n, (m, hn, rn) in enumerate(best):
            total[n][0] += m; total[n][1] += hn; total[n][2] += rn
    return _chr_f(total, beta)


# ------------------------------------------------------------- saving -----

def save_model(model, path, beam_size=1, **kwargs):
    """Save the weights, the class name, the arguments that rebuild it, and how to decode with it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    torch.save({"class": type(model).__name__, "kwargs": kwargs, "beam_size": beam_size,
                "steps_trained": getattr(model, "steps_trained", 0), "state_dict": model.state_dict()}, tmp)
    os.replace(tmp, path)


def load_model(path):
    """Rebuild a model saved with save_model; returns (model, beam_size). The class must be in this file."""
    saved = torch.load(path, weights_only=False)
    cls = globals().get(saved["class"])
    if cls is None:
        raise ValueError(f"{saved['class']} is not defined in translate.py; define the class "
                         "here, not in a notebook cell, so it can be rebuilt")
    model = cls(**saved.get("kwargs", {}))
    model.load_state_dict(saved["state_dict"])
    model.steps_trained = saved.get("steps_trained", 0)
    model.eval()
    return model, saved.get("beam_size", 1)


LAB_EPOCHS = 2


def train_lab_model():
    """The lab's own translator: Seq2Seq(hidden=256), 2 epochs, seed 0. The session starts this in the background."""
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    data = load_split()
    torch.manual_seed(0)
    model = Seq2Seq()
    t = time.time()
    hist = train(model, data, epochs=LAB_EPOCHS, log_every=200)
    save_model(model, LAB_MODEL)
    print(f"lab model trained in {time.time() - t:.0f} s, losses {[round(x, 3) for x in hist]}", flush=True)
    return model


def lab_model(wait=True):
    """The lab's pretrained translator. Waits for the background job; trains it here if there is none."""
    log = os.path.join(OUT, ".lab_model.log")
    if not os.path.exists(LAB_MODEL) and wait and os.path.exists(log):
        print("The session is still training the lab's model in the background; waiting for it.", flush=True)
        while not os.path.exists(LAB_MODEL):
            if "Error" in open(log).read() or time.time() - os.path.getmtime(log) > 300:
                break                  # the job failed, or has written nothing for five minutes
            time.sleep(2)
    if os.path.exists(LAB_MODEL):
        return load_model(LAB_MODEL)[0]
    print("No background model found, so this cell trains it now: several minutes. "
          "It is saved, so this happens once.", flush=True)
    return train_lab_model()
