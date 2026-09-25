"""bertlab: the models and Kittiwake data of Lab 11, in one importable file.

Everything here loads from the image, offline: nothing downloads.

    fill_mask(text)            bert-mini's guesses for the [MASK] in a sentence
    word_vector(sentence, w)   bert-mini's vector for word w in that sentence
    tickets(), unlabelled()    Kittiwake's 600 labelled and 200 unlabelled tickets
    split()                    the fixed 450 / 150 split of the labelled tickets
    load_classifier(name)      a BERT with a fresh 4-department head on top
    fine_tune(model, ...)      train it on tickets; returns the seconds it took
    predict(model, texts)      departments for new tickets
    save_classifier(model, p)  save the model and its tokenizer to a folder
    notices()                  Kittiwake's five service notices
    read(question, context)    the DistilBERT reader: the best answer span

The models are Google's small BERTs (Turc et al., 2019, Apache 2.0):
bert-mini, https://huggingface.co/google/bert_uncased_L-4_H-256_A-4, and
bert-tiny, https://huggingface.co/google/bert_uncased_L-2_H-128_A-2; and
DistilBERT fine-tuned on SQuAD 1.1 (Apache 2.0),
https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad.
"""
import json
import os
import time
import warnings

import torch

warnings.filterwarnings("ignore")
from transformers import (AutoModel, AutoModelForQuestionAnswering,  # noqa: E402
                          AutoModelForSequenceClassification, AutoTokenizer)
from transformers.utils import logging as _hf_logging  # noqa: E402

_hf_logging.set_verbosity_error()
_hf_logging.disable_progress_bar()
torch.set_num_threads(min(4, os.cpu_count() or 1))

MINI = "google/bert_uncased_L-4_H-256_A-4"
TINY = "google/bert_uncased_L-2_H-128_A-2"
READER = "distilbert/distilbert-base-uncased-distilled-squad"
DEPARTMENTS = ["account", "billing", "device", "network"]
DATA = os.environ.get("NLPLAB_DATA", "/opt/nlplab/data")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
_cache = {}


def _get(key, make):
    if key not in _cache:
        _cache[key] = make()
    return _cache[key]


def tokenizer(name=MINI):
    return _get(("tok", name), lambda: AutoTokenizer.from_pretrained(name))


# ------------------------------------------------------------ pretraining ---

def fill_mask(text, k=5, name=MINI):
    """The k likeliest WordPieces for the [MASK] in text, with their probabilities."""
    from transformers import AutoModelForMaskedLM
    tok = tokenizer(name)
    model = _get(("mlm", name), lambda: AutoModelForMaskedLM.from_pretrained(name).eval())
    enc = tok(text, return_tensors="pt")
    where = (enc.input_ids[0] == tok.mask_token_id).nonzero()
    if len(where) != 1:
        raise ValueError("put exactly one [MASK] in the sentence")
    with torch.no_grad():
        probs = model(**enc).logits[0, int(where[0])].softmax(-1)
    top = probs.topk(k)
    return [(tok.convert_ids_to_tokens(int(i)), round(float(p), 3)) for p, i in zip(top.values, top.indices)]


def word_vector(sentence, word, name=MINI):
    """bert-mini's last-layer vector for the first WordPiece of word in sentence."""
    tok = tokenizer(name)
    model = _get(("enc", name), lambda: AutoModel.from_pretrained(name).eval())
    enc = tok(sentence, return_tensors="pt")
    ids = enc.input_ids[0].tolist()
    piece = tok.convert_tokens_to_ids(tok.tokenize(word)[0])
    if piece not in ids:
        raise ValueError(f"{word!r} is not in {sentence!r}")
    with torch.no_grad():
        return model(**enc).last_hidden_state[0, ids.index(piece)]


def cosine(a, b):
    return round(float(torch.nn.functional.cosine_similarity(a, b, dim=0)), 3)


# ------------------------------------------------------------ fine-tuning ---

def tickets():
    import pandas as pd
    return pd.read_csv(f"{DATA}/kittiwake_tickets.csv")


def unlabelled():
    import pandas as pd
    return pd.read_csv(f"{DATA}/kittiwake_tickets_unlabeled.csv")


def split():
    """The same 450 training and 150 test tickets for everyone (seed 0, stratified)."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    d = tickets()
    return train_test_split(np.arange(len(d)), test_size=150, random_state=0, stratify=d.department)


def load_classifier(name=TINY, seed=0):
    """A pretrained BERT with a new, randomly initialised 4-department head."""
    torch.manual_seed(seed)
    return AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(DEPARTMENTS),
        id2label=dict(enumerate(DEPARTMENTS)), label2id={d: i for i, d in enumerate(DEPARTMENTS)})


def _encode(model, texts, max_length=64):
    tok = tokenizer(model.config._name_or_path or TINY)
    return tok(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")


def fine_tune(model, texts, labels, epochs=5, lr=3e-4, batch_size=16, seed=0, freeze_bert=False):
    """Train on (texts, department names); returns the seconds it took.

    freeze_bert=True trains only the new head and leaves the pretrained weights as
    they are, which is how the notebook shows what fine-tuning the rest is worth.
    """
    for p in model.base_model.parameters():
        p.requires_grad = not freeze_bert
    enc = _encode(model, texts)
    y = torch.tensor([DEPARTMENTS.index(l) for l in labels])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    torch.manual_seed(seed)
    t = time.time()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed + epoch))
        for i in range(0, len(y), batch_size):
            b = order[i:i + batch_size]
            opt.zero_grad()
            out = model(**{k: v[b] for k, v in enc.items()}, labels=y[b])
            out.loss.backward()
            opt.step()
    model.eval()
    return round(time.time() - t, 1)


def predict(model, texts):
    model.eval()
    with torch.no_grad():
        ids = model(**_encode(model, texts)).logits.argmax(-1).tolist()
    return [DEPARTMENTS[i] for i in ids]


def macro_f1(true, pred):
    from sklearn.metrics import f1_score
    return round(float(f1_score(list(true), list(pred), average="macro")), 3)


def save_classifier(model, path):
    """Save the weights, the config (with the department names) and the tokenizer."""
    model.save_pretrained(path)
    tokenizer(model.config._name_or_path or TINY).save_pretrained(path)


def load_saved(path):
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    _cache[("tok", model.config._name_or_path)] = AutoTokenizer.from_pretrained(path)
    return model


# ---------------------------------------------------- question answering ---

def notices():
    with open(f"{DATA}/kittiwake_notices.txt") as f:
        return [line.strip() for line in f if line.strip()]


def read(question, context, max_answer_tokens=30):
    """The reader's best span in context: the start and end with the highest
    start logit plus end logit, with end at or after start.

    Returns answer (the text), logit (start + end logit, comparable between
    contexts), prob (start probability times end probability, a softmax over
    this context only), and the token positions.
    """
    tok = tokenizer(READER)
    model = _get(("qa", READER), lambda: AutoModelForQuestionAnswering.from_pretrained(READER).eval())
    enc = tok(question, context, return_tensors="pt", return_offsets_mapping=True,
              truncation="only_second", max_length=384)
    offsets = enc.pop("offset_mapping")[0].tolist()
    in_context = [i for i, s in enumerate(enc.sequence_ids(0)) if s == 1]
    enc.pop("token_type_ids", None)          # DistilBERT has no segment embeddings
    with torch.no_grad():
        out = model(**enc)
    start, end = out.start_logits[0], out.end_logits[0]
    best = None
    for i in in_context:
        for j in in_context:
            if i <= j < i + max_answer_tokens:
                score = float(start[i] + end[j])
                if best is None or score > best[0]:
                    best = (score, i, j)
    score, i, j = best
    prob = float(start.softmax(-1)[i] * end.softmax(-1)[j])
    return {"answer": context[offsets[i][0]:offsets[j][1]], "logit": round(score, 2),
            "prob": round(prob, 3), "start": i, "end": j}


# ------------------------------------------------ the background warm-up ---

MINI_RESULT = os.path.join(OUT, ".bert_mini_split.json")


def mini_on_split():
    """bert-mini fine-tuned on the 450 tickets, scored on the 150: several times
    slower than bert-tiny, so setup.d runs it once at session start."""
    if os.path.exists(MINI_RESULT):
        with open(MINI_RESULT) as f:
            return json.load(f)
    d = tickets()
    tr, te = split()
    model = load_classifier(MINI)
    secs = fine_tune(model, d.text[tr], d.department[tr], epochs=5, lr=5e-5)
    res = {"model": "bert-mini", "weights": sum(p.numel() for p in model.parameters()),
           "train_seconds": secs,
           "macro_f1": macro_f1(d.department[te], predict(model, d.text[te]))}
    os.makedirs(OUT, exist_ok=True)
    tmp = MINI_RESULT + ".tmp%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, MINI_RESULT)             # atomic: a half-written file is never read
    return res
