"""slm: the small language model and the Kittiwake notices, for the Lab 12 notebooks.

Everything is offline. The model is HuggingFaceTB/SmolLM2-360M-Instruct (Apache 2.0),
the embedding model ibm-granite/granite-embedding-small-english-r2 (Apache 2.0), both
already in the image under HF_HOME. The notices are notices.txt beside this file:
the five in the image at /opt/nlplab/data/kittiwake_notices.txt, plus seventeen more
from the same invented company written for this lab, one notice per line (lines
starting with # are comments).

    tok, model = load()                     the tokenizer and the model, loaded once
    next_token_table(text, k=5)             the k most likely next tokens, with logit and probability
    generate(messages, max_new_tokens=32)   greedy by default; temperature and top_p to sample
    notices()                               the 22 notices, as a list of strings
    bm25_search(query, k=2)                 [(notice index, score)] by BM25, with bm25s
    embed_search(query, k=2)                [(notice index, cosine)] by granite embeddings
"""
import json
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
EMBED_ID = "ibm-granite/granite-embedding-small-english-r2"
THREADS = 4
DEPARTMENTS = ["billing", "network", "device", "account"]

_cache = {}


def load():
    """Load SmolLM2-360M-Instruct once per kernel: 362 million weights, 1.4 GB as float32."""
    if "model" not in _cache:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        transformers.utils.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
        torch.set_num_threads(THREADS)
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        # float32, not the bfloat16 the weights are stored in: exact softmax arithmetic, and fast on
        # processors without bfloat16 instructions.
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
        model.eval()
        _cache["tok"], _cache["model"] = tok, model
    return _cache["tok"], _cache["model"]


def as_text(messages):
    """The exact text the model reads for a chat: the chat template filled in."""
    tok, _ = load()
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def last_logits(text):
    """The model's logits for the token that would come after text: one number per vocabulary entry."""
    import torch
    tok, model = load()
    ids = tok(text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        return model(**ids).logits[0, -1]


def next_token_table(text, k=5):
    """The k most likely next tokens after text, as (token, logit, probability)."""
    tok, _ = load()
    logits = last_logits(text)
    probs = logits.softmax(-1)
    top = probs.topk(k)
    return [(tok.decode([int(i)]), round(float(logits[i]), 2), round(float(p), 4))
            for p, i in zip(top.values, top.indices)]


def generate(prompt, max_new_tokens=32, temperature=0.0, top_p=1.0, seed=None, timed=False):
    """Continue a prompt. prompt is a string (read as raw text) or a list of chat messages
    (wrapped in the chat template). temperature 0 is greedy: always the most likely token."""
    import torch
    tok, model = load()
    text = prompt if isinstance(prompt, str) else as_text(prompt)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)
    kw = {"max_new_tokens": max_new_tokens, "pad_token_id": tok.eos_token_id}
    if temperature and temperature > 0:
        if seed is not None:
            torch.manual_seed(seed)
        kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
    else:
        kw.update(do_sample=False)
    t = time.time()
    with torch.no_grad():
        out = model.generate(**ids, **kw)
    secs = time.time() - t
    new = out[0, ids["input_ids"].shape[1]:]
    reply = tok.decode(new, skip_special_tokens=True).strip()
    if timed:
        return reply, {"prompt_tokens": int(ids["input_ids"].shape[1]), "new_tokens": int(len(new)),
                       "seconds": round(secs, 2)}
    return reply


def notices():
    """The Kittiwake service notices, one string each, in file order (index 0 is the first line)."""
    if "notices" not in _cache:
        with open(os.path.join(HERE, "notices.txt")) as f:
            _cache["notices"] = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return _cache["notices"]


def bm25_search(query, k=2):
    """Rank notices by BM25 (bm25s, English stop words removed). Returns [(index, score)]."""
    import bm25s
    if "bm25" not in _cache:
        r = bm25s.BM25()
        r.index(bm25s.tokenize(notices(), stopwords="en", show_progress=False), show_progress=False)
        _cache["bm25"] = r
    ids, scores = _cache["bm25"].retrieve(bm25s.tokenize([query], stopwords="en", show_progress=False),
                                          k=k, show_progress=False)
    return [(int(i), round(float(s), 3)) for i, s in zip(ids[0], scores[0])]


def _embedder():
    if "embed" not in _cache:
        from sentence_transformers import SentenceTransformer
        _cache["embed"] = SentenceTransformer(EMBED_ID)
    return _cache["embed"]


def notice_vectors():
    """One unit-length 384-number vector per notice. Read from out/.notice_vectors.npy when the
    session's background job has made it, otherwise computed here (a few seconds) and saved."""
    import numpy as np
    if "vectors" in _cache:
        return _cache["vectors"]
    path = os.path.join(OUT, ".notice_vectors.npy")
    if os.path.exists(path):
        v = np.load(path)
        if v.shape[0] == len(notices()):
            _cache["vectors"] = v
            return v
    v = _embedder().encode(notices(), normalize_embeddings=True)
    os.makedirs(OUT, exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npy"
    np.save(tmp, v)
    os.replace(tmp, path)
    _cache["vectors"] = v
    return v


def embed_search(query, k=2):
    """Rank notices by cosine similarity of granite embeddings. Returns [(index, cosine)]."""
    v = notice_vectors()
    q = _embedder().encode([query], normalize_embeddings=True)[0]
    sims = v @ q
    order = sims.argsort()[::-1][:k]
    return [(int(i), round(float(sims[i]), 3)) for i in order]


def save_json(name, obj):
    """Write out/<name> atomically."""
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    with open(p + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(p + ".tmp", p)
    return p


# ------------------------------------------------------- routing tickets ---

ROUTING_PROMPT = ("Route this Kittiwake Mobile support ticket to one department.\n"
                  "billing: charges, bills, payments, refunds, prices\n"
                  "network: signal, calls dropping, data not working, outages\n"
                  "device: handsets, phones, repairs, screens, batteries, SIM cards\n"
                  "account: login, password, address, plan changes, adding or moving lines")
TICKETS_CSV = os.path.join(os.environ.get("NLPLAB_DATA", "/opt/nlplab/data"), "kittiwake_tickets.csv")


def tickets():
    """Lab 03's 600 labelled tickets: the first 500 to train on, the last 100 held out."""
    import pandas as pd
    d = pd.read_csv(TICKETS_CSV)
    return d.iloc[:500].reset_index(drop=True), d.iloc[500:].reset_index(drop=True)


def routing_sample():
    """The 12 held-out tickets the language model routes (fixed, so everyone sees the same)."""
    _, test = tickets()
    return test.sample(12, random_state=1).reset_index(drop=True)


def department_logits(text, prompt=ROUTING_PROMPT):
    """The four department scores for one ticket: the logits of ' billing', ' network', ' device'
    and ' account' as the next token after the chat template and 'Department:'."""
    tok, _ = load()
    ids = [tok.encode(" " + d)[0] for d in DEPARTMENTS]
    s = as_text([{"role": "system", "content": prompt}, {"role": "user", "content": text}]) + "Department:"
    return [round(float(v), 4) for v in last_logits(s)[ids]]


def routing_logits(texts, prompt=ROUTING_PROMPT, verbose=True):
    """department_logits for every text. Uses out/.routing_logits.json when the session's background
    job has already computed exactly these tickets with exactly this prompt."""
    import hashlib
    key = hashlib.sha256(json.dumps([prompt, list(texts)]).encode()).hexdigest()[:16]
    path = os.path.join(OUT, ".routing_logits.json")
    try:
        with open(path) as f:
            cached = json.load(f)
        if cached.get("key") == key:
            if verbose:
                print(f"read from out/.routing_logits.json (computed by the session at start, in {cached['seconds']} s)")
            return cached["logits"]
    except (OSError, ValueError):
        pass
    if verbose:
        print(f"computing {len(texts)} tickets now, one forward pass each ...")
    t = time.time()
    logits = [department_logits(x, prompt) for x in texts]
    save_json(".routing_logits.json", {"key": key, "logits": logits, "seconds": round(time.time() - t, 1)})
    if verbose:
        print(f"done in {time.time() - t:.0f} s")
    return logits


# ------------------------------------------------------- the question set ---

def run_question_set(answer, save=True):
    """Run answer(question) over Part 3's fixed question set, time it, and save
    out/assistant_answers.json. answer() returns a dict with at least "answer", and, when
    the model was asked, the "messages" it read and the "max_new_tokens" it was allowed."""
    from nlpcheck import QUESTION_SET
    results, t0 = [], time.time()
    for q, _ in QUESTION_SET:
        t = time.time()
        r = answer(q)
        if isinstance(r, str):
            r = {"answer": r}
        r = dict(r, question=q, seconds=round(time.time() - t, 1))
        results.append(r)
        print(f"{r['seconds']:5.1f} s  {q}\n         -> {r.get('answer')}")
    secs = round(time.time() - t0, 1)
    print(f"{len(results)} questions in {secs} s")
    if save:
        save_json("assistant_answers.json", {"seconds": secs, "results": results})
    return results


if __name__ == "__main__":
    # The session's background warm-up: read the weights once so the notebook's first load
    # comes from memory, and store the notice vectors.
    t = time.time()
    notice_vectors()
    load()
    routing_logits(list(routing_sample().text), verbose=False)
    print(f"warm-up done in {time.time() - t:.0f} s")
