"""nlpcheck: the feedback half of the Lab 05 notebooks.

Three kinds of help, used the same way in every lab of the course:

    ask("r1", "b")               answer a recall question from memory; you are
                                 told at once whether it is right, and why.
    guess("hot_cold", 0.2)       commit to a prediction BEFORE the cell that
    reveal("hot_cold", actual)   shows the answer; reveal() compares the two.
    check_05_01()                check a saved exercise the way the checkpoint
                                 will, with a message that says what to fix.

Predictions are kept in out/predictions.json. Nothing is marked on them: they
are there because a guess you wrote down is one you remember being wrong
about, and the sign-off counts how many you made. A numeric prediction counts
as right when it is within 0.1 of the answer.
"""
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PRED = os.path.join(OUT, "predictions.json")

# ---------------------------------------------------------------- recall ---

QUESTIONS = {
    # 05_01, from Lab 02 and the chapter pages
    "r1": ("0", "In Lab 02, TF-IDF gave 'a spooky film' and 'a scary movie' a cosine of 0: they share no "
                "word, and counting cannot see that two different words mean the same thing."),
    "r2": ("b", "Skip-gram takes the centre word and predicts the words around it. CBOW does the "
                "opposite: it takes the surrounding words and predicts the centre."),
    # 05_02, from 05_01
    "r3": ("queen", "king - man + woman lands nearest 'queen': the step from man to woman, added to king."),
    "r4": ("c", "Opposites such as hot and cold appear in the same sentences ('the water was hot', 'the "
                "water was cold'), so a model that learns from context puts them close together."),
    # 05_03, from 05_02
    "r5": ("a", "A pair is (centre, context) for every word within the window, on both sides. Each "
                "real pair is pulled together and a few random ones pushed apart: that is negative sampling."),
    "r6": ("b", "Gullhaven and Saltmoor share one ticket, but they share hundreds of contexts: 'calls "
                "drop in ...', 'no signal in ...'. Words used the same way end up in the same place."),
    # exit tickets
    "x1": ("c", "A static vector gives 'bank' one point whatever the sentence; only a contextual model "
                "can give the river bank and the savings bank different vectors."),
    "x2": ("b", "Hit rate at 10 asks whether any of the ten recommendations was really bought in the "
                "same invoice. It says nothing about the other nine."),
}


def ask(qid, answer):
    """Check a recall answer. Case and surrounding spaces do not matter."""
    if qid not in QUESTIONS:
        print(f"There is no question {qid!r} in this lab.")
        return
    right, why = QUESTIONS[qid]
    given = str(answer).strip().lower()
    if given in ("", "?", "none"):
        print("Answer first, from memory. Being wrong here is useful; skipping it is not.")
    elif given == right.lower():
        print(f"Right. {why}")
    else:
        print(f"Not quite: the answer is {right}. {why}")


# ----------------------------------------------------------- predictions ---

def _load():
    try:
        with open(PRED) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _same(g, a):
    try:
        return abs(float(g) - float(a)) <= 0.1
    except (TypeError, ValueError):
        return str(g).strip().lower() == str(a).strip().lower()


def guess(key, value):
    """Record a prediction before running the cell that answers it."""
    if value is None or value == "":
        print("Write a prediction first. A guess is fine; that is the point.")
        return
    os.makedirs(OUT, exist_ok=True)
    preds = _load()
    preds[key] = {"guess": value, "actual": preds.get(key, {}).get("actual")}
    with open(PRED, "w") as f:
        json.dump(preds, f, indent=1, default=str)
    print(f"Prediction recorded: {value!r}. Now run the next cell.")


def reveal(key, actual):
    """Compare the recorded prediction with what actually happened."""
    preds = _load()
    if key not in preds:
        print(f"Actual: {actual!r}. You did not record a prediction for this one; "
              "go back one cell and make one before you read on.")
        return
    g = preds[key]["guess"]
    preds[key]["actual"] = actual
    with open(PRED, "w") as f:
        json.dump(preds, f, indent=1, default=str)
    if _same(g, actual):
        print(f"You predicted {g!r}, and it is {actual!r}. Right.")
    else:
        print(f"You predicted {g!r}. It is {actual!r}. The text below the cell explains the gap.")


# -------------------------------------------------------------- checks -----

def _read(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None, f"out/{name} does not exist yet. Run the cell that saves it."
    try:
        with open(p) as f:
            return json.load(f), None
    except ValueError as e:
        return None, f"out/{name} is not valid JSON ({e}). Run the saving cell again."


def _report(problems, ok):
    if problems:
        print("Not yet:")
        for p in problems:
            print("  -", p)
        return False
    print(ok)
    return True


NEIGHBOUR_WORDS = ["phone", "refund", "spooky"]
ANALOGIES = [("man", "king", "woman"), ("france", "paris", "italy"), ("uk", "london", "canada")]


def glove_answers():
    """What 05_01 section 7 should save, computed from the GloVe files."""
    from w2vtools import load_glove, nearest
    words, ix, V = load_glove()
    neighbours = {w: [n for n, _ in nearest(V, words, V[ix[w]], 5, exclude=(w,))] for w in NEIGHBOUR_WORDS}
    analogies = {}
    for a, b, c in ANALOGIES:
        analogies[f"{b} - {a} + {c}"] = nearest(V, words, V[ix[b]] - V[ix[a]] + V[ix[c]], 1, exclude=(a, b, c))[0][0]
    return neighbours, analogies


def check_05_01():
    """Neighbours of three words and three analogies, from your own functions."""
    got, err = _read("05_01_glove.json")
    if err:
        return _report([err], "")
    neighbours, analogies = glove_answers()
    problems = []
    for w, want in neighbours.items():
        have = (got.get("neighbours") or {}).get(w)
        if have != want:
            hint = " (did you leave the word itself in? it is always its own nearest neighbour)" if have and have[0] == w else ""
            problems.append(f"neighbours of {w!r} should be {want}; you have {have}{hint}.")
    for q, want in analogies.items():
        have = (got.get("analogies") or {}).get(q)
        if have != want:
            hint = " (exclude the three input words, or the answer is usually one of them)" if have in q.split() else ""
            problems.append(f"{q} should land nearest {want!r}; you have {have!r}{hint}.")
    return _report(problems, "Right: " + "; ".join(f"{q} = {a}" for q, a in analogies.items())
                   + ". The last one is wrong in an instructive way; section 4 says why.")


TOWNS = ["gullhaven", "saltmoor", "cragwell", "marrowby", "wrenfield", "lowmere", "brackenridge"]
HANDSETS = ["nimbus", "corvid", "aster"]


def check_05_02():
    """Your pair count, and vectors in which the towns and handsets have clustered."""
    from w2vtools import kittiwake_sentences, skipgram_pairs
    import collections as c
    got, err = _read("05_02_pairs.json")
    if err:
        return _report([err], "")
    s = kittiwake_sentences()
    counts = c.Counter(t for x in s for t in x)
    vocab = [w for w, n in counts.most_common() if n >= 3]
    ix = {w: i for i, w in enumerate(vocab)}
    problems = []
    for window in (1, 3):
        want = len(skipgram_pairs(s, ix, window))
        have = got.get(f"window_{window}")
        if have != want:
            problems.append(f"with window {window} there are {want} pairs; your make_pairs gives {have}. "
                            "Count every word up to `window` places away on BOTH sides, and never the centre itself.")
    p = os.path.join(OUT, "05_02_vectors.npz")
    if not os.path.exists(p):
        problems.append("out/05_02_vectors.npz does not exist yet. Run the save cell in section 6.")
        return _report(problems, "")
    z = np.load(p)
    words, E = list(z["vocab"]), z["vectors"].astype(np.float32)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    idx = {w: i for i, w in enumerate(words)}
    for name, group in (("towns", TOWNS), ("handsets", HANDSETS)):
        g = [idx[w] for w in group if w in idx]
        if len(g) < 3:
            problems.append(f"your vectors are missing the {name}; save the model you trained on all the tickets.")
            continue
        S = E[g] @ E[g].T
        within = (S.sum() - len(g)) / (len(g) * (len(g) - 1))
        overall = float((E[g] @ E.T).mean())
        if within - overall < 0.25:
            problems.append(f"the {name} are not clustered yet (mean similarity {within:.2f} to each other, "
                            f"{overall:.2f} to everything). Train for the full number of epochs.")
    return _report(problems, "Right: the pair counts match, and in your vectors the towns and the handsets "
                             "have each found one another.")


def hit_rate_reference(recommend, queries, truth):
    return round(float(np.mean([bool(set(recommend(p)) & truth[q]) for q, p in queries])), 3)


def check_05_03():
    """The three hit rates from section 4, on the lab's 1,000 evaluation queries."""
    from w2vtools import load_baskets, part3_queries, part3_truth, cooccurrence
    got, err = _read("05_03_eval.json")
    if err:
        return _report([err], "")
    tr, te, _ = load_baskets()
    counts = collections.Counter(c for b in tr for c in b)
    known = {c for c, n in counts.items() if n >= 5}
    q, t = part3_queries(te, known, n=1000, seed=11), part3_truth(te, known, n=1000, seed=11)
    pop = [c for c, _ in counts.most_common(11)]
    together = cooccurrence(tr)
    want_pop = hit_rate_reference(lambda p: [c for c in pop if c != p][:10], q, t)
    want_co = hit_rate_reference(lambda p: together[p][:10], q, t)
    problems = []
    if got.get("popularity") != want_pop:
        problems.append(f"popularity should score {want_pop}; you have {got.get('popularity')}. Check hit_rate in section 4.")
    if got.get("cooccurrence") != want_co:
        problems.append(f"co-occurrence should score {want_co}; you have {got.get('cooccurrence')}.")
    sg = got.get("skipgram")
    if not isinstance(sg, (int, float)) or not 0.45 <= sg <= 0.75:
        problems.append(f"the skip-gram score should be between 0.45 and 0.75 for the vectors of "
                        f"section 2; you have {sg}.")
    return _report(problems, f"Right: popularity {want_pop}, skip-gram {sg}, counting co-purchases {want_co}.")


def check_on_your_own():
    """Part 3: out/recommendations.json, ten products for each of the 500 queries."""
    from w2vtools import load_baskets, part3_queries, part3_truth
    got, err = _read("recommendations.json")
    if err:
        return _report([err], "")
    tr, te, _ = load_baskets()
    counts = collections.Counter(c for b in tr for c in b)
    known = {c for c, n in counts.items() if n >= 5}
    queries = part3_queries(te, known)
    truth = part3_truth(te, known)
    problems = []
    missing = [q for q, _ in queries if q not in got]
    if missing:
        return _report([f"{len(missing)} of the 500 queries have no recommendations, for example {missing[0]}."], "")
    for q, p in queries:
        recs = got[q]
        if not isinstance(recs, list) or len(recs) != 10 or len(set(recs)) != 10:
            problems.append(f"{q} needs exactly ten different stock codes; it has {recs!r:.60}.")
            break
        if p in recs:
            problems.append(f"{q} recommends the product it was asked about ({p}); leave it out.")
            break
    if problems:
        return _report(problems, "")
    hr = round(float(np.mean([bool(set(got[q]) & truth[q]) for q, _ in queries])), 3)
    if hr < 0.50:
        return _report([f"hit rate at 10 is {hr}; the brief asks for at least 0.50. Recommending the "
                        f"best-sellers to everyone scores about 0.34."], "")
    return _report([], f"Right: hit rate at 10 is {hr} on the 500 queries (popularity scores about 0.34).")
