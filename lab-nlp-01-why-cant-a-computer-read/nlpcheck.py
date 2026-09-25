"""nlpcheck: the feedback half of the Lab 01 notebooks.

Three kinds of help, used the same way in every lab of the course:

    ask("r1", "b")               answer a recall question from memory; you are
                                 told at once whether it is right, and why.
    guess("naive", 7)            commit to a prediction BEFORE the cell that
    reveal("naive", actual)      shows the answer; reveal() compares the two.
    check_01_01()                check a saved exercise the way the checkpoint
                                 will, with a message that says what to fix.

Predictions are kept in out/predictions.json. Nothing is marked on them: they
are there because a guess you wrote down is one you remember being wrong
about, and the sign-off counts how many you made.

This file is part of the lab. Reading it will not spoil much, but the
questions work better if you answer before you look.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "data")
PRED = os.path.join(OUT, "predictions.json")


# ---------------------------------------------------------------- recall ---

QUESTIONS = {
    # 01_01, from the chapter pages
    "r1": ("b", "'Visiting relatives can be boring' has one set of words and two structures: "
                "the relatives who visit are boring, or going to visit them is. That is syntactic ambiguity."),
    "r2": ("c", "Tokenization cuts text into units. Stemming, lemmatization and tagging all work on "
                "tokens, so they come after it."),
    # 01_02, from 01_01
    "r3": ("2", "NLTK follows the Penn Treebank convention: can't becomes 'ca' and \"n't\", so the "
                "negation is a token of its own."),
    "r4": ("a", "A subword tokenizer never meets an unknown word: anything it has not seen whole, it "
                "builds from smaller pieces it has."),
    # 01_03, from 01_02
    "r5": ("noun", "WordNetLemmatizer assumes every word is a noun unless told otherwise, which is why "
                   "'running' stayed 'running' and 'was' became 'wa'."),
    "r6": ("b", "NLTK's English stop list contains 'not', 'no' and 'nor', so removing stop words blindly "
                "removes negation, and 'not working' becomes 'working'."),
    # exit tickets
    "x1": ("b", "Four words make three bigrams: (Practice, makes), (makes, man), (man, perfect). "
                "In general n tokens make n - 1 bigrams."),
    "x2": ("a", "A lemma is a real dictionary word; a stem is whatever is left after the suffix rules "
                "run, and need not be a word at all ('univers', 'hobbi')."),
    "x3": ("c", "Entity recognition relies on capital letters, punctuation and the small words around a "
                "name. Run it on the raw text, before any cleaning."),
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
    if str(g).strip().lower() == str(actual).strip().lower():
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


def _notices():
    with open(os.path.join(DATA, "kittiwake_notices.txt")) as f:
        return f.read().strip().split("\n\n")


def _report(problems, ok):
    if problems:
        print("Not yet:")
        for p in problems:
            print("  -", p)
        return False
    print(ok)
    return True


def check_01_01():
    """The outage notice: sentences and words by NLTK, pieces by SmolLM2's tokenizer."""
    import nltk
    from transformers import AutoTokenizer
    got, err = _read("01_01_tokens.json")
    if err:
        return _report([err], "")
    text = _notices()[2]
    want_s = nltk.sent_tokenize(text)
    want_w = nltk.word_tokenize(text)
    smol = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    want_g = len(smol.tokenize(text))
    problems = []
    if got.get("sentences") != want_s:
        problems.append(f"'sentences' should be nltk.sent_tokenize(outage), {len(want_s)} strings; "
                        f"you have {len(got.get('sentences') or [])}.")
    if got.get("words") != want_w:
        n = len(got.get("words") or [])
        hint = " (text.split() gives fewer: it keeps punctuation glued to words)" if n and n < len(want_w) else ""
        problems.append(f"'words' should be nltk.word_tokenize(outage), {len(want_w)} tokens; you have {n}{hint}.")
    if got.get("lm_tokens") != want_g:
        problems.append(f"'lm_tokens' should be len(smol.tokenize(outage)), {want_g}; "
                        f"you have {got.get('lm_tokens')}.")
    return _report(problems, f"Right: {len(want_s)} sentences, {len(want_w)} NLTK tokens and "
                             f"{want_g} SmolLM2 pieces for the same {len(text.split())} words.")


# Negation as NLTK's tokenizer sees it (the notebooks' route), and as it really
# appears in the tickets, where SMS drops the apostrophe (Part 3's route).
APOS_NEG_RE = re.compile(r"\bnot\b|n't", re.I)
NEG_RE = re.compile(r"\bnot\b|n't|\b(cant|dont|wont|isnt|doesnt|didnt|havent|hasnt|wasnt|arent)\b", re.I)


def check_01_02():
    """The first 20 tickets, cleaned with negation kept and lemmas by part of speech."""
    import csv
    got, err = _read("01_02_clean.json")
    if err:
        return _report([err], "")
    with open(os.path.join(DATA, "kittiwake_tickets.csv")) as f:
        rows = list(csv.DictReader(f))[:20]
    problems = []
    if not isinstance(got, dict) or len(got) != 20:
        return _report(["out/01_02_clean.json should map the first 20 ticket ids to their cleaned "
                        "word lists (20 entries)."], "")
    lost = [r["ticket_id"] for r in rows if APOS_NEG_RE.search(r["text"]) and "not" not in got.get(r["ticket_id"], [])]
    if lost:
        problems.append(f"{len(lost)} tickets say 'not' (or n't) and their cleaned words do not, for example "
                        f"{lost[0]}. The planted stop list still removes negation: fix the KEEP set in section 4.")
    words = [w for ws in got.values() for w in ws]
    if any(w != w.lower() for w in words):
        problems.append("some cleaned words still have capitals; lowercase them.")
    if any(not re.search(r"[a-z0-9]", w) for w in words):
        problems.append("some cleaned words are punctuation; drop tokens with no letter or digit.")
    for w in ("the", "and", "my", "is", "a"):
        if w in words:
            problems.append(f"'{w}' is still in the cleaned words; stop words should go (except negation).")
            break
    if "dropping" in words or "calls" in words:
        problems.append("'dropping' or 'calls' survived: lemmatize with the part of speech, so they become 'drop' and 'call'.")
    return _report(problems, "Right: 20 tickets cleaned, negation kept in every one that had it, "
                             "and verbs and plurals reduced to their lemmas.")


def check_01_03():
    """Entities from the five notices, found by spaCy on the raw text."""
    import spacy
    got, err = _read("01_03_entities.json")
    if err:
        return _report([err], "")
    nlp = spacy.load("en_core_web_sm")
    notices = _notices()
    if not isinstance(got, list) or len(got) != len(notices):
        return _report([f"out/01_03_entities.json should be a list with one entry per notice ({len(notices)})."], "")
    problems = []
    for i, (n, g) in enumerate(zip(notices, got)):
        want = {(e.text, e.label_) for e in nlp(n).ents}
        have = {tuple(x) for x in g}
        if have != want:
            extra = have - want
            hint = (" Some look like they came from cleaned or lowercased text; run spaCy on the notice "
                    "exactly as it is." if extra else "")
            problems.append(f"notice {i}: expected {len(want)} entities, found {len(have & want)} of them"
                            f" and {len(extra)} others.{hint}")
    return _report(problems, f"Right: {sum(len(g) for g in got)} entities across {len(got)} notices, "
                             "exactly what spaCy finds in the raw text.")


def check_on_your_own(sample=None):
    """Part 3: out/tickets_preprocessed.jsonl, one JSON object per ticket."""
    import csv
    import spacy
    p = os.path.join(OUT, "tickets_preprocessed.jsonl")
    if not os.path.exists(p):
        return _report(["out/tickets_preprocessed.jsonl does not exist yet. Run your last cell."], "")
    with open(os.path.join(DATA, "kittiwake_tickets.csv")) as f:
        rows = {r["ticket_id"]: r["text"] for r in csv.DictReader(f)}
    got = {}
    problems = []
    with open(p) as f:
        for n, line in enumerate(f, 1):
            try:
                o = json.loads(line)
                got[o["ticket_id"]] = o
            except (ValueError, KeyError, TypeError):
                problems.append(f"line {n} is not a JSON object with a ticket_id.")
                break
    if problems:
        return _report(problems, "")
    if set(got) != set(rows):
        return _report([f"the file has {len(got)} tickets; it needs all {len(rows)} from "
                        "data/kittiwake_tickets.csv, one per line."], "")
    for o in got.values():
        if not all(k in o for k in ("tokens", "lemmas", "entities")):
            return _report([f"{o['ticket_id']} is missing one of tokens, lemmas, entities."], "")
    ids = sorted(rows)[: sample or len(rows)]
    neg = [i for i in ids if NEG_RE.search(rows[i])]
    lost = [i for i in neg if "not" not in got[i]["lemmas"]]
    if len(lost) > 0.03 * len(neg):
        problems.append(f"{len(lost)} of {len(neg)} tickets with a negation lost it, for example {lost[0]}: "
                        f"{rows[lost[0]][:70]!r}. SMS tickets write cant and dont without the apostrophe.")
    bad_case = [i for i in ids if any(w != w.lower() for w in got[i]["lemmas"])]
    if bad_case:
        problems.append(f"{len(bad_case)} tickets have capitals in their lemmas, for example {bad_case[0]}.")
    punct = [i for i in ids if any(not re.search(r"[a-z0-9]", w) for w in got[i]["lemmas"])]
    if punct:
        problems.append(f"{len(punct)} tickets have punctuation in their lemmas, for example {punct[0]}.")
    stop = [i for i in ids if set(got[i]["lemmas"]) & {"the", "a", "an", "and", "is", "to", "of", "my"}]
    if stop:
        problems.append(f"{len(stop)} tickets still have stop words in their lemmas, for example {stop[0]}.")
    drop = [i for i in ids if re.search(r"\bdropp(ing|ed)\b|\bdrops\b", rows[i]) and "drop" not in got[i]["lemmas"]]
    if drop:
        problems.append(f"{len(drop)} tickets mention dropping but have no lemma 'drop', for example {drop[0]}.")
    nlp = spacy.load("en_core_web_sm")
    off = 0
    for i in ids:
        want = {(e.text, e.label_) for e in nlp(rows[i]).ents}
        have = {tuple(x) for x in got[i]["entities"]}
        if want != have and len(want & have) < 0.9 * max(len(want), 1):
            off += 1
    if off > 0.05 * len(ids):
        problems.append(f"{off} tickets have entities that differ from spaCy on the raw text. "
                        "Find entities before any cleaning.")
    return _report(problems, f"Right: {len(got)} tickets preprocessed; negation kept in "
                             f"{len(neg) - len(lost)} of {len(neg)}; entities match spaCy on the raw text.")
