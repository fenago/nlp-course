"""nlpcheck: the feedback half of the Lab 02 notebooks.

Three kinds of help, used the same way in every lab of the course:

    ask("r1", "b")               answer a recall question from memory; you are
                                 told at once whether it is right, and why.
    guess("naive", 7)            commit to a prediction BEFORE the cell that
    reveal("naive", actual)      shows the answer; reveal() compares the two.
    check_02_01()                check a saved exercise the way the checkpoint
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
    # 02_01, spaced retrieval from Lab 01
    "r1": ("b", "NLTK's stop list contains 'not', so removing stop words blindly turns 'not working' "
                "into 'working'. Counting words after that counts the wrong thing."),
    "r2": ("a", "A lemmatizer looks the word up and needs its part of speech; a stemmer only cuts "
                "suffixes by rule, so it can merge 'university' and 'universe'."),
    # 02_02, from 02_01 and the chapter
    "r3": ("11", "Eleven distinct words across the three reviews: this, movie, is, very, scary, and, long, "
                 "not, slow, spooky, good. Each review becomes a vector of 11 counts."),
    "r4": ("c", "A bag of words keeps counts and throws order away, so 'dog bites man' and 'man bites dog' "
                "are the same vector."),
    # 02_03, from 02_02
    "r5": ("a", "IDF is high for a word that appears in few documents and low for one that appears in "
                "many, so it rewards the words that tell documents apart."),
    "r6": ("1", "scikit-learn adds 1 to every IDF, so a word in every document gets an IDF of 1, not 0, "
                "and keeps a small weight instead of vanishing."),
    # exit tickets
    "x1": ("b", "Most entries of a document-term matrix are zero, because each document uses a tiny "
                "fraction of the vocabulary. That is what 'sparse' means."),
    "x2": ("a", "TF-IDF is high when a word is frequent in this document and rare in the others."),
    "x3": ("c", "Cosine similarity compares directions and ignores length, so a short review and a long "
                "one about the same thing still score as similar. It cannot see synonyms: 'spooky' and "
                "'scary' share no column."),
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

REVIEWS3 = ["This movie is very scary and long",
            "This movie is not scary and is slow",
            "This movie is spooky and good"]


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


def _sms():
    with open(os.path.join(DATA, "sms_spam", "SMSSpamCollection"), encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t", 1)[1] for line in f if "\t" in line]


def _reviews():
    import csv
    with open(os.path.join(DATA, "kittiwake_reviews.csv")) as f:
        return list(csv.DictReader(f))


def _zeros(m):
    return 100.0 * (1 - m.nnz / (m.shape[0] * m.shape[1]))


def check_02_01():
    """The SMS corpus: vocabulary sizes and the share of zeros, unigrams and bigrams."""
    from sklearn.feature_extraction.text import CountVectorizer
    got, err = _read("02_01_bow.json")
    if err:
        return _report([err], "")
    sms = _sms()
    u = CountVectorizer().fit_transform(sms)
    b = CountVectorizer(ngram_range=(1, 2)).fit_transform(sms)
    want = {"vocab_unigram": u.shape[1], "vocab_bigram": b.shape[1],
            "zeros_unigram": _zeros(u), "zeros_bigram": _zeros(b)}
    problems = []
    for k in ("vocab_unigram", "vocab_bigram"):
        if got.get(k) != want[k]:
            problems.append(f"'{k}' should be {want[k]} (the number of columns CountVectorizer makes); you have {got.get(k)}.")
    for k in ("zeros_unigram", "zeros_bigram"):
        v = got.get(k)
        if not isinstance(v, (int, float)) or abs(v - want[k]) > 0.05:
            hint = " It is a percentage: multiply the fraction by 100." if isinstance(v, (int, float)) and v <= 1 else ""
            problems.append(f"'{k}' should be {want[k]:.2f} percent; you have {v}.{hint}")
    return _report(problems, f"Right: {want['vocab_unigram']} words become {want['vocab_bigram']} columns with bigrams, "
                             f"and {want['zeros_unigram']:.2f} percent of the unigram matrix is zeros.")


def check_02_02():
    """Review 2 of the three horror reviews, TF-IDF by hand, matching scikit-learn."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    got, err = _read("02_02_tfidf.json")
    if err:
        return _report([err], "")
    tv = TfidfVectorizer()
    row = tv.fit_transform(REVIEWS3)[1].toarray().ravel()
    names = tv.get_feature_names_out()
    want = {w: float(v) for w, v in zip(names, row) if v > 0}
    problems = []
    if not isinstance(got, dict) or set(got) != set(want):
        have = set(got) if isinstance(got, dict) else set()
        return _report([f"the file should map each of the {len(want)} distinct words of review 2 "
                        f"({', '.join(sorted(want))}) to its TF-IDF; you have {sorted(have)}."], "")
    for w in sorted(want):
        v = got[w]
        if not isinstance(v, (int, float)) or abs(v - want[w]) > 1e-3:
            problems.append(f"'{w}' should be {want[w]:.3f}; you have {v}. Check the IDF formula "
                            "ln((1 + n) / (1 + df)) + 1, then the division by the vector's length.")
    return _report(problems, "Right: every value matches scikit-learn's to three places, so you know exactly "
                             "what TfidfVectorizer computes.")


def check_02_03():
    """The three reviews closest to 'no signal at home', by TF-IDF cosine similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    got, err = _read("02_03_search.json")
    if err:
        return _report([err], "")
    rows = _reviews()
    tv = TfidfVectorizer()
    m = tv.fit_transform([r["text"] for r in rows])
    sims = cosine_similarity(tv.transform(["no signal at home"]), m).ravel()
    by_id = {r["review_id"]: s for r, s in zip(rows, sims)}
    best = sorted(sims, reverse=True)[:3]
    top = got.get("top3") if isinstance(got, dict) else None
    if not isinstance(top, list) or len(top) != 3 or any(t not in by_id for t in top):
        return _report(["'top3' should be a list of three review ids, like 'R512'."], "")
    have = sorted((by_id[t] for t in top), reverse=True)
    if any(abs(a - b) > 1e-9 for a, b in zip(have, best)):
        return _report([f"those are not the three most similar reviews: their scores are "
                        f"{[round(float(x), 3) for x in have]}, and the best three score {[round(float(x), 3) for x in best]}. "
                        "Fit the vectorizer on the reviews, transform the query with the same vectorizer, "
                        "and sort by cosine similarity, highest first."], "")
    return _report([], f"Right: the top three score {[round(float(x), 3) for x in best]}.")


# Stems that mark praise and complaint in the Kittiwake reviews. A learner's
# lists pass on the evidence of two or more, whatever tokenizer, stop list or
# weighting produced them.
GOOD = ("excel", "friend", "reliab", "great", "cheap", "honest", "easy", "simpl", "fast", "valu", "fortun", "saves")
BAD = ("drop", "roam", "crash", "overcharg", "refund", "fail", "doubl", "lie", "pain")


# The four Part 3 queries, and the stems a relevant review must contain. The
# last query shares no useful word with the reviews it should find, so only a
# method that matches meaning finds them; "charg" is left out of its stems on
# purpose, because "no surprise charges" is praise.
QUERIES = {"dropped calls": ("drop", "call"), "roaming charges": ("roam", "charg"),
           "friendly support": ("friendly", "support"),
           "the phone company took too much money": ("overcharg", "refund", "doubl", "bill", "fee")}


def _hits(terms, stems):
    # Any word of the term starting with a stem counts: "dropped calls" and
    # "drop" both hit "drop", and "reliable" does not hit "lie".
    return [t for t in terms if any(w.startswith(s) for w in str(t).lower().split() for s in stems)]


def check_on_your_own():
    """Part 3: out/review_insights.json, distinctive terms per star rating and three searches."""
    got, err = _read("review_insights.json")
    if err:
        return _report([err], "")
    problems = []
    dist = got.get("distinctive") if isinstance(got, dict) else None
    if not isinstance(dist, dict) or set(map(str, dist)) != {"1", "2", "3", "4", "5"}:
        return _report(["'distinctive' should map each star rating, \"1\" to \"5\", to a list of terms."], "")
    dist = {str(k): v for k, v in dist.items()}
    for k, v in dist.items():
        if not isinstance(v, list) or len(v) < 5 or not all(isinstance(t, str) for t in v):
            problems.append(f"the list for {k} stars should hold at least five terms (ten is the brief).")
    if not problems:
        one, five = [t.lower() for t in dist["1"][:10]], [t.lower() for t in dist["5"][:10]]
        if len(_hits(one, BAD)) < 2 or len(_hits(one, GOOD)) > 1:
            problems.append(f"the one-star terms {one} do not look like complaints. Compare each rating's reviews "
                            "against the others, not against themselves, and leave out stop words.")
        if len(_hits(five, GOOD)) < 2 or len(_hits(five, BAD)) > 1:
            problems.append(f"the five-star terms {five} do not look like praise. Compare each rating against "
                            "the others.")
        if len(set(one) & set(five)) > 3:
            problems.append("the one-star and five-star lists share more than three terms, so they are not "
                            "distinctive. Weight by how rare a term is in the other ratings.")
    rows = {r["review_id"]: r["text"].lower() for r in _reviews()}
    search = got.get("search") if isinstance(got, dict) else None
    want = QUERIES
    if not isinstance(search, dict):
        problems.append("'search' should map each of the four queries to a list of three review ids.")
    else:
        for q, stems in want.items():
            ids = search.get(q)
            if not isinstance(ids, list) or len(ids) != 3 or any(i not in rows for i in ids):
                problems.append(f"search['{q}'] should be a list of three review ids from the reviews file.")
                continue
            miss = [i for i in ids if not any(s in rows[i] for s in stems)]
            if q.startswith("the phone company") and len(miss) > 1:
                problems.append(f"for '{q}', {len(miss)} of your three reviews are not about being overcharged, "
                                f"for example {miss[0]} ({rows[miss[0]][:60]!r}). This query shares no useful word "
                                "with the reviews it should find: a method that matches words cannot find them.")
            elif not q.startswith("the phone company") and miss:
                problems.append(f"for '{q}', review {miss[0]} ({rows[miss[0]][:60]!r}) does not mention it at all.")
    return _report(problems, f"Right: one-star reviewers talk about {', '.join(dist['1'][:5])}; five-star "
                             f"reviewers about {', '.join(dist['5'][:5])}; all four searches return relevant reviews.")
