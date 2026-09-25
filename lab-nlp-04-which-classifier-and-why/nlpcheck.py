"""nlpcheck: the feedback half of the Lab 04 notebooks.

Three kinds of help, used the same way in every lab of the course:

    ask("r1", "b")               answer a recall question from memory; you are
                                 told at once whether it is right, and why.
    guess("k1_train", 0.9)       commit to a prediction BEFORE the cell that
    reveal("k1_train", actual)   shows the answer; reveal() compares the two.
    check_04_01()                check a saved exercise the way the checkpoint
                                 will, with a message that says what to fix.

Predictions are kept in out/predictions.json. Nothing is marked on them: they
are there because a guess you wrote down is one you remember being wrong
about, and the sign-off counts how many you made.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "data")
PRED = os.path.join(OUT, "predictions.json")
STARS = os.path.join(os.environ.get("NLPLAB_DATA", "/opt/nlplab/data"), "kittiwake_reviews.csv")
K_GRID = [1, 5, 15, 25, 51, 101]
BAR = 0.92


# ---------------------------------------------------------------- recall ---

QUESTIONS = {
    # 04_01, from Lab 03
    "r1": ("a", "Recall is the share of the real positives the model caught: of all the spam, how much "
                "it flagged. Precision is the other way round: of what it flagged, how much was spam."),
    "r2": ("b", "Inside a Pipeline the vectorizer is fitted on the training data only, and the exact same "
                "transformation is applied when the model is used. That was Lab 03's train/serve fix."),
    # 04_02, from 04_01
    "r3": ("b", "With K = 1 every training sentence is its own nearest neighbour, so training accuracy is "
                "perfect and says nothing about new sentences. That is what overfitting looks like."),
    "r4": ("c", "KNN does its work when it predicts: every new sentence is compared with every stored "
                "one. Training only stores them, which is why it is fast to train and slow to use."),
    # 04_03, from 04_02
    "r5": ("a", "Naive Bayes multiplies one probability per word as if the words were independent, so "
                "'not' and 'bad' both count as negative evidence, twice over."),
    "r6": ("c", "Laplace smoothing adds 1 to every count, so a word never seen in a class gets a small "
                "probability instead of zero, and one unseen word cannot wipe out the whole product."),
    # exit tickets, from the book's Chapter 4 assessment
    "x1": ("c", "Hamming distance counts mismatches, which is what 'distance' means for categories "
                "such as colour or plan name, where a difference has no size."),
    "x2": ("a", "A larger K averages over more neighbours, so the model gets smoother and simpler: "
                "bias goes up and variance goes down."),
    "x3": ("c", "It assumes every feature is independent of every other given the class. That is "
                "almost never true of words, and it works surprisingly well anyway."),
    "x4": ("a", "True: KNN stores the training data and does all its computation when asked to predict."),
    "x5": ("a", "Supervised: it learns from examples whose labels are known."),
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


def _same(g, a):
    try:
        return abs(float(g) - float(a)) <= 0.02 * max(1.0, abs(float(a)))
    except (TypeError, ValueError):
        return str(g).strip().lower() == str(a).strip().lower()


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


def knn_search():
    """The grid search the 04_01 exercise asks for, done the reference way."""
    import clftools
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    X_train, X_test, y_train, y_test = clftools.split(clftools.load_sentences())
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("knn", KNeighborsClassifier(metric="cosine"))])
    gs = GridSearchCV(pipe, {"knn__n_neighbors": K_GRID}, cv=5).fit(X_train, y_train)
    return gs.best_params_["knn__n_neighbors"], gs.best_score_, gs.score(X_test, y_test)


def check_04_01():
    """K chosen by 5-fold cross-validation on the training set, for TF-IDF KNN."""
    got, err = _read("04_01_knn.json")
    if err:
        return _report([err], "")
    k, cv, test = knn_search()
    problems = []
    if got.get("best_k") != k:
        problems.append(f"'best_k' should be the K that GridSearchCV picks from {K_GRID} with cv=5; "
                        f"you have {got.get('best_k')}.")
    if not isinstance(got.get("cv_accuracy"), (int, float)) or abs(got["cv_accuracy"] - cv) > 0.001:
        problems.append("'cv_accuracy' should be the search's best_score_, the mean accuracy over the "
                        "five folds of the training set.")
    if not isinstance(got.get("test_accuracy"), (int, float)) or abs(got["test_accuracy"] - test) > 0.001:
        problems.append("'test_accuracy' should be the chosen model's accuracy on X_test, which the "
                        "search never saw.")
    return _report(problems, f"Right: cross-validation picks K = {k} (mean fold accuracy {cv:.3f}), "
                             f"and on the test set it scores {test:.3f}.")


BOOK = [("This is my book", "stmt"), ("They are novels", "stmt"), ("have you read this book", "question"),
        ("who is the author", "question"), ("what are the characters", "question"),
        ("This is how I bought the book", "stmt"), ("I like fictions", "stmt"),
        ("what is your favorite book", "question")]
NEW = "what is the price of the book"


def bayes_reference():
    """The 04_02 numbers, the reference way: sums, the hand posterior, and scikit-learn's."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    texts = [t for t, _ in BOOK]
    labels = [c for _, c in BOOK]
    d = len(CountVectorizer().fit(texts).vocabulary_)
    scores = {}
    for c in ("stmt", "question"):
        docs = [t for t, l in BOOK if l == c]
        v = CountVectorizer().fit(docs)
        counts = dict(zip(v.get_feature_names_out(), v.transform(docs).toarray().sum(0)))
        total = sum(counts.values())
        p = labels.count(c) / len(labels)
        for w in NEW.split():
            p *= (counts.get(w, 0) + 1) / (total + d)
        scores[c] = p
    hand = scores["question"] / sum(scores.values())
    cv = CountVectorizer()
    nb = MultinomialNB(alpha=1.0).fit(cv.fit_transform(texts), labels)
    sk = dict(zip(nb.classes_, nb.predict_proba(cv.transform([NEW]))[0]))["question"]
    return hand, sk


def check_04_02():
    """The fixed word probabilities, the hand posterior and scikit-learn's."""
    got, err = _read("04_02_bayes.json")
    if err:
        return _report([err], "")
    hand, sk = bayes_reference()
    problems = []
    for key in ("prob_sum_stmt", "prob_sum_question"):
        v = got.get(key)
        if not isinstance(v, (int, float)):
            problems.append(f"'{key}' is missing.")
        elif abs(v - 1) > 1e-6:
            problems.append(f"'{key}' is {v:.3f}, not 1: the word probabilities still divide by the number "
                            "of distinct words. Divide by the class's total word count (section 3).")
    v = got.get("posterior_question_by_hand")
    if not isinstance(v, (int, float)) or abs(v - hand) > 0.001:
        problems.append(f"'posterior_question_by_hand' should be {hand:.3f}: the smoothed product for "
                        "'question' divided by the sum of both products (section 4).")
    v = got.get("posterior_question_sklearn")
    if not isinstance(v, (int, float)) or abs(v - sk) > 0.001:
        problems.append(f"'posterior_question_sklearn' should be MultinomialNB's probability of 'question', {sk:.3f}.")
    return _report(problems, f"Right: both sets of word probabilities sum to 1; by hand P(question) = {hand:.3f}, "
                             f"and scikit-learn says {sk:.3f}.")


NAMES = ("naive_bayes", "knn_tfidf", "logreg_tfidf", "knn_embed", "logreg_embed")


def check_04_03():
    """Five classifiers on one split, and a choice for each of three jobs."""
    import clftools
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    got, err = _read("04_03_choice.json")
    if err:
        return _report([err], "")
    res = got.get("results") or {}
    problems = []
    missing = [n for n in NAMES if n not in res or not isinstance(res[n].get("accuracy"), (int, float))]
    if missing:
        return _report([f"'results' needs an accuracy for each of {', '.join(NAMES)}; missing {', '.join(missing)}."], "")
    X_train, X_test, y_train, y_test = clftools.split(clftools.load_sentences())
    ref = {"naive_bayes": make_pipeline(CountVectorizer(), MultinomialNB()),
           "logreg_tfidf": make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=2000))}
    for n, m in ref.items():
        a = m.fit(X_train, y_train).score(X_test, y_test)
        if abs(res[n]["accuracy"] - a) > 0.005:
            problems.append(f"{n}: you have {res[n]['accuracy']:.3f}; on this split it scores {a:.3f}. "
                            "Use clftools.split and the models from section 2.")
    for n in NAMES:
        if not 0.5 <= res[n]["accuracy"] <= 1:
            problems.append(f"{n}: an accuracy of {res[n]['accuracy']} is not a result this model can get.")
    ch = got.get("choices") or {}
    best = max(NAMES, key=lambda n: res[n]["accuracy"])
    if ch.get("cheapest") not in ("naive_bayes", "logreg_tfidf"):
        problems.append("'cheapest': which model could classify ten million messages a day on one small "
                        "machine? Look at the timing columns, and remember the embedding models' columns "
                        "leave out the embedding itself.")
    if ch.get("most_accurate") != best:
        problems.append(f"'most_accurate' should name the model with the highest accuracy in your own results ({best}).")
    if ch.get("no_retraining") not in ("knn_tfidf", "knn_embed"):
        problems.append("'no_retraining': which kind of model learns a new example the moment you store it?")
    return _report(problems, f"Right: five models measured on one split; {best} is the most accurate, "
                             f"{ch['cheapest']} is cheap enough for ten million a day, and nearest neighbours "
                             "learn a new example the moment it is stored.")


def _stars():
    with open(STARS) as f:
        return {r["review_id"]: int(r["stars"]) for r in csv.DictReader(f)}


def check_on_your_own():
    """Part 3: out/review_sentiment.csv, a label for every unlabelled Kittiwake review."""
    from sklearn.metrics import f1_score
    p = os.path.join(OUT, "review_sentiment.csv")
    if not os.path.exists(p):
        return _report(["out/review_sentiment.csv does not exist yet. Run your last cell."], "")
    with open(os.path.join(DATA, "kittiwake_reviews_unlabelled.csv")) as f:
        want = [r["review_id"] for r in csv.DictReader(f)]
    try:
        with open(p) as f:
            got = {r["review_id"]: r["label"] for r in csv.DictReader(f)}
    except (KeyError, csv.Error):
        return _report(["out/review_sentiment.csv needs the columns review_id and label."], "")
    problems = []
    if set(got) != set(want):
        problems.append(f"the file has {len(got)} reviews; it needs all {len(want)} from "
                        "data/kittiwake_reviews_unlabelled.csv, once each.")
    bad = [k for k, v in got.items() if str(v).strip() not in ("0", "1")]
    if bad:
        problems.append(f"every label must be 1 (positive) or 0 (negative); {bad[0]} has {got[bad[0]]!r}.")
    if problems:
        return _report(problems, "")
    stars = _stars()
    ids = [i for i in want if stars[i] != 3]
    y = [1 if stars[i] >= 4 else 0 for i in ids]
    yp = [int(got[i]) for i in ids]
    f1 = f1_score(y, yp, average="macro")
    if f1 < BAR:
        return _report([f"macro-F1 on the {len(ids)} reviews with a clear rating is {f1:.3f}; the brief asks "
                        f"for {BAR}. What does your model know about Kittiwake's vocabulary, and how much "
                        "does it listen to your 20 labelled reviews?"], "")
    return _report([], f"Right: macro-F1 {f1:.3f} on {len(ids)} Kittiwake reviews the model never saw "
                       f"labelled, against a bar of {BAR}.")
