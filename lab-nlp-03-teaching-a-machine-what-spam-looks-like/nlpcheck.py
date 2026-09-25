"""nlpcheck: the feedback half of the Lab 03 notebooks.

Three kinds of help, used the same way in every lab of the course:

    ask("r1", "b")               answer a recall question from memory; you are
                                 told at once whether it is right, and why.
    guess("naive", 7)            commit to a prediction BEFORE the cell that
    reveal("naive", actual)      shows the answer; reveal() compares the two.
    check_03_01()                check a saved exercise the way the checkpoint
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
    # 03_01: spaced retrieval from Labs 01 and 02
    "r1": ("b", "NLTK's English stop list contains 'not', so removing stop words blindly turns "
                "'not working' into 'working'. Lab 01 kept negation with a KEEP set."),
    "r2": ("a", "A word in every document has the lowest inverse document frequency, so TF-IDF gives it "
                "the least weight. Common words shout; TF-IDF turns them down."),
    "r3": ("c", "A bag of words keeps counts and throws away order, which is why 'dog bites man' and "
                "'man bites dog' are the same vector."),
    # 03_02, from 03_01 and the chapter
    "r4": ("0", "A filter that never flags anything catches none of the spam: recall 0, whatever its "
                "accuracy says."),
    "r5": ("b", "Precision asks: of the messages I flagged, how many really were spam? TP / (TP + FP)."),
    # 03_03, from 03_02
    "r6": ("a", "Lowering the threshold flags more messages: more spam caught (recall up), and more ham "
                "caught with it (precision down)."),
    "r7": ("c", "The model and its cleaning must travel together. Putting clean() inside the pipeline "
                "means whatever calls the model gets the same preprocessing it was trained with."),
    # exit tickets
    "x1": ("b", "Recall = TP / (TP + FN) = 50 / (50 + 200) = 0.2: the model found one cancer patient in five."),
    "x2": ("a", "AUC is the chance that a randomly chosen spam scores higher than a randomly chosen ham. "
                "It does not depend on any one threshold."),
    "x3": ("c", "The support vectors are the training points that sit on the edge of the margin. Move any "
                "other point (without crossing the margin) and the boundary does not change."),
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

def _report(problems, ok):
    if problems:
        print("Not yet:")
        for p in problems:
            print("  -", p)
        return False
    print(ok)
    return True


def _held_out():
    """The checkpoint's view of the data: the original file, the fixed split."""
    import sys
    sys.path.insert(0, HERE)
    import spamtools
    df = spamtools.load_sms(os.path.join(os.environ.get("NLPLAB_DATA", "/opt/nlplab/data"), "sms_spam", "SMSSpamCollection"))
    return df, spamtools.split(df)


def _json(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None, f"out/{name} does not exist yet. Run the cell that saves it."
    try:
        with open(p) as f:
            return json.load(f), None
    except ValueError as e:
        return None, f"out/{name} is not valid JSON ({e}). Run the saving cell again."


def check_03_01():
    """The corpus read correctly, and the never-flag baseline measured."""
    got, err = _json("03_01_baseline.json")
    if err:
        return _report([err], "")
    df, (Xtr, Xte, ytr, yte) = _held_out()
    problems = []
    if got.get("n_messages") != len(df):
        hint = (" Two messages were swallowed by the quote on the line above them: read the file with "
                "quoting=csv.QUOTE_NONE, as spamtools.load_sms() does." if got.get("n_messages") == len(df) - 2 else "")
        problems.append(f"n_messages should be {len(df)}; you have {got.get('n_messages')}.{hint}")
    share = (df["label"] == "spam").mean()
    if not isinstance(got.get("spam_share"), (int, float)) or abs(got["spam_share"] - share) > 0.001:
        problems.append(f"spam_share should be the fraction of messages labelled spam, {share:.3f}.")
    acc = (yte == "ham").mean()
    if not isinstance(got.get("dummy_accuracy"), (int, float)) or abs(got["dummy_accuracy"] - acc) > 0.001:
        problems.append(f"dummy_accuracy should be the test accuracy of DummyClassifier(strategy='most_frequent'), {acc:.3f}.")
    if got.get("dummy_spam_recall") != 0:
        problems.append("dummy_spam_recall should be the recall of that dummy on spam, which is 0.")
    return _report(problems, f"Right: {len(df)} messages, {share:.1%} spam, and a filter that flags nothing "
                             f"scores {acc:.1%} accuracy with a spam recall of 0.")


def _model(name):
    import joblib
    import sys
    sys.path.insert(0, HERE)
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None, f"out/{name} does not exist yet. Run the cell that saves it."
    try:
        return joblib.load(p), None
    except Exception as e:  # noqa: BLE001
        return None, (f"out/{name} will not load outside the notebook ({type(e).__name__}: {e}). "
                      "A pipeline may only use functions from a file, such as spamtools.clean.")


def _scores(model, X, y):
    from sklearn import metrics
    p = model.predict(X)
    return (metrics.accuracy_score(y, p), metrics.precision_score(y, p, pos_label="spam", zero_division=0),
            metrics.recall_score(y, p, pos_label="spam"), metrics.f1_score(y, p, pos_label="spam"), p)


def check_03_02():
    """The logistic regression pipeline: good, served the way it was tested, and a threshold."""
    import pandas as pd
    from sklearn import metrics
    model, err = _model("spam_lr.joblib")
    if err:
        return _report([err], "")
    df, (Xtr, Xte, ytr, yte) = _held_out()
    try:
        acc, prec, rec, f1, p = _scores(model, Xte, yte)
    except Exception as e:  # noqa: BLE001
        return _report([f"the saved model cannot predict ({type(e).__name__}); fit it and save it again."], "")
    problems = []
    if acc < 0.96 or prec < 0.95:
        problems.append(f"on the held-out messages the model scores accuracy {acc:.3f} and spam precision "
                        f"{prec:.3f}; a TF-IDF and logistic regression pipeline should reach 0.96 and 0.95.")
    sp = os.path.join(OUT, "03_02_served.csv")
    if not os.path.exists(sp):
        problems.append("out/03_02_served.csv does not exist yet. Run the serving cell in section 5.")
    else:
        served = pd.read_csv(sp).set_index("message_index")["served"]
        want = pd.Series(p, index=Xte.index)
        common = served.index.intersection(want.index)
        agree = (served.loc[common] == want.loc[common]).mean() if len(common) else 0
        if len(common) < len(want):
            problems.append(f"out/03_02_served.csv has {len(common)} of the {len(want)} test messages.")
        elif agree < 0.99:
            problems.append(f"serve() agrees with the model's own test predictions on only {agree:.1%} of "
                            "messages: it cleans the text before a model that was trained on raw text. "
                            "Put clean() inside the pipeline and let serve() pass the raw message.")
    got, err = _json("03_02_threshold.json")
    if err:
        problems.append(err)
    else:
        t = got.get("threshold")
        if not isinstance(t, (int, float)) or not 0 < t < 0.5:
            problems.append(f"threshold should be a number between 0 and 0.5 chosen from the cross-validated "
                            f"probabilities; you have {t!r}.")
        else:
            i = list(model.classes_).index("spam")
            q = model.predict_proba(Xte)[:, i] >= t
            r = metrics.recall_score(yte == "spam", q)
            if r < 0.88:
                problems.append(f"at threshold {t} the held-out spam recall is {r:.3f}; the brief asked for "
                                "about 0.90. Choose the threshold on the cross-validated probabilities.")
    return _report(problems, f"Right: accuracy {acc:.3f}, spam precision {prec:.3f} and recall {rec:.3f} at 0.5; "
                             "serve() and the test agree; and the lower threshold keeps the recall promise.")


def check_03_03():
    """The support vector machine that works."""
    model, err = _model("spam_svm.joblib")
    if err:
        return _report([err], "")
    df, (Xtr, Xte, ytr, yte) = _held_out()
    try:
        acc, prec, rec, f1, p = _scores(model, Xte, yte)
    except Exception:  # noqa: BLE001
        return _report(["the saved SVM has not been fitted: fill in the fit line in section 3 and run the "
                        "cells again."], "")
    if (p == "spam").sum() == 0:
        return _report(["the saved SVM still calls every held-out message ham. That is gamma='auto' on "
                        "thousands of TF-IDF columns; use LinearSVC() or leave gamma at its default."], "")
    if f1 < 0.92:
        return _report([f"the saved SVM scores spam F1 {f1:.3f} on the held-out messages; LinearSVC reaches "
                        "about 0.95. Check it is fitted on the training split as a TF-IDF pipeline."], "")
    return _report([], f"Right: the SVM flags {(p == 'spam').sum()} held-out messages, spam F1 {f1:.3f}, "
                       f"precision {prec:.3f}, recall {rec:.3f}.")


DEPTS = {"billing", "network", "device", "account"}

# The labelled tickets whose department is the one triage forwarded them to, not
# the one their text describes: about one in twenty, as in a real inbox. Known
# because the pack is generated (images/nlplab/generators/make_kittiwake.py,
# MISROUTE); a real dataset would not tell you, which is why 03_04 asks you to
# read the errors and judge them first.
MISROUTED = {
    "T10002", "T10019", "T10028", "T10055", "T10068", "T10080", "T10093", "T10135", "T10151", "T10155",
    "T10177", "T10193", "T10206", "T10215", "T10227", "T10233", "T10253", "T10376", "T10384", "T10418",
    "T10428", "T10431", "T10467", "T10477", "T10489", "T10498", "T10503", "T10528", "T10565", "T10587",
    "T10597",
}


def misrouted_among(ticket_ids):
    """How many of these labelled tickets carry a misrouted label."""
    return len(set(ticket_ids) & MISROUTED)


def check_on_your_own():
    """Part 3: the format of the router's output. The checkpoint scores it."""
    import pandas as pd
    problems = []
    p = os.path.join(OUT, "ticket_predictions.csv")
    if not os.path.exists(p):
        return _report(["out/ticket_predictions.csv does not exist yet."], "")
    got = pd.read_csv(p)
    want = pd.read_csv(os.path.join(DATA, "kittiwake_tickets_unlabeled.csv"))
    if list(got.columns) != ["ticket_id", "department"]:
        problems.append(f"the columns should be ticket_id,department; you have {','.join(got.columns)}.")
    elif set(got["ticket_id"]) != set(want["ticket_id"]) or len(got) != len(want):
        problems.append(f"it should have one row for each of the {len(want)} unlabelled tickets; you have {len(got)}.")
    elif not set(got["department"]) <= DEPTS:
        problems.append(f"departments must be one of {sorted(DEPTS)}; found {sorted(set(got['department']) - DEPTS)}.")
    rep, err = _json("router_report.json")
    if err:
        problems.append(err)
    elif not isinstance(rep.get("cv_macro_f1"), (int, float)) or not 0 <= rep["cv_macro_f1"] <= 1 or not rep.get("model"):
        problems.append("router_report.json needs 'model' (a short description) and 'cv_macro_f1' (a number from 0 to 1).")
    return _report(problems, f"Right format: {len(got)} predictions, and your cross-validated macro F1 is "
                             f"{rep['cv_macro_f1']:.3f}. The checkpoint scores the predictions against "
                             "labels this notebook has never seen." if not problems else "")
