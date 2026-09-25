"""nlpcheck: the feedback half of the Lab 11 notebooks.

Three kinds of help, used the same way in every lab of the course:

    ask("r1", "b")               answer a recall question from memory; you are
                                 told at once whether it is right, and why.
    guess("key", value)          commit to a prediction BEFORE the cell that
    reveal("key", actual)        shows the answer; reveal() compares the two.
    check_11_01()                check a saved exercise the way the checkpoint
                                 will, with a message that says what to fix.

Predictions are kept in out/predictions.json. Nothing is marked on them: they
are there because a guess you wrote down is one you remember being wrong
about, and the sign-off counts how many you made.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PRED = os.path.join(OUT, "predictions.json")


# ---------------------------------------------------------------- recall ---

QUESTIONS = {
    # 11_01, from the chapter pages and Lab 05
    "r1": ("b", "word2vec learns one vector per word, whatever the sentence. The \"charge\" on a bill and the "
                "\"charge\" of a battery get the same vector, which is the limitation BERT removes."),
    "r2": ("c", "Masked language modelling hides some tokens and trains the model to put them back from the "
                "words on both sides. The text labels itself, so no person has to."),
    # 11_02, from 11_01
    "r3": ("a", "Every token's vector from BERT depends on the whole sentence around it: that is what "
                "contextual means, and why the two senses of charge came out different."),
    "r4": ("b", "bert-mini has 4 layers and about 11 million weights; BERT Base has 12 layers and 110 million."),
    # 11_03, from 11_02
    "r5": ("c", "Fine-tuning adds a small head and trains every weight, the pretrained ones included, a little. "
                "Freezing BERT and training only the head scored far lower."),
    "r6": ("a", "About 1 ticket in 20 carries the wrong department, so even a perfect reader of the text scores "
                "0.921 macro-F1 on the 200. TF-IDF and the fine-tuned BERTs all land at that ceiling."),
    # exit tickets, from the book's assessment
    "x1": ("c", "BERT reads the whole input at once, and every layer lets each token attend to both sides: "
                "bidirectional."),
    "x2": ("a", "BERT Base stacks 12 encoder layers (BERT Large, 24)."),
    "x3": ("b", "Token, segment and position embeddings are summed for every input token; the position "
                "embedding is what tells the encoder where each token sits."),
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


def _report(problems, ok):
    if problems:
        print("Not yet:")
        for p in problems:
            print("  -", p)
        return False
    print(ok)
    return True


CHARGE = ["there is a charge on my bill i do not recognise.",
          "they added a late payment charge to my invoice.",
          "my phone will not charge overnight any more."]


def check_11_01():
    """The masked guesses and the two senses of charge, recomputed with bert-mini."""
    got, err = _read("11_01_results.json")
    if err:
        return _report([err], "")
    import bertlab
    problems = []
    fills = got.get("fills") or {}
    if "I love to learn [MASK] science." not in fills:
        problems.append("the book's sentence has no saved guesses: run section 2.")
    a, b, c = (bertlab.word_vector(s, "charge") for s in CHARGE)
    same, diff = bertlab.cosine(a, b), bertlab.cosine(a, c)
    if got.get("same_sense") is None or got.get("different_sense") is None:
        problems.append("same_sense or different_sense is still None: fill in section 5 and run it.")
    else:
        if abs(got["same_sense"] - same) > 0.01:
            problems.append(f"same_sense should be the cosine between the charge in sentences 0 and 1 "
                            f"({same}); you saved {got['same_sense']}.")
        if abs(got["different_sense"] - diff) > 0.01:
            problems.append(f"different_sense should be the cosine between the charge in sentences 0 and 2 "
                            f"({diff}); you saved {got['different_sense']}.")
    return _report(problems, f"Right: the two billing charges are {same} alike, a bill and a battery {diff}.")


PASS_TICKETS = 0.85


def check_11_02():
    """out/ticket_bert, loaded and scored on the 200 unlabelled tickets."""
    import bertlab
    p = os.path.join(OUT, "ticket_bert")
    if not os.path.exists(os.path.join(p, "config.json")):
        return _report(["out/ticket_bert does not exist yet: run the save cell in section 7."], "")
    try:
        model = bertlab.load_saved(p)
    except Exception as e:
        return _report([f"out/ticket_bert could not be loaded: {e}"], "")
    import pandas as pd
    answers = f"{bertlab.DATA}/answers/tickets_answers.csv"
    if not os.path.exists(answers):
        # Outside CourseLabs (Colab, a download) the held-out answers are not published.
        print("Your model is saved, but this check scores it against held-out answers that only "
              "a CourseLabs session has, so it cannot run here.")
        return None
    u = bertlab.unlabelled().merge(pd.read_csv(answers), on="ticket_id")
    f1 = bertlab.macro_f1(u.department, bertlab.predict(model, u.text))
    problems = []
    if f1 < PASS_TICKETS:
        problems.append(f"on the 200 unlabelled tickets your model scores macro-F1 {f1}; the bar is "
                        f"{PASS_TICKETS}. Did it train on all 600 tickets, with every weight unfrozen?")
    return _report(problems, f"Right: your fine-tuned BERT scores macro-F1 {f1} on the 200 tickets "
                             "it has never seen (the label-noise ceiling is 0.921).")


def check_11_03():
    """The reader: one right answer, and the Flex 30 question answered after retrieval."""
    got, err = _read("11_03_results.json")
    if err:
        return _report([err], "")
    problems = []
    if "14 october" not in str(got.get("mast", "")).lower():
        problems.append("the mast answer should be 14 October: rerun section 2.")
    if "32.50" in str(got.get("flex_by_prob", "")) or not got.get("flex_by_prob"):
        problems.append("flex_by_prob should be the (wrong) answer probability picks; rerun section 4 as shipped.")
    if got.get("flex_by_retrieval") is None:
        problems.append("flex_by_retrieval is still None: set best_notice to the highest TF-IDF score in section 5.")
    elif "32.50" not in got["flex_by_retrieval"]:
        problems.append(f"flex_by_retrieval is {got['flex_by_retrieval']!r}; reading the notice with the highest "
                        "TF-IDF score gives an answer containing $32.50. Use scores.argmax().")
    return _report(problems, f"Right: across all five notices the reader says {got.get('flex_by_prob')!r} by "
                             f"probability and {got.get('flex_by_logit')!r} by logit; retrieve first, and it "
                             f"says {got.get('flex_by_retrieval')!r}.")


# ------------------------------------------------------------- on your own ---

DESK = os.path.join(HERE, "desk_questions.json")
PASS_DESK = 12


def _norm(s):
    return " ".join(str(s).lower().replace("\u2019", "'").split())


def grade_desk(answers):
    """(right, rows): each answer against the facts the notices state."""
    import bertlab
    with open(DESK) as f:
        qs = json.load(f)
    text = _norm(" ".join(bertlab.notices()))
    rows, right = [], 0
    for q in qs:
        a = answers.get(q["question"])
        if a is None:
            rows.append((q["question"], None, "no answer saved"))
            continue
        a = _norm(a)
        if a and a not in text:
            ok, why = False, "not a span of any notice"
        elif q["facts"] is None:
            ok, why = a == "", "should be \"\" (the notices do not say)" if a else "abstained"
        elif not a:
            ok, why = False, "abstained, but the notices do answer it"
        elif len(a) > 80:
            ok, why = False, "too long to be an answer"
        else:
            ok = any(f in a for f in q["facts"])
            why = "right" if ok else "wrong span"
        right += ok
        rows.append((q["question"], a, why))
    return right, rows


def check_on_your_own():
    got, err = _read("kittiwake_answers.json")
    if err:
        return _report([err], "")
    right, rows = grade_desk(got)
    for q, a, why in rows:
        print(f"  {why:42s} {q}  ->  {a!r}")
    problems = []
    if right < PASS_DESK:
        problems.append(f"{right} of {len(rows)} are right; the brief asks for at least {PASS_DESK}.")
    return _report(problems, f"Right: {right} of {len(rows)} questions answered or declined correctly.")
