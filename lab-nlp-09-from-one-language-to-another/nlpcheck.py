"""nlpcheck: the feedback half of the Lab 09 notebooks.

Three kinds of help, used the same way in every lab of the course:

    ask("r1", "b")               answer a recall question from memory; you are
                                 told at once whether it is right, and why.
    guess("key", value)          commit to a prediction BEFORE the cell that
    reveal("key", actual)        shows the answer; reveal() compares the two.
    check_09_01()                check a saved exercise the way the checkpoint
                                 will, with a message that says what to fix.

Predictions are kept in out/predictions.json. Nothing is marked on them: they
are there because a guess you wrote down is one you remember being wrong
about, and the sign-off counts how many you made.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PRED = os.path.join(OUT, "predictions.json")


# ---------------------------------------------------------------- recall ---

QUESTIONS = {
    # 09_01, from Lab 08 and the chapter pages
    "r1": ("b", "An LSTM carries a separate cell state that its gates add to and erase from, so the gradient "
                "can travel back along it without being multiplied down at every step."),
    "r2": ("c", "Translation is many to many, and not in step: a sentence in, a sentence of a different "
                "length out, with the words in a different order."),
    # 09_02, from 09_01
    "r3": ("a", "With teacher forcing the decoder's input at each step is the true previous word of the "
                "target, whatever it guessed; its guess is only scored."),
    "r4": ("b", "An untrained decoder spreads its probability about evenly over the vocabulary, so the "
                "cross-entropy is about ln(8,004), which is 8.99."),
    # 09_03, from 09_02
    "r5": ("a", "A beam of width 1 keeps only the single likeliest prefix at each step, which is greedy decoding."),
    "r6": ("c", "Every extra word multiplies in another probability below 1, so an uncorrected score "
                "always prefers the shorter sentence. Dividing by length ** alpha corrects it."),
    # exit tickets
    "x1": ("a", "The context vector is the encoder's final hidden state; the encoder's outputs at each "
                "step are discarded, and the decoder starts from that state."),
    "x2": ("b", "A three-token sentence has no 4-grams, so BLEU's 4-gram precision is 0 and so is the "
                "geometric mean. That is why BLEU is computed over a whole corpus."),
    "x3": ("c", "The encoder has to fit both sentences into the same 256 numbers it uses for one, and it "
                "never saw a source that long in training. Attention removes the first problem."),
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


def _close(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def check_09_01():
    """Teacher forcing: the shift, the untrained loss, and the race with and without the teacher."""
    import math
    got, err = _read("09_01_results.json")
    if err:
        return _report([err], "")
    problems = []
    tgt, tin, tout = got.get("tgt"), got.get("tgt_in"), got.get("tgt_out")
    if tin is None or tout is None:
        problems.append("tgt_in or tgt_out is None: section 4 has not been filled in and run.")
    elif tin != [r[:-1] for r in tgt] or tout != [r[1:] for r in tgt]:
        problems.append("tgt_in should be every target without its last token (tgt[:, :-1]) and tgt_out every "
                        "target without its first, <sos> (tgt[:, 1:]). Check which end you cut.")
    ul = got.get("untrained_loss")
    if ul is None or not abs(ul - math.log(8004)) < 0.3:
        problems.append(f"the untrained loss should be close to ln(8,004) = 8.99; got {ul}. Rerun section 4 "
                        "with the model as built, before any training.")
    tf, no = got.get("tf_chrf"), got.get("no_tf_chrf")
    if tf is None or no is None:
        problems.append("the two training runs in section 5 have not both been run and scored.")
    elif not tf > no + 2:
        problems.append(f"the teacher-forced model (chrF {tf:.1f}) should clearly beat the one trained on its "
                        f"own guesses ({no:.1f}); rerun section 5 as shipped.")
    return _report(problems, f"Right: the shift is correct, the untrained loss is {ul:.2f} (ln 8,004 = 8.99), and "
                             f"teacher forcing reached chrF {tf:.1f} against {no:.1f} on the same pairs.")


def check_09_02():
    """Greedy against beam, and your BLEU against the lab's."""
    got, err = _read("09_02_results.json")
    if err:
        return _report([err], "")
    need = ["greedy_chrf", "beam_chrf", "greedy_bleu", "beam_bleu", "my_bleu"]
    missing = [k for k in need if got.get(k) is None]
    if missing:
        return _report([f"out/09_02_results.json is missing {', '.join(missing)}. Run every cell, including "
                        "your clipped counts in section 5."], "")
    problems = []
    if not _close(got["my_bleu"], got["greedy_bleu"], 1e-6):
        problems.append(f"your BLEU ({got['my_bleu']:.3f}) differs from translate.bleu ({got['greedy_bleu']:.3f}) "
                        "on the same translations. Each n-gram earns at most as many matches as its highest "
                        "count in any one reference: min(count, most[gram]).")
    if not got["beam_chrf"] > got["greedy_chrf"]:
        problems.append("beam search should score above greedy decoding; rerun section 4 as shipped.")
    return _report(problems, f"Right: greedy chrF {got['greedy_chrf']:.1f}, beam chrF {got['beam_chrf']:.1f}, and "
                             f"your BLEU matches the lab's exactly ({got['my_bleu']:.2f}).")


def check_09_03():
    """The bottleneck measured: scores by length, and two sentences at once."""
    got, err = _read("09_03_results.json")
    if err:
        return _report([err], "")
    problems = []
    if not isinstance(got.get("by_length"), dict) or len(got["by_length"]) < 5:
        problems.append("the scores by length from section 2 are missing; run that cell.")
    j, s = got.get("joined_chrf"), got.get("separate_chrf")
    if j is None or s is None:
        problems.append("joined_chrf or separate_chrf is None: section 4 has not been filled in and run.")
    elif not s > j + 5:
        problems.append(f"translating the two halves separately (chrF {s:.1f}) should clearly beat translating "
                        f"them joined ({j:.1f}). separate should join each pair of halves' translations: a + b.")
    return _report(problems, f"Right: one sentence at a time scores chrF {s:.1f}; the same pairs joined into one "
                             f"input score {j:.1f}.")


PASS_CHRF = 46.0    # the lab model with a beam of 5 scores 45.1; see the README


def check_on_your_own():
    """Part 3: out/translator.pt, loaded and scored on the held-out sentences with its own beam size."""
    import sys
    sys.path.insert(0, HERE)
    import translate
    p = os.path.join(OUT, "translator.pt")
    if not os.path.exists(p):
        return _report(["out/translator.pt does not exist yet. Save your model with "
                        "translate.save_model(model, 'out/translator.pt', beam_size=..., ...)."], "")
    try:
        model, k = translate.load_model(p)
    except Exception as e:
        return _report([f"out/translator.pt could not be rebuilt: {e}"], "")
    problems = []
    limit = translate.LAB_STEPS + translate.BUDGET_STEPS
    if getattr(model, "steps_trained", 0) > limit:
        problems.append(f"the saved model has {model.steps_trained} training steps in all; the lab model's "
                        f"{translate.LAB_STEPS} plus the budget of {translate.BUDGET_STEPS} is {limit}. "
                        f"Start again from translate.lab_model() and pass max_steps={translate.BUDGET_STEPS} or less.")
    if not 1 <= int(k) <= translate.MAX_BEAM:
        problems.append(f"the saved beam size is {k}; the brief allows 1 to {translate.MAX_BEAM}.")
        k = 1
    print(f"Translating the held-out sentences with beam size {k}: about half a minute to a minute.", flush=True)
    d = translate.load_split()
    src, refs = translate.eval_split(d)
    try:
        hyps = translate.translate_all(model, src, d, k=int(k))
    except Exception as e:
        return _report([f"the model loaded but failed to translate: {e}"], "")
    score = translate.chrf(hyps, refs)
    if _read("translator_report.json")[1]:
        problems.append("out/translator_report.json is missing: save the report cell too.")
    if score < PASS_CHRF:
        problems.append(f"on the {len(src)} held-out sentences your translator scores chrF {score:.1f}; the brief "
                        f"asks for at least {PASS_CHRF}.")
    return _report(problems, f"Right: chrF {score:.1f} on the {len(src)} held-out sentences, beam size {k}.")
