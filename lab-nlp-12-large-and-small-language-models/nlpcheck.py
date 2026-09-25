"""nlpcheck: the feedback half of the Lab 12 notebooks.

Three kinds of help, used the same way in every lab of the course:

    ask("r1", "b")               answer a recall question from memory; you are
                                 told at once whether it is right, and why.
    guess("key", value)          commit to a prediction BEFORE the cell that
    reveal("key", actual)        shows the answer; reveal() compares the two.
    check_12_01()                check a saved exercise the way the checkpoint
                                 will, with a message that says what to fix.

Predictions are kept in out/predictions.json. Nothing is marked on them: they
are there because a guess you wrote down is one you remember being wrong
about, and the sign-off counts how many you made.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PRED = os.path.join(OUT, "predictions.json")


# ---------------------------------------------------------------- recall ---

QUESTIONS = {
    # 12_01, from Lab 11 and the chapter pages
    "r1": ("b", "BERT is trained to fill in masked words using the words on both sides. A decoder-only "
                "model like SmolLM2 only ever sees the words to the left, and predicts the next one."),
    "r2": ("a", "Softmax exponentiates every score and divides by the total, so the results are positive "
                "and add up to 1: a probability for every entry in the vocabulary."),
    # 12_02, from 12_01
    "r3": ("a", "Temperature 0 is greedy decoding: the most likely token every time, so the same prompt "
                "gives the same text on every run."),
    "r4": ("b", "A logit is the raw score the model's last layer gives a token. Softmax turns the logits "
                "into probabilities; dividing them by a temperature first flattens or sharpens them."),
    # 12_03, from 12_02
    "r5": ("b", "The chat template wraps every message in role markers (<|im_start|>user ... <|im_end|>) "
                "exactly as in the conversations the model was fine-tuned on, and ends with the "
                "assistant's marker so the next tokens are the reply."),
    "r6": ("c", "A system prompt is only more text at the front of the input. It changes the probabilities "
                "of what comes next; it cannot add knowledge the weights do not hold."),
    # exit tickets
    "x1": ("c", "Top-p keeps the smallest set of most likely tokens whose probabilities add up to p, and "
                "samples only from those. At a high temperature that set grows to thousands of tokens."),
    "x2": ("a", "The model scored 'billing' highest for most tickets whatever they said. Subtracting each "
                "department's average score removes that prior, which is what calibration does."),
    "x3": ("b", "Retrieval puts the facts in the prompt; the model only has to copy and phrase them. It "
                "does not make the model know more, which is why the notice has to be the right one."),
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


def ref_temperature(logits, T):
    import torch
    return (torch.as_tensor(logits, dtype=torch.float32) / T).softmax(-1)


def ref_top_p(probs, p):
    """How many of the most likely tokens top-p keeps: the fewest whose probabilities reach p."""
    import torch
    s = torch.as_tensor(probs).sort(descending=True).values
    return int((s.cumsum(0) < p).sum()) + 1


def check_12_01():
    """Greedy decoding by hand, temperature and top-p, against the references."""
    got, err = _read("12_01_next_token.json")
    if err:
        return _report([err], "")
    problems = []
    if not got.get("greedy_ids"):
        problems.append("'greedy_ids' is empty: your loop in section 4 has not been filled in and run.")
    elif got["greedy_ids"] != got.get("generate_ids"):
        problems.append("your greedy loop and model.generate() chose different tokens. Each step must append "
                        "the argmax of the logits for the whole sequence so far.")
    temps = got.get("top1_at_T") or {}
    ref = got.get("reference_top1_at_T") or {}
    for T in ("0.5", "1.0", "1.5"):
        if temps.get(T) is None:
            problems.append(f"no top-token probability at temperature {T}: with_temperature() is not filled in.")
        elif ref.get(T) is None or abs(temps[T] - ref[T]) > 1e-4:
            problems.append(f"at temperature {T} your top probability {temps.get(T)} is not softmax(logits / T).")
    kept = got.get("top_p_kept") or {}
    rk = got.get("reference_top_p_kept") or {}
    if not kept or any(kept.get(k) != rk.get(k) for k in rk):
        problems.append(f"top-p kept {kept} tokens where the reference keeps {rk}. Keep the most likely tokens "
                        "until their probabilities add up to p, and include the one that crosses it.")
    return _report(problems, f"Right: your greedy loop matches generate() token for token, and at p = 0.9 "
                             f"top-p keeps {rk.get('1.0')} tokens at T = 1 and {rk.get('1.5')} at T = 1.5.")


def check_12_02():
    """The raw and templated replies, and calibrated ticket routing against the classifier."""
    got, err = _read("12_02_prompts.json")
    if err:
        return _report([err], "")
    problems = []
    if "Customer:" not in (got.get("raw_reply") or ""):
        problems.append("'raw_reply' should be the raw continuation from section 2, which writes the customer's "
                        "next line too. Rerun section 2 as shipped.")
    if not got.get("template_tokens") or got["template_tokens"] < 20:
        problems.append("'template_tokens' should count the tokens the chat template adds; rerun section 3.")
    y = got.get("labels") or []
    L = got.get("logits") or []
    cal = got.get("calibrated_predictions")
    if not y or len(L) != len(y):
        problems.append("the ticket logits or labels are missing; run section 4.")
    elif not cal or len(cal) != len(y):
        problems.append("'calibrated_predictions' is missing: fill in the calibration in section 5 and run it.")
    else:
        import torch
        T = torch.tensor(L)
        from slm import DEPARTMENTS
        ref = [DEPARTMENTS[int(i)] for i in (T - T.mean(0)).argmax(1)]
        if cal != ref:
            problems.append("your calibrated predictions differ from subtracting each department's mean logit "
                            "over the 12 tickets. Subtract the mean of each column, then take the argmax.")
    if (got.get("classifier_accuracy") or 0) < 0.85:
        problems.append("the TF-IDF classifier's accuracy on the 100 held-out tickets is missing or low; rerun "
                        "section 6 as shipped.")
    if problems:
        return _report(problems, "")
    acc = lambda p: sum(a == b for a, b in zip(p, y)) / len(y)
    raw = [__import__("slm").DEPARTMENTS[max(range(4), key=lambda j: r[j])] for r in L]
    return _report([], f"Right: the prompt alone routes {acc(raw):.0%} of the 12 tickets, calibrated {acc(cal):.0%}; "
                       f"the classifier {got['classifier_accuracy']:.0%} of 100.")


def check_12_03():
    """Closed-book against grounded answers, and the retrieval results."""
    got, err = _read("12_03_grounding.json")
    if err:
        return _report([err], "")
    problems = []
    if not got.get("closed_book"):
        problems.append("'closed_book' is missing: run section 2.")
    if "32.50" not in (got.get("grounded_flex30") or ""):
        problems.append("the grounded Flex 30 answer in section 5 does not contain $32.50. Put the retrieved "
                        "notice in the message the model reads.")
    g3 = str(got.get("your_3g", "")).lower()
    if not re.search(r"31 march|march 31", g3):
        problems.append("your ask_kittiwake() answer to the 3G question does not say 31 March. Check it builds the "
                        "messages from the retrieved notice.")
    if 14 not in (got.get("your_3g_passages") or []):
        problems.append("ask_kittiwake() did not retrieve notice 14, the 3G notice.")
    return _report(problems, "Right: closed book the model invents a price; with the notice in front of it, it "
                             "reads $32.50, and your own function answers the 3G question from notice 14.")


# ------------------------------------------------------------ on your own --

# The fixed question set. Each answerable question lists the facts its answer must
# contain, as regular expressions over the normalised answer (lower case, no $ or commas).
QUESTION_SET = [
    ("How much will the Flex 30 plan cost from 1 November?", [r"32\.50"]),
    ("When will the mast at St. Aldric's Road in Gullhaven be upgraded?", [r"14 october|october 14"]),
    ("What credit will customers affected by the Cragwell outage get?", [r"(?<![\d.])5(\.00)?(?![\d.])"]),
    ("My Nimbus X2 is stuck after the update. What should I do?", [r"20 seconds"]),
    ("When does the 3G network switch off?", [r"31 march|march 31"]),
    ("How much does roaming cost per day on a Flex plan?", [r"(?<![\d.])4(\.00)?(?![\d.])"]),
    ("Does Kittiwake sell broadband for the home?", None),
    ("Can I pay my bill in cash at a store?", None),
]
DECLINE = re.compile(r"don.?t know|do not know|not (in|mentioned|stated|covered)|no information|"
                     r"can.?t find|cannot find|isn.?t in|unable to (answer|find)|not able to answer")
PASS_FACTS = 5  # of the 6 answerable questions; and every unanswerable one declined


def normalise(text):
    return re.sub(r"\s+", " ", str(text).lower().replace("$", "").replace(",", "")).strip()


def grade(results):
    """Score saved assistant results: (facts right, answerable, declined right, unanswerable, rows)."""
    rows, facts, declined = [], 0, 0
    by_q = {r.get("question"): r for r in results}
    for q, need in QUESTION_SET:
        a = normalise((by_q.get(q) or {}).get("answer", ""))
        if need is None:
            ok = bool(DECLINE.search(a))
            declined += ok
        else:
            ok = all(re.search(n, a) for n in need) and not DECLINE.search(a)
            facts += ok
        rows.append((q, ok))
    n_ans = sum(1 for _, n in QUESTION_SET if n is not None)
    return facts, n_ans, declined, len(QUESTION_SET) - n_ans, rows


def check_on_your_own():
    """Part 3: out/assistant_answers.json, graded on the fixed question set."""
    got, err = _read("assistant_answers.json")
    if err:
        return _report([err], "")
    results = got.get("results") or []
    missing = [q for q, _ in QUESTION_SET if q not in {r.get("question") for r in results}]
    if missing:
        return _report([f"{len(missing)} of the {len(QUESTION_SET)} questions have no saved answer; run the "
                        "whole set with slm.run_question_set(answer)."], "")
    facts, n_ans, dec, n_un, rows = grade(results)
    problems = []
    if facts < PASS_FACTS:
        problems.append(f"{facts} of {n_ans} answerable questions contain the fact from the notices; the brief "
                        f"asks for {PASS_FACTS}. Wrong ones: " + "; ".join(q for (q, ok), (_, n) in
                        zip(rows, QUESTION_SET) if n is not None and not ok))
    if dec < n_un:
        problems.append(f"{dec} of {n_un} unanswerable questions were declined; every one must be. A small model "
                        "rarely says it does not know on its own: look at the retrieval scores.")
    if not any(r.get("messages") for r in results):
        problems.append("no result records the messages the model read, so nothing can be replayed. Return "
                        "them from answer() as the skeleton shows.")
    return _report(problems, f"Right: {facts} of {n_ans} answers carry the fact, {dec} of {n_un} unanswerable "
                             f"questions declined, in {got.get('seconds')} s for the set.")
