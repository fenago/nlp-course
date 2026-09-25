"""nlpcheck: the feedback half of the Lab 10 notebooks.

Three kinds of help, used the same way in every lab of the course:

    ask("r1", "b")               answer a recall question from memory; you are
                                 told at once whether it is right, and why.
    guess("key", value)          commit to a prediction BEFORE the cell that
    reveal("key", actual)        shows the answer; reveal() compares the two.
    check_10_01()                check a saved exercise the way the checkpoint
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
    # 10_01, from Lab 09 and the chapter pages
    "r1": ("b", "The classic encoder-decoder handed over one vector, the encoder's last hidden state. Every "
                "word of a long sentence had to squeeze through it, which is the bottleneck attention removes."),
    "r2": ("a", "Softmax exponentiates and divides by the total, so every weight is positive and they add up to "
                "1. For [2, 1, 0] they are 0.665, 0.245 and 0.090: not proportional to the scores."),
    # 10_02, from 10_01
    "r3": ("c", "Dot products of long random vectors are large, and a softmax of large numbers gives nearly all "
                "the weight to one word. Dividing by sqrt(d_k) keeps the scores at a size where the softmax "
                "still spreads its weight, and where gradients still flow."),
    "r4": ("b", "The heads split the dimensions rather than adding weights: with 8 dimensions and 4 heads, each "
                "head works on 2. Every one of the three layers had 288 weights."),
    # 10_03, from 10_02
    "r5": ("a", "Without a position signal, attention sees a set: the outputs at positions 0 and 7 are computed "
                "from exactly the same inputs, so no amount of training can make them differ in the right way."),
    "r6": ("a", "BERT is an encoder: it reads the whole sentence at once in both directions. A causal mask is "
                "for a decoder, which writes one word at a time and must not see the words it has not written."),
    # exit tickets
    "x1": ("c", "The context vector is the weighted sum of the value vectors: 0.2 v_The + 0.5 v_FBI + 0.3 v_is. "
                "Attention mixes; it does not pick one word."),
    "x2": ("b", "Each block is multi-head attention and then a feed-forward network applied to every position on "
                "its own, each wrapped in a residual connection and layer normalisation."),
    "x3": ("a", "Each head has its own query, key and value projections, so each can attend to a different kind "
                "of relationship (the next word, the nearest noun, the end marker) at the same time."),
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


def action_gets_results():
    """The three words, their embeddings and the three projection matrices of 10_01.

    Made from a fixed seed, as NumPy arrays of float64, so a check can recompute
    exactly the attention the notebook should have found.
    """
    import numpy as np
    rng = np.random.default_rng(10)
    X = rng.standard_normal((3, 4)).round(2)
    W_q, W_k, W_v = (rng.standard_normal((4, 4)).round(2) for _ in range(3))
    return ["Action", "gets", "results"], X, W_q, W_k, W_v


def reference_attention():
    import numpy as np
    _, X, W_q, W_k, W_v = action_gets_results()
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    s = Q @ K.T / np.sqrt(K.shape[-1])

    def softmax(a):
        e = np.exp(a - a.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    A = softmax(s)
    Ac = softmax(np.where(np.triu(np.ones((3, 3), dtype=bool), k=1), -np.inf, s))
    return {"A": A, "Z": A @ V, "A_causal": Ac, "Z_causal": Ac @ V}


def _close(a, b, tol=1e-6):
    import numpy as np
    try:
        a = np.array(a, dtype=float)
    except (TypeError, ValueError):
        return False
    return a.shape == np.shape(b) and bool(np.abs(a - b).max() <= tol)


def check_10_01():
    """Attention, the causal mask and the hand-run heads of 10_01."""
    got, err = _read("10_01_attention.json")
    if err:
        return _report([err], "")
    ref = reference_attention()
    problems = []
    for k, what in (("A", "the worked example"), ("Z", "the worked example"),
                    ("A_causal", "your causal mask in section 3"), ("Z_causal", "your causal mask in section 3")):
        if got.get(k) is None:
            problems.append(f"'{k}' is missing or None: {what} has not been filled in and run.")
        elif not _close(got[k], ref[k]):
            problems.append(f"'{k}' does not match attention recomputed from the same inputs ({what}). "
                            "The mask must be True above the diagonal (np.triu with k=1), and the masked "
                            "scores set to -np.inf before the softmax.")
    d = got.get("mha_max_diff")
    if d is None or not d < 1e-5:
        problems.append("the hand-run heads in section 4 should match nn.MultiheadAttention to within 1e-5; "
                        "rerun that cell as shipped.")
    return _report(problems, "Right: your attention and your causal attention match the reference to within "
                             "1e-6, and the layer taken apart matches nn.MultiheadAttention.")


def check_10_02():
    """The position signal, the reversal with and without positions, the saved model."""
    import torch
    import sys
    sys.path.insert(0, HERE)
    import attnlab
    got, err = _read("10_02_results.json")
    if err:
        return _report([err], "")
    problems = []
    pe = got.get("pe_50_16")
    if pe is None or not _close(pe, attnlab.sinusoidal_positions(50, 16).numpy(), 1e-5):
        problems.append("your position signal does not match the formula: sines in the even columns "
                        "(pe[:, 0::2] = torch.sin(angle)), cosines in the odd ones (pe[:, 1::2]).")
    if got.get("acc_without") is None:
        problems.append("there is no result without positions: build the model in section 5 with "
                        "positions=False and train it.")
    elif got["acc_without"] > 0.6:
        problems.append(f"without positions the model scored {got['acc_without']:.2f}; it should stay far "
                        "below 1. Check that you passed positions=False.")
    p = os.path.join(OUT, "reverse_model.pt")
    if not os.path.exists(p):
        problems.append("out/reverse_model.pt is missing: run the save cell.")
    else:
        m = attnlab.load_model(p)
        x, y = attnlab.reverse_batch(1000, seed=99)
        with torch.no_grad():
            acc = float((m(x).argmax(-1) == y).float().mean())
        if acc < 0.95:
            problems.append(f"the saved model with positions reverses only {acc:.2f} of new digits; "
                            "rerun section 4 as shipped.")
    if problems:
        return _report(problems, "")
    return _report([], f"Right: the position signal matches the formula, the saved model reverses {acc:.1%} "
                       f"of 1,000 new sequences, and without positions it managed {got['acc_without']:.1%}.")


def bert_reference():
    """Recompute everything 10_03 measures, from the offline bert-mini."""
    import math
    import torch
    from transformers import AutoModel, AutoTokenizer, logging
    logging.set_verbosity_error()
    logging.disable_progress_bar()
    name = "google/bert_uncased_L-4_H-256_A-4"
    tok = AutoTokenizer.from_pretrained(name)
    bert = AutoModel.from_pretrained(name, attn_implementation="eager").eval()

    def maps(s):
        e = tok(s, return_tensors="pt")
        with torch.no_grad():
            a = torch.stack(bert(**e, output_attentions=True).attentions)[:, 0]
        return tok.convert_ids_to_tokens(e["input_ids"][0]), a

    t, a = maps("The phone would not charge because it was broken.")
    it, phone = t.index("it"), t.index("phone")
    w = a[:, :, it, phone]
    layer, head = divmod(int(w.argmax()), w.shape[1])
    return {"best_layer": layer, "best_head": head, "best_weight": float(w.max()),
            "sink_last_layer": float(a[-1, :, it, t.index("[SEP]")].mean())}


def check_10_03():
    """bert-mini's attention: layer 0 by hand, the [SEP] sink, the head that finds the phone."""
    got, err = _read("10_03_bert.json")
    if err:
        return _report([err], "")
    ref = bert_reference()
    problems = []
    if got.get("by_hand_max_diff") is None or not got["by_hand_max_diff"] < 1e-4:
        problems.append("layer 0 by hand should match BERT's own attention to within 1e-4; rerun section 2.")
    if got.get("best_layer") is None:
        problems.append("best_layer is still None: write the loop in section 4.")
    elif (got["best_layer"], got["best_head"]) != (ref["best_layer"], ref["best_head"]) \
            or abs(got.get("best_weight", 0) - ref["best_weight"]) > 0.01:
        problems.append(f"the loop found layer {got['best_layer']}, head {got['best_head']} "
                        f"({got.get('best_weight', 0):.2f}), but the head where 'it' gives 'phone' the most weight "
                        "is another one. Check that you compare every layer and head and keep the largest.")
    if got.get("sink_last_layer") is None or abs(got["sink_last_layer"] - ref["sink_last_layer"]) > 0.01:
        problems.append("the last layer's weight on [SEP] is missing or wrong; rerun section 3 as shipped.")
    if got.get("pair_change") is None:
        problems.append("the pair test in section 5 has not been run.")
    return _report(problems, f"Right: layer 0 matches by hand, 'it' parks {ref['sink_last_layer']:.2f} of its "
                             f"last-layer attention on [SEP], and layer {ref['best_layer']}, head "
                             f"{ref['best_head']} gives 'phone' {ref['best_weight']:.2f}.")


PASS_ACC = 0.88
PASS_F1 = 0.88


def check_on_your_own():
    """Part 3: out/headline_model.pt, loaded and scored on the held-out headlines."""
    import sys
    import torch.nn as nn
    sys.path.insert(0, HERE)
    import attnlab
    p = os.path.join(OUT, "headline_model.pt")
    if not os.path.exists(p):
        return _report(["out/headline_model.pt does not exist yet. Save your model with "
                        "attnlab.save_model(model, 'out/headline_model.pt', ...)."], "")
    try:
        model = attnlab.load_model(p)
    except Exception as e:
        return _report([f"out/headline_model.pt could not be rebuilt: {e}"], "")
    if not any(isinstance(m, nn.MultiheadAttention) for m in model.modules()):
        return _report([f"{type(model).__name__} has no attention layer. The brief asks for a transformer: "
                        "use attnlab.HeadlineTransformer, or a class built from attnlab.TransformerBlock."], "")
    d = attnlab.load_split()
    try:
        m = attnlab.evaluate(model, d["X_test"], d["y_test"])
    except Exception as e:
        return _report([f"the model loaded but failed on the test data: {e}"], "")
    report, err = _read("headline_report.json")
    problems = []
    if err:
        problems.append("out/headline_report.json is missing: save the report cell too.")
    if m["accuracy"] < PASS_ACC or m["macro_f1"] < PASS_F1:
        problems.append(f"on the held-out headlines your model scores accuracy {m['accuracy']:.3f} and "
                        f"macro-F1 {m['macro_f1']:.3f}; the brief asks for at least {PASS_ACC} and {PASS_F1}.")
    return _report(problems, f"Right: accuracy {m['accuracy']:.3f}, macro-F1 {m['macro_f1']:.3f} on the "
                             f"{len(d['y_test'])} held-out headlines.")
