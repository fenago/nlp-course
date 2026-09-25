"""nlpcheck: the feedback half of the Lab 08 notebooks.

Three kinds of help, used the same way in every lab of the course:

    ask("r1", "b")               answer a recall question from memory; you are
                                 told at once whether it is right, and why.
    guess("key", value)          commit to a prediction BEFORE the cell that
    reveal("key", actual)        shows the answer; reveal() compares the two.
    check_08_01()                check a saved exercise the way the checkpoint
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
    # 08_01, from Lab 07 and the chapter pages
    "r1": ("b", "Each step back multiplies the gradient by W_hh and by the slope of tanh, which is at most 1. "
                "Many factors below 1 multiplied together are the vanishing gradient."),
    "r2": ("b", "Clipping rescales the whole gradient down to a maximum size before the step, keeping its "
                "direction. It tames exploding gradients; it cannot revive vanished ones."),
    # 08_02, from 08_01
    "r3": ("c", "c(t) = f * c(t-1) + i * g: going back one step along the cell state multiplies the gradient "
                "by the forget gate f alone, with no weight matrix and no tanh."),
    "r4": ("a", "PyTorch stacks the four gates in the order input, forget, candidate (g), output, so an LSTM "
                "has four times the weights of an RNN of the same size."),
    # 08_03, from 08_02
    "r5": ("b", "With the padding on the right and no packing, a short headline's final state comes after "
                "several steps of zeros, and the gradient has to cross them to reach any real word."),
    "r6": ("a", "pack_padded_sequence takes each row's real length, so the recurrent layer never steps "
                "through a pad and returns each row's state after its own last word."),
    # exit tickets
    "x1": ("b", "The forget gate is a sigmoid: 1 keeps a slot of the cell state, 0 throws it away."),
    "x2": ("a", "The candidate g is a tanh layer that proposes new values between -1 and 1; the input gate "
                "decides how much of it is written."),
    "x3": ("c", "Loss falling from 1.386 means the network has started to carry the first word; 1.386 is "
                "ln 4, the loss of guessing evenly among four classes."),
    "x4": ("b", "A BiLSTM is two LSTMs, one reading forwards and one backwards, with their states joined."),
    "x5": ("a", "An average of embeddings gives exactly the same answer for any order of the same words, so "
                "shuffling cannot change it."),
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


def lstm_step_inputs():
    """One word, a last hidden state and cell state, and a tiny seeded nn.LSTM(3, 2).

    Freshly made each call, so a check can recompute the step the notebook should
    have computed by hand.
    """
    import torch
    torch.manual_seed(8)
    lstm = torch.nn.LSTM(3, 2, batch_first=True)
    g = torch.Generator().manual_seed(8)
    x = torch.randn(1, 3, generator=g)
    h0 = torch.randn(1, 2, generator=g)
    c0 = torch.randn(1, 2, generator=g)
    return x, h0, c0, lstm


def reference_step():
    import torch
    x, h0, c0, lstm = lstm_step_inputs()
    with torch.no_grad():
        _, (h, c) = lstm(x.unsqueeze(0), (h0.unsqueeze(0), c0.unsqueeze(0)))
    return h[0].tolist(), c[0].tolist()


def _close(a, b, tol=1e-5):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def check_08_01(part="all"):
    """One LSTM step by hand against nn.LSTM, then the three gradient ratios."""
    got, err = _read("08_01_cell.json")
    if err:
        return _report([err], "")
    h_ref, c_ref = reference_step()
    problems = []
    if got.get("c1") is None or got.get("h1") is None:
        problems.append("c1 or h1 is missing: fill in the output gate and c1 in section 3 and run both cells.")
    elif not _close(got["c1"], c_ref):
        problems.append("your c1 does not match nn.LSTM's cell state. It is f * c0 + i * g.")
    elif not _close(got["h1"], h_ref):
        problems.append("c1 matches but h1 does not: check the output gate uses W_io, W_ho and b_o with a sigmoid.")
    if part == "cell" or problems:
        return _report(problems, "Right: your c1 and h1 match nn.LSTM to within 1e-5.")
    need = ("ratio_rnn", "ratio_lstm", "ratio_lstm_open")
    if any(not isinstance(got.get(k), (int, float)) for k in need):
        return _report(["the gradient ratios are missing: run sections 4 and 5, then the save cell."], "")
    if not got["ratio_rnn"] < got["ratio_lstm"] < got["ratio_lstm_open"]:
        problems.append("expected the RNN's first-word gradient below the LSTM's, and the LSTM's below the "
                        "open-forget-gate LSTM's; rerun sections 4 and 5 as shipped.")
    if not got["ratio_lstm_open"] > 0.1:
        problems.append("with the forget gate opened to bias 3 the first word should keep a large share of "
                        "the gradient; did open_forget_gate run on the model you measured?")
    return _report(problems, f"Right: c1 and h1 match nn.LSTM; first-word gradient {got['ratio_rnn']:.0e} (RNN), "
                             f"{got['ratio_lstm']:.0e} (LSTM), {got['ratio_lstm_open']:.2f} (forget gate open).")


def check_08_02():
    """The memory task, the headlines, and the padding test, in the relationships the pages explain."""
    got, err = _read("08_02_results.json")
    if err:
        return _report([err], "")
    mem, head = got.get("memory", {}), got.get("headlines", {})
    missing = [k for k in ("rnn", "lstm", "lstm_open") if k not in mem] + \
              [k for k in ("rnn", "lstm", "rnn_padded", "lstm_padded") if k not in head]
    if missing:
        return _report([f"out/08_02_results.json has no result for {', '.join(missing)}. Run every cell, "
                        "including your padded LSTM in section 5, then the save cell."], "")
    acc = {k: v["accuracy"] for k, v in head.items()}
    problems = []
    if not mem["lstm_open"] > mem["rnn"] + 0.3:
        problems.append(f"on the first-word task the open-gate LSTM ({mem['lstm_open']:.2f}) should far outscore the RNN "
                        f"({mem['rnn']:.2f}); rerun section 2 as shipped.")
    if not acc["rnn_padded"] < 0.5:
        problems.append("the RNN reading the padding should score near guessing; rerun section 4 as shipped.")
    if not acc["lstm_padded"] > acc["rnn_padded"] + 0.25:
        problems.append(f"your padded LSTM scores {acc['lstm_padded']:.3f}; it should be far above the padded RNN "
                        f"({acc['rnn_padded']:.3f}). Use cell=\"lstm\" and pack=False.")
    return _report(problems, f"Right: first-word task RNN {mem['rnn']:.2f}, LSTM {mem['lstm']:.2f}, open-gate LSTM "
                             f"{mem['lstm_open']:.2f}; headlines "
                             f"RNN {acc['rnn']:.3f}, LSTM {acc['lstm']:.3f}; reading the padding, RNN "
                             f"{acc['rnn_padded']:.3f}, LSTM {acc['lstm_padded']:.3f}.")


def check_08_03():
    """LSTM, BiLSTM, bag, and the shuffle test."""
    got, err = _read("08_03_results.json")
    if err:
        return _report([err], "")
    need = ["lstm", "bilstm", "bag", "lstm_shuffled", "bag_shuffled"]
    missing = [k for k in need if k not in got]
    if missing:
        return _report([f"out/08_03_results.json has no result for {', '.join(missing)}. Run every cell, "
                        "including your shuffle line in section 4, then the save cell."], "")
    acc = {k: got[k]["accuracy"] for k in need}
    problems = []
    if acc["bilstm"] < 0.85:
        problems.append(f"the BiLSTM scores {acc['bilstm']:.3f}, not about 0.9; rerun section 2 as shipped.")
    if abs(acc["bag"] - acc["bag_shuffled"]) > 1e-9:
        problems.append("the bag should score exactly the same on shuffled headlines; rerun section 5 as shipped.")
    if not acc["lstm_shuffled"] < acc["lstm"]:
        problems.append("the LSTM scores the same on the shuffled headlines, so they were not shuffled: complete "
                        "the line in section 4 with torch.randperm(k, generator=g).")
    return _report(problems, f"Right: LSTM {acc['lstm']:.3f}, BiLSTM {acc['bilstm']:.3f}, bag {acc['bag']:.3f}; "
                             f"shuffled, the LSTM drops to {acc['lstm_shuffled']:.3f} and the bag stays at "
                             f"{acc['bag_shuffled']:.3f}.")


PASS_F1 = 0.895


def _recurrent(model):
    import torch.nn as nn
    return any(isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)) for m in model.modules())


def check_on_your_own():
    """Part 3: out/headline_model.pt, loaded and scored on the held-out headlines."""
    import sys
    sys.path.insert(0, HERE)
    import headlines
    p = os.path.join(OUT, "headline_model.pt")
    if not os.path.exists(p):
        return _report(["out/headline_model.pt does not exist yet. Save your model with "
                        "headlines.save_model(model, 'out/headline_model.pt', ...)."], "")
    try:
        model = headlines.load_model(p)
    except Exception as e:
        return _report([f"out/headline_model.pt could not be rebuilt: {e}"], "")
    problems = []
    if not _recurrent(model):
        problems.append("the brief asks for a network that reads in order: the model has no nn.RNN, nn.LSTM "
                        "or nn.GRU layer.")
    d = headlines.load_split()
    try:
        m = headlines.evaluate(model, d["X_test"], d["y_test"])
    except Exception as e:
        return _report([f"the model loaded but failed on the test data: {e}"], "")
    report, err = _read("headline_report.json")
    if err:
        problems.append("out/headline_report.json is missing: save the report cell too.")
    if m["macro_f1"] < PASS_F1:
        problems.append(f"on the held-out headlines your model scores macro-F1 {m['macro_f1']:.3f}; the brief "
                        f"asks for at least {PASS_F1}. Compare with what the notebooks measured before you "
                        "train longer.")
    return _report(problems, f"Right: accuracy {m['accuracy']:.3f}, macro-F1 {m['macro_f1']:.3f} on the "
                             f"{len(d['y_test'])} held-out headlines.")
