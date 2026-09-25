"""nlpcheck: the feedback half of the Lab 07 notebooks.

Three kinds of help, used the same way in every lab of the course:

    ask("r1", "b")               answer a recall question from memory; you are
                                 told at once whether it is right, and why.
    guess("key", value)          commit to a prediction BEFORE the cell that
    reveal("key", actual)        shows the answer; reveal() compares the two.
    check_07_01()                check a saved exercise the way the checkpoint
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
    # 07_01, from the chapter pages and Chapter 6
    "r1": ("b", "A sigmoid squashes any number into the range 0 to 1, and its slope is largest (0.25) "
                "at 0 and nearly flat far from it. That flatness is why saturated neurons learn slowly."),
    "r2": ("c", "The chain rule multiplies the local derivatives along the path from the loss back to "
                "the weight. Backpropagation is the chain rule, applied to every weight at once."),
    # 07_02, from 07_01
    "r3": ("a", "Gradient descent subtracts the learning rate times the gradient: W_new = W_old - lr * dE/dW. "
                "A positive slope means stepping down, so the weight gets smaller."),
    "r4": ("b", "loss.backward() fills .grad on every tensor that requires it, by walking the graph "
                "PyTorch recorded during the forward pass. It does not change any weight."),
    # 07_03, from 07_02
    "r5": ("c", "With momentum 0.9, each step carries 0.9 of the last one, so a steady gradient adds up "
                "to about 1 / (1 - 0.9) = 10 times the plain step."),
    "r6": ("a", "A learning rate that is far too large overshoots the minimum further on every step, "
                "until the loss is infinite and then not a number (nan)."),
    # exit tickets
    "x1": ("b", "f = (x + y) * z. df/dx = z = -4, df/dy = z = -4, df/dz = x + y = 3: (-4, -4, 3)."),
    "x2": ("a", "Adam keeps a running average of each weight's gradient and of its square, and scales "
                "each weight's step by them, so every weight gets its own effective learning rate."),
    "x3": ("c", "The same recurrent weight multiplies the gradient at every step back in time; a factor "
                "below 1 multiplied fifty times is almost nothing. That is the vanishing gradient."),
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


def house_network():
    """The book's [4, 5, 1] network on its first house, with seeded weights.

    Returns X, Y and the four parameter tensors, freshly made, so a check can
    recompute the gradients the notebook should have found.
    """
    import torch
    X = torch.tensor([[0.00632, 18.0, 2.31, 0.0]]) / torch.tensor([0.1, 20.0, 10.0, 1.0])
    Y = torch.tensor([[0.24]])
    g = torch.Generator().manual_seed(7)
    W_ih = torch.rand(4, 5, generator=g)
    b_ih = torch.rand(5, generator=g)
    W_ho = torch.rand(5, 1, generator=g)
    b_ho = torch.rand(1, generator=g)
    return X, Y, W_ih, b_ih, W_ho, b_ho


def reference_gradients():
    import torch
    X, Y, *params = house_network()
    for p in params:
        p.requires_grad_(True)
    W_ih, b_ih, W_ho, b_ho = params
    O = torch.sigmoid(torch.sigmoid(X @ W_ih + b_ih) @ W_ho + b_ho)
    (0.5 * ((Y - O) ** 2).sum()).backward()
    return {"dW_ih": W_ih.grad.tolist(), "db_ih": b_ih.grad.tolist(),
            "dW_ho": W_ho.grad.tolist(), "db_ho": b_ho.grad.tolist()}


def _close(a, b, tol=1e-6):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def check_07_01():
    """Hand-computed gradients for the house network, against autograd."""
    got, err = _read("07_01_gradients.json")
    if err:
        return _report([err], "")
    ref = reference_gradients()
    problems = []
    for k, what in (("dW_ho", "the worked example"), ("db_ho", "the worked example"),
                    ("dW_ih", "your chain rule in section 4"), ("db_ih", "your chain rule in section 4")):
        if got.get(k) is None:
            problems.append(f"'{k}' is missing or None: {what} has not been filled in and run.")
        elif not _close(got[k], ref[k]):
            problems.append(f"'{k}' does not match what loss.backward() gives ({what}). Check which "
                            "terms you multiplied: delta2 @ W_ho.T, times h * (1 - h), then X.T @ that.")
    return _report(problems, "Right: all four hand-computed gradients match autograd to within 1e-6.")


def check_07_02():
    """The optimiser race: five runs, related the way the mechanisms say."""
    got, err = _read("07_02_race.json")
    if err:
        return _report([err], "")
    need = ["sgd_0.1", "sgd_1.0", "momentum_0.1", "adam_0.01", "sgd_50.0"]
    missing = [k for k in need if k not in got]
    if missing:
        return _report([f"out/07_02_race.json has no result for {', '.join(missing)}. "
                        "Run every race cell, including the one you complete in section 5."], "")
    loss = {k: got[k].get("final_loss") for k in need}
    problems = []
    if loss["sgd_50.0"] is not None and not (isinstance(loss["sgd_50.0"], float) and math.isnan(loss["sgd_50.0"])) \
            and not (isinstance(loss["sgd_50.0"], str) and loss["sgd_50.0"].lower() == "nan"):
        problems.append("SGD at learning rate 50 should have diverged to nan; rerun it as shipped.")
    fin = {k: v for k, v in loss.items() if k != "sgd_50.0"}
    if any(not isinstance(v, (int, float)) or math.isnan(v) for v in fin.values()):
        return _report(problems + ["one of the four sensible runs has no finite final_loss."], "")
    if not fin["adam_0.01"] < fin["momentum_0.1"] < fin["sgd_0.1"]:
        problems.append(f"expected Adam (0.01) below momentum (0.1) below plain SGD (0.1); got "
                        f"{fin['adam_0.01']:.3f}, {fin['momentum_0.1']:.3f}, {fin['sgd_0.1']:.3f}. "
                        "Check the optimiser you built in section 5 uses momentum=0.9 and lr=0.1.")
    if abs(fin["momentum_0.1"] - fin["sgd_1.0"]) > 0.08:
        problems.append(f"momentum at 0.1 ({fin['momentum_0.1']:.3f}) should land close to plain SGD at 1.0 "
                        f"({fin['sgd_1.0']:.3f}); is momentum set to 0.9?")
    return _report(problems, f"Right: SGD {fin['sgd_0.1']:.3f}, momentum {fin['momentum_0.1']:.3f} (about SGD at 1.0, "
                             f"{fin['sgd_1.0']:.3f}), Adam {fin['adam_0.01']:.3f}, and SGD at 50 diverged.")


def check_07_03():
    """The recurrent network measured: the book's model, RNN, bag, pooled RNN, gradients."""
    got, err = _read("07_03_results.json")
    if err:
        return _report([err], "")
    problems = []
    need = ["majority", "scalar_ids", "rnn", "bag", "rnn_mean", "grad_ratio_untrained", "grad_ratio_trained"]
    missing = [k for k in need if got.get(k) is None]
    if missing:
        return _report([f"out/07_03_results.json is missing {', '.join(missing)}. Run every cell, "
                        "including your pooled RNN in section 6."], "")
    f1 = {k: got[k]["macro_f1"] for k in ("scalar_ids", "rnn", "bag", "rnn_mean")}
    if f1["scalar_ids"] > 0.4:
        problems.append("the book's scalar-id network should barely beat guessing; rerun section 3 as shipped.")
    if not f1["rnn_mean"] > f1["rnn"] + 0.1:
        problems.append(f"the pooled RNN (macro-F1 {f1['rnn_mean']:.3f}) should clearly beat the last-state RNN "
                        f"({f1['rnn']:.3f}). Use newswire.RNNMeanClassifier and train it like the RNN above.")
    if not got["grad_ratio_untrained"] < 1e-6:
        problems.append("the untrained ratio should be tiny (well below one in a million); rerun section 5.")
    if not got["grad_ratio_untrained"] < got["grad_ratio_trained"]:
        problems.append("training should make the first word's gradient less tiny, not more; rerun section 5.")
    return _report(problems, f"Right: book's model {f1['scalar_ids']:.2f}, RNN {f1['rnn']:.2f}, pooled RNN "
                             f"{f1['rnn_mean']:.2f}, bag {f1['bag']:.2f} macro-F1; first-word gradient "
                             f"{got['grad_ratio_untrained']:.0e} of the last untrained.")


PASS_F1 = 0.70
PASS_ACC = 0.85


def check_on_your_own():
    """Part 3: out/newswire_model.pt, loaded and scored on the held-out test split."""
    import sys
    sys.path.insert(0, HERE)
    import newswire
    p = os.path.join(OUT, "newswire_model.pt")
    if not os.path.exists(p):
        return _report(["out/newswire_model.pt does not exist yet. Save your model with "
                        "newswire.save_model(model, 'out/newswire_model.pt', ...)."], "")
    try:
        model = newswire.load_model(p)
    except Exception as e:
        return _report([f"out/newswire_model.pt could not be rebuilt: {e}"], "")
    d = newswire.load_split()
    try:
        m = newswire.evaluate(model, d["X_test"], d["y_test"])
    except Exception as e:
        return _report([f"the model loaded but failed on the test data: {e}"], "")
    report, err = _read("newswire_report.json")
    problems = []
    if err:
        problems.append("out/newswire_report.json is missing: save the report cell too.")
    if m["macro_f1"] < PASS_F1 or m["accuracy"] < PASS_ACC:
        problems.append(f"on the held-out test split your model scores accuracy {m['accuracy']:.3f} and "
                        f"macro-F1 {m['macro_f1']:.3f}; the brief asks for at least {PASS_ACC} and {PASS_F1}. "
                        "Look at which topics it gets wrong before training longer.")
    return _report(problems, f"Right: accuracy {m['accuracy']:.3f}, macro-F1 {m['macro_f1']:.3f} on the "
                             f"{len(d['y_test'])} held-out newswires.")
