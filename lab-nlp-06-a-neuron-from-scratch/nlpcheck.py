"""nlpcheck: the feedback half of the Lab 06 notebooks.

Three kinds of help, used the same way in every lab of the course:

    ask("r1", "b")               answer a recall question from memory; you are
                                 told at once whether it is right, and why.
    guess("xor_epochs", 50)      commit to a prediction BEFORE the cell that
    reveal("xor_epochs", x)      shows the answer; reveal() compares the two.
    check_06_01()                check a saved exercise the way the checkpoint
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
DATA = os.path.join(HERE, "data")
PRED = os.path.join(OUT, "predictions.json")


# ---------------------------------------------------------------- recall ---

QUESTIONS = {
    # 06_01, from earlier labs (spaced retrieval)
    "r1": ("b", "Logistic regression (Lab 03) passes a weighted sum through the sigmoid, which squeezes any "
                "number into 0 to 1. A single artificial neuron with a sigmoid is exactly that model."),
    "r2": ("b", "An embedding (Lab 02) is a list of numbers that places a text by its meaning, so texts that mean "
                "similar things get nearby vectors. Part 3 of this lab feeds them to a network."),
    # 06_02, from 06_01
    "r3": ("b", "The perceptron draws one straight line. No straight line puts (0,1) and (1,0) on one side and "
                "(0,0) and (1,1) on the other, so no amount of training finds one."),
    "r4": ("and", "1 + 1 - 1.5 is positive only when both inputs are 1: the neuron fires for AND. With a bias "
                  "of -0.5 it would fire for OR."),
    # 06_03, from 06_02
    "r5": ("1", "W2(W1 x + b1) + b2 multiplies out to W x + b: any stack of linear layers is one linear layer "
                "in disguise, which is why the network in 06_02 could not bend until it had an activation."),
    "r6": ("b", "Units that start the same get the same gradient, so they take the same step every time and "
                "stay copies of each other. Random starting weights break the symmetry."),
    # exit tickets
    "x1": ("d", "Initialise the weights, compute an output for a sample, change the weights if it was wrong, then "
                "move on to the next sample: 1, 4, 3, 2."),
    "x2": ("c", "Leaky ReLU keeps a small slope (0.01) for negative inputs, so the gradient is never exactly zero "
                "and a neuron cannot switch off for good."),
    "x3": ("b", "An untrained network gives each of 4 classes about a quarter of the probability, and "
                "-ln(1/4) = ln 4, about 1.386. A first loss far from that is a sign of a bug."),
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


X4 = [[0, 0], [0, 1], [1, 0], [1, 1]]
TARGETS = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "XOR": [0, 1, 1, 0]}


def reference_perceptron(targets, epochs=20, lr=0.25, seed=0):
    """The perceptron of 06_01, written independently, for the check to compare with."""
    import numpy as np
    rng = np.random.default_rng(seed)
    w = rng.uniform(-0.05, 0.05, 3)
    Xb = np.hstack([-np.ones((4, 1)), np.array(X4, float)])
    errors = []
    for _ in range(epochs):
        n = 0
        for x, t in zip(Xb, targets):
            y = 1 if x @ w > 0 else 0
            n += int(y != t)
            w = w + lr * (t - y) * x
        errors.append(n)
    acc = float(np.mean([(1 if x @ w > 0 else 0) == t for x, t in zip(Xb, targets)]))
    return w.tolist(), errors, acc


def check_06_01():
    """The perceptron's learning rule, trained on AND, OR and XOR."""
    got, err = _read("06_01_perceptron.json")
    if err:
        return _report([err], "")
    problems = []
    for name, t in TARGETS.items():
        w, errors, acc = reference_perceptron(t)
        g = got.get(name) or {}
        if g.get("errors") != errors:
            hint = (" Every epoch has the same count: perceptron_step is still returning the weights unchanged."
                    if g.get("errors") and len(set(g["errors"])) == 1 and name != "XOR" else "")
            problems.append(f"{name}: the mistakes per epoch should be {errors[:6]}..., you have "
                            f"{(g.get('errors') or [])[:6]}...{hint}")
        elif any(abs(a - b) > 1e-9 for a, b in zip(g.get("weights", []), w)):
            problems.append(f"{name}: the mistakes match but the final weights do not; check the update is "
                            "w + lr * (target - y) * x.")
    if not problems and got.get("AND", {}).get("accuracy") != 1.0:
        problems.append("AND should be learned perfectly.")
    return _report(problems, "Right: AND and OR learned perfectly, and XOR still wrong on "
                             f"{reference_perceptron(TARGETS['XOR'])[1][-1]} of 4 after 20 epochs.")


ACTIVATIONS = ("Tanh", "ReLU", "Sigmoid", "GELU", "LeakyReLU", "ELU", "SiLU")


def check_06_02():
    """The fixed XOR network: an activation between the layers, and XOR learned."""
    got, err = _read("06_02_xor.json")
    if err:
        return _report([err], "")
    problems = []
    layers = got.get("layers") or []
    if not any(any(a in str(l) for a in ACTIVATIONS) for l in layers):
        problems.append("the network still has no activation between its layers (the planted bug in section 3): "
                        "add nn.Tanh() between the two nn.Linear layers.")
    if got.get("predictions") != [0, 1, 1, 0]:
        problems.append(f"the network predicts {got.get('predictions')} for the four inputs; XOR is [0, 1, 1, 0].")
    if not isinstance(got.get("loss"), (int, float)) or got["loss"] > 0.1:
        problems.append(f"the final loss is {got.get('loss')}; a network that has learned XOR ends well under 0.1.")
    if got.get("identical_hidden_rows") is not True:
        problems.append("section 4's constant start should leave every hidden unit's weights identical; run it as shipped.")
    return _report(problems, f"Right: {len([l for l in layers])} layers with an activation, XOR learned "
                             f"(loss {got.get('loss', 0):.5f}), and the constant start kept every hidden unit identical.")


LOGITS = [[2.0, 1.0, 0.1, -1.0], [0.5, 2.5, 0.3, 0.0], [1.2, 0.9, 0.75, 0.2]]
LABELS = [0, 1, 3]


def reference_cross_entropy():
    total = 0.0
    for z, t in zip(LOGITS, LABELS):
        m = max(z)
        log_sum = m + math.log(sum(math.exp(v - m) for v in z))
        total += log_sum - z[t]
    return total / len(LABELS)


def check_06_03():
    """Cross-entropy by hand, matched against PyTorch, and the untrained loss."""
    got, err = _read("06_03_losses.json")
    if err:
        return _report([err], "")
    want = reference_cross_entropy()
    problems = []
    h = got.get("by_hand")
    if not isinstance(h, (int, float)):
        problems.append("'by_hand' is missing: finish the cross-entropy cell in section 5.")
    elif abs(h - want) > 1e-4:
        hint = ""
        if abs(h + want) < 1e-4:
            hint = " The sign is flipped: the loss is minus the log of the probability."
        problems.append(f"'by_hand' is {h:.4f}; the cross-entropy of those logits is {want:.4f}.{hint}")
    if abs((got.get("torch") or 0) - want) > 1e-4:
        problems.append("'torch' should be nn.CrossEntropyLoss() on the same logits and labels; run section 5 again.")
    u = got.get("untrained")
    if not isinstance(u, (int, float)) or abs(u - math.log(4)) > 0.05:
        problems.append(f"'untrained' is {u}; an untrained 4-class network's loss should be close to ln 4 = 1.386.")
    return _report(problems, f"Right: {want:.4f} by hand and from PyTorch, and an untrained 4-class network "
                             f"starts at {u:.3f}, next to ln 4 = {math.log(4):.3f}.")


def check_on_your_own():
    """Part 3: out/ticket_predictions.csv and out/network_report.json."""
    import csv
    p = os.path.join(OUT, "ticket_predictions.csv")
    problems = []
    if not os.path.exists(p):
        return _report(["out/ticket_predictions.csv does not exist yet. Run your saving cell."], "")
    with open(p) as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(DATA, "kittiwake_tickets_unlabeled.csv")) as f:
        ids = {r["ticket_id"] for r in csv.DictReader(f)}
    got = {r.get("ticket_id"): r.get("department") for r in rows}
    if set(got) != ids:
        problems.append(f"the file has {len(got)} distinct ticket ids; it needs all {len(ids)} unlabelled tickets.")
    bad = {v for v in got.values()} - {"billing", "network", "device", "account"}
    if bad:
        problems.append(f"departments must be billing, network, device or account; found {sorted(bad)[:3]}.")
    rep, err = _read("network_report.json")
    if err:
        problems.append(err)
    else:
        if not isinstance(rep.get("hidden"), int) or rep["hidden"] < 1:
            problems.append("network_report.json needs 'hidden', the size of your hidden layer (at least 1).")
        v = rep.get("validation_macro_f1")
        if not isinstance(v, (int, float)) or not 0 <= v <= 1:
            problems.append("network_report.json needs 'validation_macro_f1', measured on tickets you held back.")
    return _report(problems, "The files are complete. The checkpoint scores the predictions against the "
                             "departments you have not seen; your own validation score is your best estimate of it.")
