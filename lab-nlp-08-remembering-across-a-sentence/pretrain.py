"""pretrain: train 08_03's LSTM, BiLSTM and bag once, in the background, at session start.

workshop/setup.d/10-prepare.sh runs this at low priority while the learner reads
the chapter, so 08_03 can load the three models in a second instead of spending
several minutes of the session training them. The training is exactly what
08_03 would otherwise do itself (seed 0, 3 epochs); if the files are
not there yet, 08_03 trains the models itself and says so. Each file is written
atomically, so a half-written model is never loaded.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import headlines  # noqa: E402

MODELS = {
    "lstm": lambda: headlines.SequenceClassifier(cell="lstm"),
    "bilstm": lambda: headlines.SequenceClassifier(cell="lstm", bidirectional=True),
    "bag": lambda: headlines.BagClassifier(),
}


def path(name):
    return os.path.join(HERE, "out", f".pretrained_{name}.pt")


def trained(name, X, y):
    """The 08_03 model called name: loaded if the background job saved it, else trained now."""
    model = MODELS[name]()
    if os.path.exists(path(name)):
        try:
            model.load_state_dict(torch.load(path(name)))
            print(f"{name}: loaded, trained in the background when the session started")
            return model
        except Exception:
            model = MODELS[name]()
    print(f"{name}: not trained in the background yet, so training it now")
    torch.manual_seed(0)
    headlines.train(model, X, y, epochs=3)
    return model


if __name__ == "__main__":
    torch.set_num_threads(2)
    d = headlines.load_split()
    for name, make in MODELS.items():
        if os.path.exists(path(name)):
            continue
        torch.manual_seed(0)
        model = make()
        headlines.train(model, d["X_train"], d["y_train"], epochs=3)
        tmp = path(name) + ".tmp%d" % os.getpid()
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, path(name))
        print("saved", path(name))
