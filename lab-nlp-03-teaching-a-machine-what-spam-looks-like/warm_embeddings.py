#!/usr/bin/env python3
"""warm_embeddings.py: fill the embedding cache before the learner needs it.

workshop/setup.d/10-prepare.sh starts this in the background, at low priority,
when the session starts. It embeds exactly the texts Lab 03 asks for, so that by
the time a learner has read the chapter pages the notebooks find them cached:

  - page 01's zero-shot block: its three tickets and the four team descriptions;
  - 03_03 section 4: the stratified 1,000 training and 600 test messages and the
    three comparison sentences;
  - 03_04 and page 09: all 800 Kittiwake tickets (any subset, a cross-validation
    fold for instance, is then cached too, because the cache is one file per text).

It writes through spamtools.embed(), the same code and the same cache the
notebooks use, and every file is written atomically, so it is safe for a
notebook to run at the same time: whichever gets to a text first computes it.
Progress goes to out/.embeddings/warm.log; out/.embeddings/READY appears at the end.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import spamtools  # noqa: E402

TEAMS = {
    "billing": "a question or complaint about a bill, a charge, a payment or a refund",
    "network": "a problem with signal, coverage, dropped calls, texts or mobile data",
    "device": "a problem with a phone, handset, SIM card, screen, battery or delivery of a device",
    "account": "a request to change account details, log in, add or remove users, or cancel a contract",
}
PAGE01 = ["My calls keep dropping in Gullhaven.", "You charged me twice this month.",
          "Why was I charged roaming fees when I never left Saltmoor?"]
THREE = ["You have won a free prize", "You have been selected for a complimentary reward",
         "Are you coming to dinner tonight?"]


def main():
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:6.1f}s {m}", flush=True)  # noqa: E731
    # setup.d runs prepare.sh first, so the data is normally here already. If it
    # is not, run prepare.sh once; if it is still missing, stop quietly: the
    # notebooks compute whatever they need themselves.
    if not os.path.exists(spamtools.SMS):
        import subprocess
        subprocess.run(["bash", os.path.join(HERE, "prepare.sh")], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
    if not os.path.exists(spamtools.SMS):
        log("the data is not in ~/exercises/data yet; nothing warmed")
        return
    X_train, X_test, y_train, y_test = spamtools.split(spamtools.load_sms())
    X_small, _, _, _ = train_test_split(X_train, y_train, train_size=1000, random_state=42, stratify=y_train)
    X_check, _, _, _ = train_test_split(X_test, y_test, train_size=600, random_state=42, stratify=y_test)
    tickets = pd.read_csv(os.path.join(HERE, "data", "kittiwake_tickets.csv"))["text"]
    unlabeled = pd.read_csv(os.path.join(HERE, "data", "kittiwake_tickets_unlabeled.csv"))["text"]
    # Smallest first, so the first page's block is ready soonest.
    for name, texts in [("page 01 and team descriptions", PAGE01 + list(TEAMS.values())),
                        ("03_03 comparison sentences", THREE),
                        ("800 Kittiwake tickets", list(tickets) + list(unlabeled)),
                        ("1,000 training messages", list(X_small)),
                        ("600 test messages", list(X_check))]:
        spamtools.embed(texts)
        log(f"cached: {name} ({len(texts)})")
    ready = os.path.join(HERE, "out", ".embeddings", "READY")
    with open(ready, "w") as f:
        f.write(f"warmed in {time.time() - t0:.0f} s\n")
    log("done")


if __name__ == "__main__":
    main()
