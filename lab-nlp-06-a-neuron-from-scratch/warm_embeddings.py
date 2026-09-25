#!/usr/bin/env python3
"""warm_embeddings.py: embed the 800 Kittiwake tickets before Part 3 needs them.

workshop/setup.d/10-prepare.sh starts this in the background, at low priority,
when the session starts. It writes through neurontools.embed(), the same code
and the same cache Part 3 uses, and every file is written atomically, so it is
safe for a notebook to run at the same time: whichever gets to a ticket first
computes it. Progress goes to out/.embeddings/warm.log; out/.embeddings/READY
appears at the end.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import neurontools  # noqa: E402


def main():
    t0 = time.time()
    log = lambda m: print(f"{time.time() - t0:6.1f}s {m}", flush=True)  # noqa: E731
    data = os.path.join(HERE, "data")
    labelled = os.path.join(data, "kittiwake_tickets.csv")
    if not os.path.exists(labelled):
        subprocess.run(["bash", os.path.join(HERE, "prepare.sh")], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False)
    if not os.path.exists(labelled):
        log("the data is not in ~/exercises/data yet; nothing warmed")
        return
    import pandas as pd
    texts = list(pd.read_csv(labelled)["text"]) + \
        list(pd.read_csv(os.path.join(data, "kittiwake_tickets_unlabeled.csv"))["text"])
    log(f"embedding {len(texts)} tickets")
    neurontools.embed(texts)
    log("done")
    open(os.path.join(HERE, "out", ".embeddings", "READY"), "w").write(f"{len(texts)} tickets\n")


if __name__ == "__main__":
    main()
