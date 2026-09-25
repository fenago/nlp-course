"""warm_embeddings: fill the embedding cache while the learner reads the chapter.

Started in the background by workshop/setup.d/10-prepare.sh, at low priority.
It embeds exactly the texts the notebooks ask for, smallest batch first, through
the same clftools.embed() the notebooks use, so every vector lands where they
will look. It writes warm.log and, when everything is cached, READY, both under
out/.embeddings/. Nothing depends on it: a notebook that gets there first
computes what is missing itself.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out", ".embeddings")


def log(msg, t0):
    print(f"{time.time() - t0:7.1f}s {msg}", flush=True)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(os.path.join(HERE, "data", "kittiwake_reviews_unlabelled.csv")):
        os.system(f"bash {HERE}/prepare.sh >/dev/null 2>&1")
    try:
        import clftools
        sentences = clftools.load_sentences()
        labelled, unlabelled = clftools.load_reviews()
    except Exception as e:  # the data is missing: say so once and stop
        log(f"data not found ({e}); nothing to warm", t0)
        return
    batches = [
        ("04_01 example phrases", ["great phone", "terrible phone", "excellent handset"]),
        ("20 labelled Kittiwake reviews", list(labelled["text"])),
        ("3,000 UCI sentences", list(sentences["text"])),
        ("380 unlabelled Kittiwake reviews", list(unlabelled["text"])),
    ]
    for name, texts in batches:
        clftools.embed(texts)
        log(f"cached: {name} ({len(texts)})", t0)
    open(os.path.join(OUT, "READY"), "w").write(f"{time.time() - t0:.0f} s\n")
    log("done", t0)


if __name__ == "__main__":
    main()
