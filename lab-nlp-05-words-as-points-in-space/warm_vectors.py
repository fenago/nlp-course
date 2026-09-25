"""warm_vectors.py: train 05_03's product vectors in the background.

workshop/setup.d/10-prepare.sh starts this at low priority when the session
starts, so that by the time you reach 05_03 its training cell loads the vectors
in under a second instead of spending a minute or two training them. It writes
out/product_vectors.npz atomically (w2vtools.product_vectors writes a temporary
file and renames it). If it has not finished when you get there, or fails, the
notebook trains the vectors itself; nothing depends on this. Its log is
out/.warm_vectors.log.
"""
import time

import w2vtools

start = time.time()
train_baskets, _, _ = w2vtools.load_baskets()
w2vtools.product_vectors(train_baskets, verbose=True)
print(f"done in {time.time() - start:.0f} s")
