import numpy as np
from collections import Counter

sub = np.load("submission.npy", allow_pickle=True).item()

for prob, mechs in sub.items():
    counts = [m["x0"].shape[0] for m in mechs]
    ctr = Counter(counts)
    print(f"{prob}:")
    print("  total mechanisms:", len(counts))
    print("  unique node counts:", sorted(ctr.keys()))
    print("  most common:", ctr.most_common(5))
    print("  min nodes:", min(counts), " max nodes:", max(counts))
    print()
