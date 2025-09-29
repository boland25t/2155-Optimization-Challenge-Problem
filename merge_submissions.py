# merge_submissions.py
import argparse
from typing import Dict, List, Any
import numpy as np

# --- LINKS imports ---
try:
    from LINKS.Optimization._Tools import Tools
except Exception:
    from _Tools import Tools  # fallback if your PYTHONPATH already points to LINKS


# --------------------- Utilities ---------------------

def mech_key(m: Dict[str, Any]) -> bytes:
    """Exact fingerprint across runs (include target_joint)."""
    return (
        np.asarray(m["x0"], dtype=np.float64).tobytes()
        + np.asarray(m["edges"], dtype=np.int64).tobytes()
        + np.asarray(m["fixed_joints"], dtype=np.int64).tobytes()
        + np.asarray(m["motor"], dtype=np.int64).tobytes()
        + np.int64(m.get("target_joint", m["x0"].shape[0] - 1)).tobytes()
    )


def evaluate_batch(
    tools: Tools, items: List[Dict[str, Any]], target_curve: np.ndarray, batch_size: int = 256
) -> np.ndarray:
    """Return F = [[distance, material], ...] in the same order as `items`."""
    F = np.zeros((len(items), 2), dtype=float)
    for s in range(0, len(items), batch_size):
        chunk = items[s : s + batch_size]
        X = [np.asarray(m["x0"]) for m in chunk]
        E = [np.asarray(m["edges"]) for m in chunk]
        FJ = [np.asarray(m["fixed_joints"]) for m in chunk]
        M  = [np.asarray(m["motor"]) for m in chunk]
        d, mat = tools(X, E, FJ, M, target_curve, target_idx=None)
        F[s : s + len(chunk), 0] = np.asarray(d, float).reshape(-1)
        F[s : s + len(chunk), 1] = np.asarray(mat, float).reshape(-1)
    # sanitize for safety
    F = np.nan_to_num(F, nan=1e9, posinf=1e9, neginf=1e9)
    F[:, 0] = np.maximum(F[:, 0], 0.0)
    F[:, 1] = np.maximum(F[:, 1], 0.0)
    return F


def non_dominated_mask(F: np.ndarray) -> np.ndarray:
    """Boolean mask for non-dominated points (min-min, 2D). O(n^2), fine for <=1000s."""
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = (F <= F[i]).all(axis=1) & (F < F[i]).any(axis=1)
        dom[i] = False
        keep[dom] = False
    return keep


def hv2d_min(points: np.ndarray, ref: np.ndarray) -> float:
    """
    Dominated hypervolume in 2D for minimization.
    points: (k,2) [f1=distance, f2=material], ref: (2,) (worse-than-worst reference).
    """
    if points.size == 0:
        return 0.0
    # Sort by f1 ascending, break ties by f2
    P = points[np.lexsort((points[:, 1], points[:, 0]))]
    # running min on f2 for the "staircase"
    f1 = np.r_[P[:, 0], ref[0]]
    f2 = np.minimum.accumulate(np.r_[P[:, 1], ref[1]][::-1])[::-1]
    area = 0.0
    for i in range(len(P)):
        w = (f1[i + 1] - f1[i])
        h = (ref[1] - f2[i])
        if w > 0 and h > 0:
            area += w * h
    return float(area)


def hv_greedy_select(F: np.ndarray, cap: int, ref_pad: float = 1.05) -> List[int]:
    """
    Greedy build an index set (<=cap) that maximizes 2D HV wrt a ref slightly beyond worst.
    Always filters to the ND set first.
    """
    # 1) keep only ND candidates
    nd_mask = non_dominated_mask(F)
    idx = np.where(nd_mask)[0]
    Fnd = F[idx]
    if len(idx) <= cap:
        return idx.tolist()

    # 2) greedy HV selection on the ND pool
    ref = Fnd.max(axis=0) * ref_pad
    chosen: List[int] = []
    chosen_F = np.empty((0, 2), float)
    remaining = list(range(Fnd.shape[0]))
    current_hv = 0.0

    for _ in range(cap):
        best_gain = -1.0
        best_j = None
        # simple greedy; for 2D and <=1000 this is fast enough
        for j in remaining:
            cand_F = np.vstack([chosen_F, Fnd[j : j + 1]])
            gain = hv2d_min(cand_F, ref) - current_hv
            if gain > best_gain:
                best_gain = gain
                best_j = j
        if best_j is None:
            break
        chosen.append(idx[best_j])
        chosen_F = np.vstack([chosen_F, Fnd[best_j]])
        current_hv += best_gain
        remaining.remove(best_j)

    return chosen


# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Merge multiple submissions into a best-of ND/HV set.")
    ap.add_argument("submissions", nargs="+", help="Paths to submission.npy files to merge")
    ap.add_argument("--curves", default="target_curves.npy", help="Path to target_curves.npy")
    ap.add_argument("--out", default="merged_submission.npy", help="Output merged submission path")
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--max_size", type=int, default=20)
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--cap", type=int, default=1000, help="Max mechanisms per problem")
    # Leave feasibility OFF by default; you can enforce later if desired
    ap.add_argument("--max_distance", type=float, default=None)
    ap.add_argument("--max_material", type=float, default=None)
    args = ap.parse_args()

    curves = np.load(args.curves, allow_pickle=False)
    keys = [f"Problem {i}" for i in range(1, len(curves) + 1)]

    # Load all submissions
    subs: List[Dict[str, List[Dict[str, Any]]]] = []
    for p in args.submissions:
        d = np.load(p, allow_pickle=True).item()
        subs.append(d)

    # Merge + de-dup per problem
    merged: Dict[str, List[Dict[str, Any]]] = {k: [] for k in keys}
    for k in keys:
        seen = set()
        bag: List[Dict[str, Any]] = []
        for sub in subs:
            for m in sub.get(k, []):
                tj = int(m.get("target_joint", m["x0"].shape[0] - 1))
                mm = {
                    "x0": np.asarray(m["x0"], float),
                    "edges": np.asarray(m["edges"], np.int64),
                    "fixed_joints": np.asarray(m["fixed_joints"], np.int64),
                    "motor": np.asarray(m["motor"], np.int64),
                    "target_joint": tj,
                }
                h = mech_key(mm)
                if h not in seen:
                    seen.add(h)
                    bag.append(mm)
        merged[k] = bag

    tools = Tools(
        timesteps=args.timesteps, max_size=args.max_size, material=True, scaled=False, device=args.device
    )
    tools.compile()

    final: Dict[str, List[Dict[str, Any]]] = {k: [] for k in keys}
    print("\n=== Merging summary ===")
    for i, k in enumerate(keys):
        items = merged[k]
        if not items:
            print(f"{k}: EMPTY (will score 0)")
            continue

        # Evaluate all candidates for this problem
        F = evaluate_batch(tools, items, curves[i])

        # Optional feasibility mask BEFORE selection (default: None -> no mask)
        mask = np.ones(len(items), dtype=bool)
        if args.max_distance is not None:
            mask &= (F[:, 0] <= args.max_distance)
        if args.max_material is not None:
            mask &= (F[:, 1] <= args.max_material)

        pool_idx = np.where(mask)[0]

        # If we don't exceed the cap, KEEP EVERYTHING in the pool (no selection)
        if len(pool_idx) <= args.cap:
            chosen_idx = pool_idx.tolist()
        else:
            # Over cap: select ND/HV-greedy from the pool
            Fpool = F[pool_idx]
            chosen_rel = hv_greedy_select(Fpool, cap=args.cap)
            chosen_idx = [int(pool_idx[j]) for j in chosen_rel]

        selected = [items[j] for j in chosen_idx]
        final[k] = selected

        # Recap (robust identity mapping not needed since we kept indices)
        best_d = float(np.min(F[chosen_idx, 0])) if len(chosen_idx) else float("inf")
        nd_est = int(non_dominated_mask(F).sum())
        print(f"{k}: in={len(items)}  ND~={nd_est}  selected={len(selected)}  best_distance≈{best_d:.4f}")

    np.save(args.out, final, allow_pickle=True)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
