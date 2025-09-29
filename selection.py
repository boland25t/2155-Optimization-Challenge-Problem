from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
try:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    _HAS_PYMOO = True
except Exception:
    _HAS_PYMOO = False

def _pairwise_dist(Z: np.ndarray) -> np.ndarray:
    diff = Z[:, None, :] - Z[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))

def _simple_nd_front(F: np.ndarray) -> List[int]:
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dominated[i] = False
        keep[dominated] = False
    return np.where(keep)[0].tolist()

def _as_index_list(idxs) -> list[int]:
    arr = np.atleast_1d(idxs).astype(int)
    return arr.tolist()

def select_for_submission(
    refined: List[Dict[str, Any]],
    cap: int = 1000,
    max_distance: Optional[float] = None,
    max_material: Optional[float] = None,
    require_feasible: bool = True,
) -> List[Dict[str, Any]]:
    """
    1) (Optional) Feasibility filter: keep distance<=max_distance and material<=max_material.
    2) Non-dominated sort on remaining; fill fronts up to `cap`.
    3) If a front must be truncated, pick the most diverse (nearest-neighbor distance in normalized space).
    4) If no feasible points and require_feasible=True, fall back to best-by-distance (trim to cap).
    """
    # --- Step 1: feasibility mask ---
    D = np.array([r["distance"] for r in refined], dtype=float)
    M = np.array([r["material"] for r in refined], dtype=float)

    mask = np.ones(len(refined), dtype=bool)
    if max_distance is not None:
        mask &= D <= max_distance
    if max_material is not None:
        mask &= M <= max_material

    pool_idx = np.where(mask)[0]
    if len(pool_idx) == 0:
        if require_feasible:
            # fallback: sort all by distance (or penalized distance) and take up to cap
            order = np.argsort(D)
            chosen = order[:min(cap, len(order))].tolist()
            return [refined[i] for i in chosen]
        else:
            pool_idx = np.arange(len(refined))

    pool = [refined[i] for i in pool_idx]
    F = np.column_stack([D[pool_idx], M[pool_idx]])

    # If everything fits, still return ND set (remove dominated feasible points)
    if len(pool) <= cap:
        if _HAS_PYMOO:
            nd = _as_index_list(NonDominatedSorting(method="fast_non_dominated_sort").do(F, only_non_dominated_front=True))
        else:
            nd = _simple_nd_front(F)
        return [pool[i] for i in nd]

    # Build fronts
    if _HAS_PYMOO:
        fronts = NonDominatedSorting(method="fast_non_dominated_sort").do(F, only_non_dominated_front=False)
    else:
        remaining = list(range(F.shape[0]))
        fronts = []
        while remaining:
            sub = F[remaining]
            nd_local = _simple_nd_front(sub)
            front = [remaining[i] for i in nd_local]
            fronts.append(np.array(front, dtype=int))
            remaining = [i for i in remaining if i not in front]

    selected_rel: List[int] = []
    for front in fronts:
        if len(selected_rel) + len(front) <= cap:
            selected_rel.extend(front.tolist())
        else:
            k = cap - len(selected_rel)
            subF = F[front]
            # normalize
            mins = subF.min(axis=0)
            spans = np.where(subF.max(axis=0) > mins, subF.max(axis=0)-mins, 1.0)
            Z = (subF - mins) / spans
            # pick most diverse
            Dmat = _pairwise_dist(Z)
            np.fill_diagonal(Dmat, np.inf)
            crowd = np.min(Dmat, axis=1)
            take = np.argsort(-crowd)[:k]
            selected_rel.extend(front[take].tolist())
            break

    # Map back to absolute indices of 'refined'
    selected_abs = [pool_idx[i] for i in selected_rel]
    return [refined[i] for i in selected_abs]
