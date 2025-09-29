# nsga_runner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# pymoo in pure-python mode works even if C-extensions aren't compiled
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from LINKS.Optimization._Tools import Tools
from LINKS.Optimization._Tools import PreprocessedBatch as _PreBatch

# Group seeds by number of joints (n = x0.shape[0]) to reduce first-time JAX compiles.
def group_seeds_by_joint_count(seeds):
    groups = {}
    for mech in seeds:
        n = int(mech["x0"].shape[0])
        groups.setdefault(n, []).append(mech)
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))  # stable, ascending by n

def _warmup_tools_for_group(tools, mech, target_curve, pre):
    # One dummy eval to trigger compilation for this (n, max_size, timesteps) shape
    try:
        X_dummy = [mech["x0"]]  # list of (n,2)
        distances, materials = tools(X_dummy, pre, target_curve, target_idx=None) \
            if tools.material else (tools(X_dummy, pre, target_curve, target_idx=None), None)
        _ = float(np.asarray(distances)[0])
    except Exception:
        pass


@dataclass
class NSGAConfig:
    pop_size: int = 128
    n_gen: int = 100
    # variation operators (pymoo defaults are fine; these are standard)
    eta_cx: float = 15.0
    prob_cx: float = 0.9
    eta_mut: float = 20.0
    prob_mut: Optional[float] = None  # if None -> 1/n_var
    # execution
    device: str = "gpu"
    max_size: int = 20
    timesteps: int = 200
    n_workers_eval: int = 0  # 0/None = single thread; >0 to parallelize seeds
    seed: int = 1234
    scaled_distance: bool = False
    include_material: bool = True


class _SeedProblem(Problem):
    """
    A pymoo Problem wrapper that optimizes x0 for a *fixed* mechanism topology.
    Uses LINKS Tools in batched mode with a preprocessed bundle for speed.
    """

    def __init__(self, mech: Dict[str, Any], target_curve: np.ndarray, tools: Tools):
        # decision vars: positions (n,2) flattened into length 2n with bounds [0,1]
        self.n = mech["x0"].shape[0]
        self.mech = mech
        self.target_curve = target_curve
        self.tools = tools

        # Preprocess topology once (As, node_types, mappings, etc.)
        pre = tools.get_preprocessed(
            x0s=[mech["x0"]],
            edges=[mech["edges"]],
            fixed_joints=[mech["fixed_joints"]],
            motors=[mech["motor"]],
        )
        self.pre = pre  # LINKS PreprocessedBatch

        super().__init__(n_var=self.n * 2,
                         n_obj=2,
                         n_constr=0,
                         xl=0.0,
                         xu=1.0,
                         elementwise_evaluation=False)

    def _evaluate(self, X: np.ndarray, out: Dict, *args, **kwargs):
        P = X.shape[0]
        pops = [X[i].reshape(self.n, 2) for i in range(P)]

        # Expand the single-mechanism preprocessed bundle to match population size P
        As_rep         = np.repeat(self.pre.As,        P, axis=0)
        node_types_rep = np.repeat(self.pre.node_types, P, axis=0)
        motors_rep     = self.pre.motors   * P
        orders_rep     = self.pre.orders   * P
        mappings_rep   = self.pre.mappings * P
        valid_rep      = np.ones(P, dtype=bool)

        preP = _PreBatch(As_rep, node_types_rep, motors_rep, orders_rep, mappings_rep, valid_rep, is_single=False)

        if self.tools.material:
            distances, materials = self.tools(pops, preP, self.target_curve, target_idx=None)
            F = np.column_stack([np.asarray(distances), np.asarray(materials)])
        else:
            distances = self.tools(pops, preP, self.target_curve, target_idx=None)
            F = np.column_stack([np.asarray(distances), np.zeros_like(distances)])

        #fix one of the runtime errors
        F = np.nan_to_num(F, nan=1e9, posinf=1e9, neginf=1e9)
        F[:, 0] = np.maximum(F[:, 0], 0.0)  # distance >= 0
        F[:, 1] = np.maximum(F[:, 1], 0.0)  # material >= 0
        out["F"] = F


def _run_nsga_for_seed(mech: Dict[str, Any], target_curve: np.ndarray, cfg: NSGAConfig, rng: np.random.Generator):
    """
    Run NSGA-II for a single seed topology; return a dict with:
      {
        "mech": mech,
        "X": (P_all, 2n),
        "F": (P_all, 2)
      }
    Only store last generation due to history error
    """
    # independent Tools per call to avoid accidental cross-thread device state
    tools = Tools(
        timesteps=cfg.timesteps,
        max_size=cfg.max_size,
        material=cfg.include_material,
        scaled=cfg.scaled_distance,
        device=cfg.device,
    )
    tools.compile()

    problem = _SeedProblem(mech, target_curve, tools)

     # ---- JAX warm-up for Tools on this topology (avoids long silent first eval) ----
    try:
        # one dummy evaluation to trigger compilation with correct shapes
        P = 1
        X_dummy = np.repeat(problem.mech["x0"][None, ...], P, axis=0)  # shape (1, n, 2)
        distances, materials = tools(
            X_dummy, problem.pre, problem.target_curve, target_idx=None
        )
        # no-op use to silence linters
        _ = float(np.asarray(distances)[0])
        _ = float(np.asarray(materials)[0])
    except Exception:
        # warm-up is best-effort; ignore if anything odd
        pass

    # operators
    cx  = SBX(eta=cfg.eta_cx, prob=cfg.prob_cx)
    # If cfg.prob_mut is None, use 1/n_var per NSGA convention
    prob_mut = cfg.prob_mut if cfg.prob_mut is not None else 1.0 / problem.n_var
    mut = PM(eta=cfg.eta_mut, prob=prob_mut)
    sampling = FloatRandomSampling()

    algo = NSGA2(pop_size=cfg.pop_size, sampling=sampling, crossover=cx, mutation=mut, eliminate_duplicates=True)
    termination = get_termination("n_gen", cfg.n_gen)

    # set RNG for reproducibility - use different seeds for numpy and pymoo
    np_state = np.random.get_state()
    try:
        # Generate two different seeds from the provided RNG
        np_seed = rng.integers(0, 2**32 - 1, dtype=np.uint32)
        pymoo_seed = rng.integers(0, 2**32 - 1)
        np.random.seed(np_seed)
        res = minimize(problem, algo, termination, seed=pymoo_seed, save_history=False, verbose=False)
    finally:
        np.random.set_state(np_state)

    # Gather last generation
    X_fin = res.X
    F_fin = res.F

    return {"mech": mech, "X": X_fin, "F": F_fin}


def run_nsga_for_curve(
    seeds: List[Dict[str, Any]],
    target_curve: np.ndarray,
    cfg: NSGAConfig,
) -> List[Dict[str, Any]]:
    """
    Runs NSGA-II per seed topology, but batches seeds by joint count (n) to
    amortize JAX/XLA compiles. Returns a list of dicts (one per seed): {mech, X, F}.
    """
    rng = np.random.default_rng(cfg.seed)
    groups = group_seeds_by_joint_count(seeds)

    results: List[Dict[str, Any]] = []

    for n, group in groups.items():
        # Prepare a Tools instance to warm-up once for this joint count
        tools = Tools(
            timesteps=cfg.timesteps,
            max_size=cfg.max_size,
            material=cfg.include_material,
            scaled=cfg.scaled_distance,
            device=cfg.device,
        )
        tools.compile()

        # Build a preprocessed bundle for the FIRST mech in this group (topology fixed per seed)
        # We'll only use this for warm-up; the actual run uses its own Tools/Problem per seed.
        mech0 = group[0]
        # Reuse _SeedProblem’s preprocessing path
        tmp_problem = _SeedProblem(mech0, target_curve, tools)
        # Warm-up once for this shape (n,2)
        _warmup_tools_for_group(tools, mech0, target_curve, tmp_problem.pre)

        # Now actually run NSGA per seed (optionally parallelize inside the group)
        if cfg.n_workers_eval and cfg.n_workers_eval > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=cfg.n_workers_eval) as ex:
                # Create a separate RNG for each mechanism to ensure diversity
                futs = [ex.submit(_run_nsga_for_seed, mech, target_curve, cfg,
                                 np.random.default_rng(rng.integers(0, 2**32 - 1)))
                       for mech in group]
                for f in as_completed(futs):
                    results.append(f.result())
        else:
            for mech in group:
                # Create a separate RNG for each mechanism to ensure diversity
                mech_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                results.append(_run_nsga_for_seed(mech, target_curve, cfg, mech_rng))

        print(f"[NSGA] group n={n:>2d}  seeds={len(group)}  done", flush=True)

    return results


def pairwise_distances(Z: np.ndarray) -> np.ndarray:
    """Compute full Euclidean distance matrix for rows of Z."""
    diff = Z[:, None, :] - Z[None, :, :]
    dmat = np.sqrt((diff ** 2).sum(-1))
    return dmat

def cap_nsga_candidates(
    results_per_seed: List[Dict[str, Any]],
    cap: int = 1000
) -> List[Dict[str, Any]]:
    """
    Merge final NSGA populations across seeds and trim to <= cap using
    non-dominated sorting + crowding on the last partially included front.
    Returns a flat list of {"mech", "x0", "distance", "material"} dicts.
    """
    # 1) Flatten everything
    bag: List[Dict[str, Any]] = []
    for R in results_per_seed:
        mech = R["mech"]
        n = mech["x0"].shape[0]
        X = R["X"]
        F = R["F"]
        for i in range(X.shape[0]):
            bag.append({
                "mech": mech,
                "x0": X[i].reshape(n, 2),
                "distance": float(F[i, 0]),
                "material": float(F[i, 1]),
            })

    if len(bag) <= cap:
        return bag

    # 2) Build objective matrix
    F = np.array([[c["distance"], c["material"]] for c in bag])

    # 3) Non-dominated sorting
    nds = NonDominatedSorting(method="fast_non_dominated_sort")
    fronts = nds.do(F, only_non_dominated_front=False)

    # 4) Fill until cap
    selected_idx: List[int] = []
    for front in fronts:
        if len(selected_idx) + len(front) <= cap:
            selected_idx.extend(front.tolist())
        else:
            # Need to select a subset from this 'front' to fill to cap
            remaining = cap - len(selected_idx)
            # Crowding distance on this front
            subF = F[front]
            # simple crowding: normalize per column, then compute distances in objective space
            # (pymoo has a crowding calc in algorithms, but this lightweight version suffices)
            # Normalize:
            mins = subF.min(axis=0)
            maxs = subF.max(axis=0)
            span = np.where(maxs > mins, maxs - mins, 1.0)
            Z = (subF - mins) / span
            # crowding via distance to nearest neighbors in normalized space
            # (higher is better)
            dmat = pairwise_distances(Z)
            np.fill_diagonal(dmat, np.inf)
            crowd = np.min(dmat, axis=1)
            # pick the 'remaining' with largest crowding (diversity)
            pick_local = np.argsort(-crowd)[:remaining]
            selected_idx.extend(front[pick_local].tolist())
            break

    selected = [bag[i] for i in selected_idx]
    return selected
