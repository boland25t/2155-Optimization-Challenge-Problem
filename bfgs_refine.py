# bfgs_refine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from math import inf
import time
from tqdm.auto import tqdm

try:
    from LINKS.Optimization._DifferentiableTools import DifferentiableTools
    from LINKS.Optimization._Tools import Tools, PreprocessedBatch as _PreBatch
    from LINKS.Optimization._DifferentiableTools import PreprocessedBatch as DPreBatch
except Exception:
    from _DifferentiableTools import DifferentiableTools
    from _Tools import Tools, PreprocessedBatch as _PreBatch

@dataclass
class BFGSConfig:
    timesteps: int = 200
    max_size: int = 20
    device: str = "gpu"
    material_threshold: float = 10.0   # M
    penalty_lambda: float = 5.0        # λ
    maxiter: int = 200
    n_workers: int = 0                 # 0/None = single thread
    seed: Optional[int] = 123

"""
def _objective_and_grad(dtools: DifferentiableTools,
                        pre: _PreBatch,
                        target_curve: np.ndarray,
                        x0: np.ndarray,
                        M: float,
                        lam: float) -> Tuple[float, np.ndarray]:
    # value + grads
    distances, materials, gD, gM = dtools([x0], pre, target_curve, target_idx=None)
    D   = float(distances)
    mat = float(materials)

    # --- keep gradients from collapsing due to NaNs/Infs ---
    gD = np.asarray(gD)
    gM = np.asarray(gM)
    gD = np.where(np.isfinite(gD), gD, 0.0)
    gM = np.where(np.isfinite(gM), gM, 0.0)

    # soft penalty
    over = max(0.0, mat - M)
    f = D + lam * (over * over)

    # combine grads; fall back to distance grad if combined is ~0
    grad = gD if over == 0.0 else (gD + (2.0 * lam * over) * gM)
    if not np.isfinite(grad).all() or np.linalg.norm(grad) < 1e-12:
        grad = gD

    return f, grad
"""

def _objective_and_grad(dtools: DifferentiableTools,
                        pre: _PreBatch,
                        target_curve: np.ndarray,
                        x0: np.ndarray,
                        M: float,
                        lam: float) -> Tuple[float, np.ndarray]:
    # value + grads
    distances, materials, gD, gM = dtools([x0], pre, target_curve, target_idx=None)
    D   = float(distances)
    mat = float(materials)

    # sanitize grads
    gD = np.asarray(gD); gM = np.asarray(gM)
    gD = np.where(np.isfinite(gD), gD, 0.0)
    gM = np.where(np.isfinite(gM), gM, 0.0)

    over = mat - M
    # scalar objective: distance + penalty if over budget
    f = D + (lam * over * over if over > 0.0 else 0.0)

    if over > 0.0:
        # active constraint: push material down
        grad = gD + (2.0 * lam * over) * gM
    else:
        # within budget: remove the part of gD that would increase material
        gDf = gD.reshape(-1)
        gMf = gM.reshape(-1)
        dot = float(np.dot(gDf, gMf))
        if dot > 0.0:
            denom = float(np.dot(gMf, gMf)) + 1e-12
            proj = (dot / denom) * gM
            grad = gD - proj
        else:
            grad = gD

    # last-resort safeguard: if projected grad collapses, fall back to distance grad
    if not np.isfinite(grad).all() or np.linalg.norm(grad) < 1e-12:
        grad = gD

    return f, grad

def _refine_one(candidate: Dict[str, Any],
                target_curve: np.ndarray,
                cfg: BFGSConfig) -> Dict[str, Any]:
    mech = candidate["mech"]
    n = mech["x0"].shape[0]

    # Preprocess once for this topology
    dtools = DifferentiableTools(timesteps=cfg.timesteps, max_size=cfg.max_size,
                                 material=True, device=cfg.device)
    dtools.compile()

    tools = Tools(timesteps=cfg.timesteps, max_size=cfg.max_size,
                  material=True, device=cfg.device)
    
    # Use the underlying solver to get all arrays, then instantiate the correct
    # PreprocessedBatch class from DifferentiableTools (not the Tools one).
    As_, x0s_, node_types_, motors, orders, mappings, valid = dtools.solver.preprocess(
        x0s=[mech["x0"]],
        edges=[mech["edges"]],
        fixed_nodes=[mech["fixed_joints"]],
        motors=[mech["motor"]],
    )

    # Build the preprocessed bundle expected by dtools._preproc_call
    # For one topology + one candidate, len==1 is fine.
    pre = DPreBatch(As_, node_types_, motors, orders, mappings, valid, is_single=False)

    x_init = candidate["x0"].copy().reshape(-1)  # (2n,)
    bounds = [(0.0, 1.0)] * (2 * n)

    # SciPy minimize (L-BFGS-B) – do a local import so module stays light if SciPy isn't present
    from scipy.optimize import minimize

    def fun(x_flat):
        x = x_flat.reshape(n, 2)
        f, g = _objective_and_grad(dtools, pre, target_curve, x, cfg.material_threshold, cfg.penalty_lambda)
        return f, g.reshape(-1)
    
    x_init = candidate["x0"].copy().reshape(-1)

    # small bounded jitter so we're not exactly on a flat/boundary point
    eps = 1e-3
    rng = np.random.default_rng((cfg.seed or 0))
    x_init = np.clip(x_init + rng.normal(0, eps, size=x_init.shape), 0.0, 1.0)

    res = minimize(
        fun=fun, x0=x_init, method="L-BFGS-B",
        jac=True, bounds=bounds,
        options={"maxiter": cfg.maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50}
    )

    x_ref = res.x.reshape(n, 2)

    # Evaluate final distance/material (cheap)
    d, m, _, _ = dtools([x_ref],
                    [mech["edges"]],
                    [mech["fixed_joints"]],
                    [mech["motor"]],
                    target_curve,
                    target_idx=None)
    d = float(d) if np.isscalar(d) else float(np.asarray(d)[0])
    m = float(m) if np.isscalar(m) else float(np.asarray(m)[0])

    return {
        "mech": mech,
        "x0": x_ref,
        "distance": d,
        "material": m,
        "success": bool(res.success),
        "nit": int(res.nit),
        "fun": float(res.fun)
    }

def refine_candidates(candidates: List[Dict[str, Any]],
                      target_curve: np.ndarray,
                      cfg: BFGSConfig) -> List[Dict[str, Any]]:
    # Optional: deterministic shuffle
    if cfg.seed is not None:
        rng = np.random.default_rng(cfg.seed)
        idx = np.arange(len(candidates))
        rng.shuffle(idx)
        candidates = [candidates[i] for i in idx]

    # --- One-time JAX compile warm-up for this curve ---
    if len(candidates) > 0:
        print("[BFGS] Compiling JAX kernels (first run can take a minute)...", flush=True)
        t0 = time.time()
        c0 = candidates[0]
        mech = c0["mech"]

        dtools = DifferentiableTools(timesteps=cfg.timesteps,
                                     max_size=cfg.max_size,
                                     material=True, device=cfg.device)
        dtools.compile()
        from LINKS.Optimization._DifferentiableTools import PreprocessedBatch as DPreBatch
        As_, x0s_, node_types_, motors, orders, mappings, valid = dtools.solver.preprocess(
            x0s=[mech["x0"]],
            edges=[mech["edges"]],
            fixed_nodes=[mech["fixed_joints"]],
            motors=[mech["motor"]],
        )
        pre = DPreBatch(As_, node_types_, motors, orders, mappings, valid, is_single=False)
        # dummy call to trigger compile
        _ = dtools([mech["x0"]], pre, target_curve, target_idx=None)
        print(f"[BFGS] JAX compile finished in {time.time()-t0:.1f}s", flush=True)

    if cfg.n_workers and cfg.n_workers > 1:
        out: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=cfg.n_workers) as ex:
            futs = [ex.submit(_refine_one, c, target_curve, cfg) for c in candidates]
            for f in tqdm(as_completed(futs), total=len(futs), desc="[BFGS] refining"):
                out.append(f.result())
    else:
        out = []
        for c in tqdm(candidates, desc="[BFGS] refining"):
            out.append(_refine_one(c, target_curve, cfg))


    # --- Quick printout summary ---
    out_sorted = sorted(out, key=lambda d: d["distance"])
    best = out_sorted[0]
    print(f"[BFGS refine] {len(out)} candidates refined "
          f"(best distance={best['distance']:.4f}, material={best['material']:.3f}, "
          f"nit={best['nit']}, success={best['success']})")

    return out
