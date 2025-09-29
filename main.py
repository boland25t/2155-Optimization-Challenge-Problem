# main.py
import os, time, numpy as np

from seeds import MechanismSeeder, SeederConfig
from nsga_runner import NSGAConfig, run_nsga_for_curve, cap_nsga_candidates  # or cap_by_distance
from bfgs_refine import BFGSConfig, refine_candidates
from selection import select_for_submission
from typing import Dict, List, Any
try:
    from LINKS.CP import make_empty_submission, evaluate_submission
except Exception:
    make_empty_submission = None
    evaluate_submission = None


TOP_K_PER_CURVE = 15
CAP_PER_CURVE   = 1000

def pack_one_curve_for_grader(refined_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Take a sorted-by-distance refined list and convert to the grader schema."""
    out = []
    for s in refined_list:
        mech = s["mech"]
        out.append({
            "x0":           s["x0"].astype(np.float64),
            "edges":        np.asarray(mech["edges"], dtype=np.int64),
            "fixed_joints": np.asarray(mech["fixed_joints"], dtype=np.int64),
            "motor":        np.asarray(mech["motor"], dtype=np.int64),
            # be explicit; don't rely on 'None' auto-pick
            "target_joint": int(mech.get("target_joint", mech["x0"].shape[0] - 1)),
        })
    return out

def optimize_one_curve(curve_idx, curve, s_cfg, n_cfg, b_cfg, base_seed=None):
    print(f"\n=== Curve {curve_idx} ===")
    t0 = time.time()

    # Create curve-specific configs with different seeds if base_seed provided
    if base_seed is not None:
        rng = np.random.default_rng(base_seed)
        # Create new configs with curve-specific seeds
        s_cfg = SeederConfig(**{**s_cfg.__dict__, 'seed': rng.integers(0, 2**32-1)})
        n_cfg = NSGAConfig(**{**n_cfg.__dict__, 'seed': rng.integers(0, 2**32-1)})
        b_cfg = BFGSConfig(**{**b_cfg.__dict__, 'seed': rng.integers(0, 2**32-1)})

    # 1) Seeds
    seeder = MechanismSeeder(s_cfg)
    seeds = seeder.generate_for_curve(curve)
    print(f"[Seeds] {len(seeds)} mechanisms")

    # 2) NSGA (final pops only), then cap
    per_seed = run_nsga_for_curve(seeds, curve, n_cfg)
    candidates = cap_nsga_candidates(per_seed, cap=CAP_PER_CURVE)
    print(f"[NSGA] capped candidates: {len(candidates)}")

    # 3) BFGS refine (with warm-up + tqdm already in the module)
    refined = refine_candidates(candidates, curve, b_cfg)
    refined.sort(key=lambda d: d["distance"])

    #sort to ND
    selected = select_for_submission(refined, cap=1000, max_distance=0.75, max_material=10.0, require_feasible=True)

    packed = pack_one_curve_for_grader(selected)

    best_sel = min(selected, key=lambda d: d["distance"]) if selected else None
    if best_sel:
        print(f"[Curve {curve_idx}] best dist={best_sel['distance']:.5f}  "
          f"material={best_sel['material']:.3f}  "
          f"selected={len(selected)}")
    else:
        print(f"[Curve {curve_idx}] WARNING: no feasible selections; falling back occurred.")
    return packed

if __name__ == "__main__":
    # (Optional) calmer JAX memory behavior:
    # os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".80")
    # os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    target_curves = np.load("target_curves.npy")   # shape (num_curves, 200, 2)

    # --- Configs in one place ---
    s_cfg = SeederConfig(
        n_seeds=32,           # seeds per curve
        min_nodes=6, max_nodes=20,
        fixed_probability=0.15,
        timesteps=200, device="gpu",
        preevaluate=True,
        seed=1234,
        warmup_randomizer=False,        # skip heavy warmup; NSGA does a tiny warmup now
        n_tests_per_mech=32, max_tries_per_mech=100
    )
    n_cfg = NSGAConfig(
        pop_size=160, n_gen=100,
        eta_cx=15.0, prob_cx=0.9,
        eta_mut=15.0, prob_mut=None,    # auto=1/n_var in our wrapper
        device="gpu", max_size=s_cfg.max_nodes, timesteps=s_cfg.timesteps,
        n_workers_eval=0,               # per-seed serial; bump if you want
        seed=4321,
        scaled_distance=False,
        include_material=True
    )
    b_cfg = BFGSConfig(
        timesteps=200, max_size=s_cfg.max_nodes, device="gpu",
        material_threshold=10.0, penalty_lambda=5.0,
        maxiter=200, n_workers=0, seed=999
    )

    # --- Save grader-ready file ---
    # build the submission dict with required keys
    if make_empty_submission is not None:
        submission: Dict[str, List[Dict[str, Any]]] = make_empty_submission()
    else:
        submission = {f"Problem {i}": [] for i in range(1, 7)}

    # Use a master seed to generate curve-specific seeds for better diversity
    master_seed = 42  # You can change this to any value for different runs
    master_rng = np.random.default_rng(master_seed)

    for i, curve in enumerate(target_curves):
        # Generate a unique base seed for each curve
        curve_base_seed = master_rng.integers(0, 2**32-1)
        packed = optimize_one_curve(i, curve, s_cfg, n_cfg, b_cfg, base_seed=curve_base_seed)   # <= returns LIST already
        # If your optimize_one_curve currently returns 'packed' as list of dicts, just append it:
        # Ensure key names are "Problem 1".."Problem 6"
        submission[f"Problem {i+1}"] = packed

    # save exactly one file containing the dict
    np.save("submission4.npy", submission, allow_pickle=True)
    print("\nSaved submission3.npy (dict with keys 'Problem 1'..'Problem 6')")
