import argparse
import time
from typing import Dict, List, Any

import numpy as np

from seeds import MechanismSeeder, SeederConfig
from nsga_runner import NSGAConfig, run_nsga_for_curve, cap_nsga_candidates
from bfgs_refine import BFGSConfig  # refine is optional; we skip it here for speed
from selection import select_for_submission


from LINKS.CP import make_empty_submission, evaluate_submission


# ----------------------------
# Helpers
# ----------------------------
def pack_one_curve_for_grader(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    The grader expects a list of mechanism dicts for each 'Problem i'.
    We assume each item already contains simulator-ready fields (x0, A, etc.).
    """
    return items


def optimize_one_curve(
    curve_idx: int,
    target_curve: np.ndarray,
    seeds_for_curve: List[Dict[str, Any]],
    n_cfg: NSGAConfig,
    *,
    cap_per_curve: int,
) -> List[Dict[str, Any]]:
    t0 = time.time()

    # --- NSGA on the provided seeds (no fresh seeding here) ---
    per_seed_candidates = run_nsga_for_curve(seeds_for_curve, target_curve, n_cfg)

    # --- Cap per-curve to keep things tractable downstream ---
    candidates = cap_nsga_candidates(per_seed_candidates, cap=cap_per_curve)

    # --- Filter to a clean set (feasible & decent quality) ---
    selected = select_for_submission(
        candidates,
        cap=1000,  # generous cap; final grading script can prune further
        max_distance=0.75,
        max_material=10.0,
        require_feasible=True,
    )

    dt = time.time() - t0
    best = min(selected, key=lambda d: d["distance"]) if selected else None
    msg = (
        f"[Curve {curve_idx+1}] seeds={len(seeds_for_curve)}  "
        f"cand={len(candidates)}  sel={len(selected)}  "
        f"best_dist={best['distance']:.4f}"
        if best
        else "N/A"
    )
    print(msg + f"  ({dt:.1f}s)")

    return pack_one_curve_for_grader(selected)


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Global-pool seeding + per-curve NSGA pipeline"
    )
    p.add_argument(
        "--curves",
        type=str,
        default="target_curves.npy",
        help="Path to target curves .npy",
    )
    p.add_argument(
        "--out",
        type=str,
        default="submission.npy",
        help="Output submission file (.npy)",
    )
    # Global pool + per-curve picks
    p.add_argument(
        "--pool", type=int, default=240, help="Total mechanisms in global pool"
    )
    p.add_argument("--per_curve", type=int, default=40, help="Seeds used per curve")
    p.add_argument(
        "--strategy",
        type=str,
        default="diverse_by_nodes",
        choices=["diverse_by_nodes", "random"],
        help="Down-select strategy from the global pool",
    )
    p.add_argument(
        "--annotate_metrics",
        action="store_true",
        help="Annotate initial distance/material for logging only",
    )
    # Caps / sizes
    p.add_argument("--cap_per_curve", type=int, default=1000, help="NSGA cap per curve")
    # Seeder config (mirror your current defaults)
    p.add_argument("--min_nodes", type=int, default=6)
    p.add_argument("--max_nodes", type=int, default=20)
    p.add_argument("--fixed_probability", type=float, default=0.15)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"])
    p.add_argument("--preevaluate", action="store_true", default=True)
    p.add_argument("--no-preevaluate", dest="preevaluate", action="store_false")
    p.add_argument("--warmup_randomizer", action="store_true", default=False)
    p.add_argument("--n_tests_per_mech", type=int, default=36)
    p.add_argument("--max_tries_per_mech", type=int, default=100)
    p.add_argument(
        "--seed", type=int, default=1234, help="Base RNG seed for seeding/selection"
    )
    # NSGA config (mirror your current defaults)
    p.add_argument("--pop_size", type=int, default=160)
    p.add_argument("--n_gen", type=int, default=150)
    p.add_argument("--eta_cx", type=float, default=15.0)
    p.add_argument("--prob_cx", type=float, default=0.9)
    p.add_argument("--eta_mut", type=float, default=15.0)
    p.add_argument(
        "--prob_mut", type=float, default=-1.0, help="If <0, auto=1/n_var in wrapper"
    )
    p.add_argument("--scaled_distance", action="store_true", default=False)
    p.add_argument("--include_material", action="store_true", default=True)
    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()

    # 1) Load target curves
    target_curves: np.ndarray = np.load(args.curves)
    print(f"Loaded target curves: shape={target_curves.shape}")

    # 2) Build configs
    s_cfg = SeederConfig(
        n_seeds=args.per_curve,  # not used for pool; kept for internal APIs
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        fixed_probability=args.fixed_probability,
        timesteps=args.timesteps,
        device=args.device,
        preevaluate=args.preevaluate,
        seed=args.seed,
        warmup_randomizer=args.warmup_randomizer,
        n_tests_per_mech=args.n_tests_per_mech,
        max_tries_per_mech=args.max_tries_per_mech,
    )
    n_cfg = NSGAConfig(
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        eta_cx=args.eta_cx,
        prob_cx=args.prob_cx,
        eta_mut=args.eta_mut,
        prob_mut=None if args.prob_mut < 0 else args.prob_mut,
        device=args.device,
        max_size=s_cfg.max_nodes,
        timesteps=s_cfg.timesteps,
        n_workers_eval=0,  # bump if your eval path supports parallelism
        seed=args.seed + 777,  # decouple from seeding
        scaled_distance=args.scaled_distance,
        include_material=args.include_material,
    )
    # (Optional) refine config kept here if you later re-enable BFGS
    _b_cfg = BFGSConfig(
        timesteps=args.timesteps,
        max_size=s_cfg.max_nodes,
        device=args.device,
        material_threshold=10.0,
        penalty_lambda=5.0,
        maxiter=500,
        n_workers=0,
        seed=args.seed + 999,
    )

    # 3) Build a single global pool (curve-agnostic)
    print(f"[Seeding] Generating global pool of {args.pool} mechanisms...")
    seeder = MechanismSeeder(s_cfg)
    GLOBAL_POOL = seeder.generate_pool(args.pool)
    print(f"[Seeding] Global pool ready: {len(GLOBAL_POOL)}")

    # 4) Prepare submission container
    if make_empty_submission is not None:
        submission: Dict[str, List[Dict[str, Any]]] = make_empty_submission()
    else:
        submission = {f"Problem {i+1}": [] for i in range(len(target_curves))}

    # 5) Per-curve optimization using down-selected seeds from the global pool
    master_rng = np.random.default_rng(args.seed + 424242)
    for i, curve in enumerate(target_curves):
        # Deterministic but different draw per curve
        seeds_for_curve = seeder.select_for_curve(
            GLOBAL_POOL,
            target_curve=curve,
            k=args.per_curve,
            strategy=args.strategy,
            annotate_metrics=args.annotate_metrics,
            rng_seed=int(master_rng.integers(0, 2**32 - 1)),
        )
        print(
            f"[Curve {i+1}] Using {len(seeds_for_curve)} seeds from global pool ({args.strategy})"
        )

        packed = optimize_one_curve(
            i,
            curve,
            seeds_for_curve,
            n_cfg,
            cap_per_curve=args.cap_per_curve,
        )
        submission[f"Problem {i+1}"] = packed

    # 6) Save
    np.save(args.out, submission, allow_pickle=True)
    print(f"\nSaved {args.out} (keys: " + ", ".join(sorted(submission.keys())) + ")")


if __name__ == "__main__":
    main()
