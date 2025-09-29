# example_nsga.py
import numpy as np
from seeds import MechanismSeeder, SeederConfig
from nsga_runner import NSGAConfig, run_nsga_for_curve, cap_nsga_candidates
from bfgs_refine import BFGSConfig, refine_candidates

# load curves and pick one
target_curves = np.load("target_curves.npy")
curve = target_curves[0]

# 1) seeds
s_cfg = SeederConfig(
    n_seeds=16,
    min_nodes=6, max_nodes=20, fixed_probability=0.15,
    timesteps=200, device="gpu",
    preevaluate=True, seed=1234,
    warmup_randomizer=False,
)
seeder = MechanismSeeder(s_cfg)
seeds = seeder.generate_for_curve(curve)

# 2) nsga
n_cfg = NSGAConfig(
    pop_size=128,
    n_gen=60,
    device="gpu",
    max_size=s_cfg.max_nodes,
    timesteps=s_cfg.timesteps,
    include_material=True,
    scaled_distance=False,
    n_workers_eval=0,   # start single-threaded; bump later if you like
    seed=4321,
)
per_seed_results = run_nsga_for_curve(seeds, curve, n_cfg)

# 3) flatten candidates (no pruning; ready for BFGS)
candidates = cap_nsga_candidates(per_seed_results, cap=1000)
print(f"Total NSGA candidates: {len(candidates)}")
print("Sample candidate:", {k: type(v) for k, v in candidates[0].items()})

#Run BFGS

b_cfg = BFGSConfig(
    timesteps=200, max_size=s_cfg.max_nodes, device="gpu",
    material_threshold=10.0, penalty_lambda=5.0,
    maxiter=200, n_workers=0, seed=999
)
refined = refine_candidates(candidates, curve, b_cfg)

# Sort by distance and take top-K (e.g., 15) for submission
refined.sort(key=lambda d: d["distance"])
top_k = refined[:15]