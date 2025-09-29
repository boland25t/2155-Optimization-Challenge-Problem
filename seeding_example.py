# example_seeding.py
import numpy as np
from seeds import MechanismSeeder, SeederConfig

# Load one curve (e.g., from starter file target_curves.npy)
target_curves = np.load("target_curves.npy")  # shape (num_curves, 200, 2)
curve = target_curves[0]

cfg = SeederConfig(
    n_seeds=64,
    min_nodes=6,
    max_nodes=20,
    fixed_probability=0.15,
    timesteps=200,
    device="gpu",
    preevaluate=True,
    seed=1234,
)

seeder = MechanismSeeder(cfg)
seeds = seeder.generate_for_curve(curve)

# seeds is a list[dict]; each dict already matches grader schema fields
# (x0, edges, fixed_joints, motor, target_joint) and may carry distance/material
print("First seed keys:", seeds[0].keys())
print("Sample seed sizes:", seeds[0]["x0"].shape, seeds[0]["edges"].shape)
