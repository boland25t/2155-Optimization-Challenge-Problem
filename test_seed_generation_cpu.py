"""
Quick test to understand seed generation issues - CPU only version
"""

import os
# Force CPU usage before importing JAX
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
from LINKS.Optimization import MechanismRandomizer, Tools

# Initialize tools with explicit CPU
tools = Tools(device='cpu', material=True)
tools.compile()

randomizer = MechanismRandomizer(
    min_size=6,
    max_size=12,
    device='cpu'
)

# Load target curves
target_curves = np.load('target_curves.npy')
target_curve = target_curves[1]  # Use curve 2

print("Testing seed generation with different node counts (CPU only)...")
print("="*60)

materials = []
distances = []
successful_mechs = []

# Try different sizes
for n in range(6, 13):
    print(f"\nTrying mechanisms with {n} nodes:")

    for attempt in range(5):  # Try 5 times per size
        try:
            # Generate mechanism
            mech = randomizer(n=n, n_tests=1, max_tries=100)  # Reduced for CPU

            # Handle return format
            if isinstance(mech, (list, tuple)):
                if len(mech) == 0:
                    print(f"  Attempt {attempt+1}: No valid mechanism generated")
                    continue
                mech = mech[0]

            # Evaluate
            result = tools(
                mech['x0'],
                mech['edges'],
                mech['fixed_joints'],
                mech['motor'],
                target_curve,
                target_idx=mech.get('target_joint', n-1)
            )

            if isinstance(result, tuple):
                dist, mat = result
            else:
                dist = float(result)
                mat = np.inf

            dist = float(np.asarray(dist).item()) if np.isfinite(dist) else 1e9
            mat = float(np.asarray(mat).item()) if np.isfinite(mat) else 1e9

            materials.append(mat)
            distances.append(dist)
            successful_mechs.append((n, mat, dist, mech))

            status = "✓" if mat <= 10 else "✗"
            print(f"  Attempt {attempt+1}: material={mat:.2f} {status}, distance={dist:.3f}")

        except Exception as e:
            print(f"  Attempt {attempt+1}: Failed - {str(e)[:50]}...")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if materials:
    materials = np.array(materials)
    distances = np.array(distances)

    print(f"\nMaterial Statistics:")
    print(f"  Min: {materials.min():.2f}")
    print(f"  Mean: {materials.mean():.2f}")
    print(f"  Max: {materials.max():.2f}")
    print(f"  Under 10: {(materials <= 10).sum()}/{len(materials)}")

    print(f"\nDistance Statistics:")
    print(f"  Min: {distances.min():.3f}")
    print(f"  Mean: {distances.mean():.3f}")
    print(f"  Max: {distances.max():.3f}")

    # Show best mechanisms by material
    print(f"\nBest 5 mechanisms by material:")
    successful_mechs.sort(key=lambda x: x[1])  # Sort by material
    for i, (n, mat, dist, _) in enumerate(successful_mechs[:5]):
        print(f"  {i+1}. Nodes={n}, Material={mat:.2f}, Distance={dist:.3f}")

    # Distribution of materials
    print(f"\nMaterial distribution:")
    print(f"  < 10: {(materials < 10).sum()} mechanisms")
    print(f"  10-15: {((materials >= 10) & (materials < 15)).sum()} mechanisms")
    print(f"  15-20: {((materials >= 15) & (materials < 20)).sum()} mechanisms")
    print(f"  > 20: {(materials >= 20).sum()} mechanisms")

    print("\nSuggestion:")
    if (materials <= 10).sum() == 0:
        print("  No mechanisms with material < 10 found.")
        print("  Consider using material threshold of", f"{materials.min() * 1.1:.1f}")
    else:
        print(f"  Found {(materials <= 10).sum()} mechanisms with material < 10")
        print("  The pipeline should work with proper configuration.")

    # Save best mechanism for inspection
    if successful_mechs and successful_mechs[0][1] <= 15:
        best_mech = successful_mechs[0][3]
        print(f"\nBest mechanism details:")
        print(f"  Nodes: {best_mech['x0'].shape[0]}")
        print(f"  Edges: {len(best_mech['edges'])}")
        print(f"  Fixed joints: {len(best_mech['fixed_joints'])}")
else:
    print("No valid mechanisms could be generated!")
    print("Check that JAX is working properly with: python -c \"import jax; print(jax.devices())\"")