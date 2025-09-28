"""
Quick test to understand seed generation issues
"""

import numpy as np
from LINKS.Optimization import MechanismRandomizer, Tools

# Initialize tools
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

print("Testing seed generation with different node counts...")
print("="*60)

materials = []
distances = []

# Try different sizes
for n in range(6, 13):
    print(f"\nTrying mechanisms with {n} nodes:")

    for attempt in range(5):  # Try 5 times per size
        try:
            # Generate mechanism
            mech = randomizer(n=n, n_tests=64, max_tries=200)

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

            print(f"  Attempt {attempt+1}: material={mat:.2f}, distance={dist:.3f}")

        except Exception as e:
            print(f"  Attempt {attempt+1}: Failed - {e}")

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

    # Show which node counts work best
    print(f"\nBest material found: {materials.min():.2f}")
    best_idx = np.argmin(materials)
    print(f"  (This was from the mechanism at index {best_idx})")

    # Distribution of materials
    print(f"\nMaterial distribution:")
    print(f"  < 10: {(materials < 10).sum()} mechanisms")
    print(f"  10-15: {((materials >= 10) & (materials < 15)).sum()} mechanisms")
    print(f"  15-20: {((materials >= 15) & (materials < 20)).sum()} mechanisms")
    print(f"  > 20: {(materials >= 20).sum()} mechanisms")

    print("\nSuggestion:")
    if (materials <= 10).sum() == 0:
        print("  No mechanisms with material < 10 found.")
        print("  Consider relaxing the material threshold or using smaller mechanisms.")
    else:
        print(f"  Found {(materials <= 10).sum()} mechanisms with material < 10")
        print("  The pipeline should work with proper configuration.")
else:
    print("No valid mechanisms could be generated!")