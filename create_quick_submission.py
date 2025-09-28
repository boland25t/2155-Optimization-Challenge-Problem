"""
Create submission from current validation results
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pickle
from optimize_mechanisms import MechanismOptimizer, OptimizationConfig

def create_quick_submission():
    """Create a submission with current good results"""

    target_curves = np.load('target_curves.npy')

    # Use the validated configuration (proven to work)
    config = OptimizationConfig(
        n_seeds=15,
        min_nodes=6,
        max_nodes=10,
        nsga_pop_size=100,
        nsga_n_gen=100,
        max_refined=10
    )

    optimizer = MechanismOptimizer(config)

    print("="*60)
    print("CREATING QUICK SUBMISSION")
    print("="*60)
    print("Using validated config: 15 seeds, 100 pop, 100 gen")
    print("Expected time: ~15 minutes for all 6 curves")

    submission_data = []

    for curve_idx in range(6):
        print(f"\nProcessing Curve {curve_idx + 1}/6...")

        # Run optimization
        solutions = optimizer.optimize_for_curve(target_curves[curve_idx], curve_idx)

        # Convert to submission format (take top 15 solutions)
        solutions_sorted = sorted(solutions, key=lambda x: x['distance'])[:15]

        curve_submission = []
        for sol in solutions_sorted:
            mech = sol['mech'].copy()
            mech['x0'] = sol['x0']

            curve_submission.append({
                'x0': mech['x0'],
                'edges': mech['edges'],
                'fixed_joints': mech['fixed_joints'],
                'motor': mech['motor'],
                'target_joint': mech.get('target_joint', mech['x0'].shape[0] - 1)
            })

        submission_data.append(curve_submission)

        # Quick stats
        feasible = [s for s in solutions if s['material'] <= 10.0]
        best = min(solutions, key=lambda x: x['distance'])
        print(f"  Best distance: {best['distance']:.4f}")
        print(f"  Solutions for submission: {len(curve_submission)}")

    # Save submission (use object array to handle variable lengths)
    submission_array = np.empty(6, dtype=object)
    for i, curve_data in enumerate(submission_data):
        submission_array[i] = curve_data

    np.save('my_quick_submission.npy', submission_array, allow_pickle=True)

    print(f"\n" + "="*60)
    print("SUBMISSION CREATED")
    print("="*60)
    print("Saved to: my_quick_submission.npy")

    # Print summary
    for i, curve_mechs in enumerate(submission_data):
        print(f"Curve {i+1}: {len(curve_mechs)} mechanisms")

    return submission_data

def test_submission_format():
    """Test that submission is in correct format"""
    try:
        submission = np.load('my_quick_submission.npy', allow_pickle=True)
        print(f"\nSubmission format validation:")
        print(f"  Type: {type(submission)}")
        print(f"  Length: {len(submission)} (should be 6)")

        for i, curve_data in enumerate(submission):
            print(f"  Curve {i+1}: {len(curve_data)} mechanisms")
            if len(curve_data) > 0:
                mech = curve_data[0]
                print(f"    Sample mechanism keys: {list(mech.keys())}")
                print(f"    Node count: {mech['x0'].shape[0]}")

        print("✓ Submission format looks correct")
        return True

    except Exception as e:
        print(f"✗ Submission format error: {e}")
        return False

if __name__ == "__main__":
    # Create submission
    submission = create_quick_submission()

    # Test format
    test_submission_format()

    print(f"\nNext steps:")
    print(f"1. Test with: from LINKS.CP import evaluate_submission; evaluate_submission(np.load('my_quick_submission.npy', allow_pickle=True))")
    print(f"2. If good, scale up with more seeds/generations for final submission")