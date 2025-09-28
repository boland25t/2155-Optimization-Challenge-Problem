"""
Full-scale optimization for all 6 target curves
Generates up to 1000 solutions per curve
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force CPU to avoid Metal issues

import numpy as np
import time
import pickle
from optimize_mechanisms import MechanismOptimizer, OptimizationConfig

def run_full_optimization():
    """Run the complete optimization pipeline for all curves"""

    # Load target curves
    target_curves = np.load('target_curves.npy')

    # Production configuration based on successful test
    config = OptimizationConfig(
        # Seed generation
        n_seeds=100,        # Generate many diverse seeds
        min_nodes=6,        # Minimum mechanism size
        max_nodes=10,       # Maximum mechanism size (10 still has ~30% success rate)
        material_threshold=10.0,

        # NSGA-2 parameters
        nsga_pop_size=150,  # Larger population for better exploration
        nsga_n_gen=200,     # More generations for convergence

        # Gradient refinement
        max_refined=30,     # Refine top 30 solutions per seed batch
        lbfgs_maxiter=200,  # More iterations for better refinement
        penalty_lambda=100.0  # Strong penalty for material violations
    )

    print("="*60)
    print("FULL MECHANISM OPTIMIZATION PIPELINE")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Seeds per curve: {config.n_seeds}")
    print(f"  Node range: {config.min_nodes}-{config.max_nodes}")
    print(f"  NSGA-2: {config.nsga_pop_size} pop, {config.nsga_n_gen} generations")
    print(f"  Refinement: Top {config.max_refined} solutions")
    print(f"  Material threshold: {config.material_threshold}")

    # Initialize optimizer
    optimizer = MechanismOptimizer(config)

    # Store results for all curves
    all_results = {}

    # Process each target curve
    for curve_idx in range(len(target_curves)):
        print(f"\n{'='*60}")
        print(f"OPTIMIZING FOR TARGET CURVE {curve_idx + 1}/6")
        print(f"{'='*60}")

        start_time = time.time()

        # Run optimization
        solutions = optimizer.optimize_for_curve(target_curves[curve_idx], curve_idx)

        # Store results
        all_results[curve_idx] = solutions

        # Report statistics
        elapsed = time.time() - start_time
        feasible = [s for s in solutions if s['material'] <= 10.0 and s['distance'] <= 0.75]

        print(f"\nCurve {curve_idx + 1} Complete:")
        print(f"  Time: {elapsed/60:.1f} minutes")
        print(f"  Total solutions: {len(solutions)}")
        print(f"  Feasible solutions: {len(feasible)}")
        if feasible:
            best = min(feasible, key=lambda x: x['distance'])
            print(f"  Best distance: {best['distance']:.4f} (material: {best['material']:.2f})")

        # Save intermediate results after each curve
        with open(f'results_curve_{curve_idx}.pkl', 'wb') as f:
            pickle.dump(solutions, f)
        print(f"  Saved to: results_curve_{curve_idx}.pkl")

    # Final summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")

    total_solutions = sum(len(sols) for sols in all_results.values())
    total_feasible = sum(len([s for s in sols if s['material'] <= 10.0 and s['distance'] <= 0.75])
                         for sols in all_results.values())

    print(f"\nOverall Statistics:")
    print(f"  Total solutions: {total_solutions}")
    print(f"  Total feasible: {total_feasible}")
    print(f"  Average per curve: {total_solutions/6:.0f}")

    # Save all results
    with open('all_optimization_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nAll results saved to: all_optimization_results.pkl")

    # Create submission format
    create_submission_file(all_results)

    return all_results


def create_submission_file(results, max_per_curve=1000):
    """Create submission file from optimization results"""

    submission = []

    for curve_idx in range(6):
        curve_solutions = results.get(curve_idx, [])

        # Sort by distance and take top solutions
        curve_solutions = sorted(curve_solutions, key=lambda x: x['distance'])[:max_per_curve]

        # Format for submission
        curve_submission = []
        for sol in curve_solutions:
            mech = sol['mech'].copy()
            mech['x0'] = sol['x0']

            curve_submission.append({
                'x0': mech['x0'],
                'edges': mech['edges'],
                'fixed_joints': mech['fixed_joints'],
                'motor': mech['motor'],
                'target_joint': mech.get('target_joint', mech['x0'].shape[0] - 1)
            })

        submission.append(curve_submission)

    # Save submission
    np.save('my_full_submission.npy', submission, allow_pickle=True)
    print(f"\nSubmission saved to: my_full_submission.npy")

    # Print submission statistics
    for i, curve_mechs in enumerate(submission):
        print(f"  Curve {i+1}: {len(curve_mechs)} mechanisms")

    return submission


def run_single_curve_test(curve_idx=1):
    """Test on a single curve first"""

    print(f"Running single curve test on Curve {curve_idx + 1}")

    target_curves = np.load('target_curves.npy')

    # Smaller config for single curve test
    config = OptimizationConfig(
        n_seeds=30,         # Fewer seeds for testing
        min_nodes=6,
        max_nodes=8,        # Focus on successful range
        nsga_pop_size=100,
        nsga_n_gen=100,
        max_refined=20,
        material_threshold=10.0
    )

    optimizer = MechanismOptimizer(config)

    start_time = time.time()
    solutions = optimizer.optimize_for_curve(target_curves[curve_idx], curve_idx)
    elapsed = time.time() - start_time

    print(f"\nTest complete in {elapsed/60:.1f} minutes")
    print(f"Found {len(solutions)} solutions")

    # Check quality
    feasible = [s for s in solutions if s['material'] <= 10.0]
    if feasible:
        best = min(feasible, key=lambda x: x['distance'])
        print(f"Best feasible: distance={best['distance']:.4f}, material={best['material']:.2f}")

    return solutions


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Run single curve test
        curve_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        run_single_curve_test(curve_idx)
    else:
        # Run full optimization
        print("Starting full optimization for all 6 curves...")
        print("This will take several hours. Run with --test flag for single curve test.")

        response = input("Continue with full optimization? (y/n): ")
        if response.lower() == 'y':
            run_full_optimization()
        else:
            print("Aborted. Run with --test flag for single curve test:")
            print("  python run_full_optimization.py --test [curve_index]")