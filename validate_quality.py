"""
Quality validation across all target curves
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import matplotlib.pyplot as plt
import time
from optimize_mechanisms import MechanismOptimizer, OptimizationConfig
from LINKS.Visualization import MechanismVisualizer
from LINKS.Kinematics import MechanismSolver
from LINKS.Geometry import CurveEngine

def validate_all_curves():
    """Test optimization quality on all 6 target curves"""

    target_curves = np.load('target_curves.npy')

    # Moderate test config for quality assessment
    config = OptimizationConfig(
        n_seeds=15,
        min_nodes=6,
        max_nodes=10,
        nsga_pop_size=100,
        nsga_n_gen=100,
        max_refined=10
    )

    print("="*60)
    print("QUALITY VALIDATION ACROSS ALL CURVES")
    print("="*60)

    optimizer = MechanismOptimizer(config)
    visualizer = MechanismVisualizer()
    solver = MechanismSolver(device='cpu')
    curve_engine = CurveEngine(device='cpu', normalize_scale=False)

    results = {}

    for curve_idx in range(len(target_curves)):
        print(f"\n{'='*40}")
        print(f"TESTING CURVE {curve_idx + 1}")
        print(f"{'='*40}")

        start_time = time.time()
        solutions = optimizer.optimize_for_curve(target_curves[curve_idx], curve_idx)
        elapsed = time.time() - start_time

        # Analyze results
        feasible = [s for s in solutions if s['material'] <= 10.0 and s['distance'] <= 0.75]

        if solutions:
            best_overall = min(solutions, key=lambda x: x['distance'])
            best_feasible = min(feasible, key=lambda x: x['distance']) if feasible else None

            results[curve_idx] = {
                'time': elapsed,
                'total_solutions': len(solutions),
                'feasible_solutions': len(feasible),
                'best_distance': best_overall['distance'],
                'best_material': best_overall['material'],
                'best_feasible_distance': best_feasible['distance'] if best_feasible else None,
                'best_solution': best_overall
            }

            print(f"Time: {elapsed:.1f}s")
            print(f"Solutions: {len(solutions)} total, {len(feasible)} feasible")
            print(f"Best distance: {best_overall['distance']:.4f} (material: {best_overall['material']:.2f})")
            if best_feasible:
                print(f"Best feasible: {best_feasible['distance']:.4f} (material: {best_feasible['material']:.2f})")
        else:
            print("No solutions found!")
            results[curve_idx] = None

    # Summary report
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    total_time = sum(r['time'] for r in results.values() if r)
    total_solutions = sum(r['total_solutions'] for r in results.values() if r)
    total_feasible = sum(r['feasible_solutions'] for r in results.values() if r)

    print(f"\nOverall Statistics:")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Total solutions: {total_solutions}")
    print(f"  Total feasible: {total_feasible}")
    print(f"  Success rate: {len([r for r in results.values() if r])/6:.1%}")

    print(f"\nPer-curve results:")
    for i, result in results.items():
        if result:
            status = "✓" if result['feasible_solutions'] > 0 else "⚠"
            print(f"  Curve {i+1} {status}: {result['best_distance']:.4f} distance, "
                  f"{result['feasible_solutions']} feasible, {result['time']:.1f}s")
        else:
            print(f"  Curve {i+1} ✗: Failed")

    # Create comparison visualization
    create_quality_plots(results, target_curves)

    return results

def create_quality_plots(results, target_curves):
    """Create plots comparing optimization results across curves"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    distances = []
    materials = []
    times = []

    for i, result in results.items():
        if result:
            distances.append(result['best_distance'])
            materials.append(result['best_material'])
            times.append(result['time'])
        else:
            distances.append(np.nan)
            materials.append(np.nan)
            times.append(np.nan)

    # Plot 1: Distance comparison
    axes[0].bar(range(1, 7), distances, color='skyblue', alpha=0.7)
    axes[0].axhline(y=0.75, color='r', linestyle='--', alpha=0.5, label='Target')
    axes[0].set_xlabel('Curve')
    axes[0].set_ylabel('Best Distance')
    axes[0].set_title('Best Distance per Curve')
    axes[0].legend()

    # Plot 2: Material comparison
    axes[1].bar(range(1, 7), materials, color='lightgreen', alpha=0.7)
    axes[1].axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Limit')
    axes[1].set_xlabel('Curve')
    axes[1].set_ylabel('Material Usage')
    axes[1].set_title('Material Usage per Curve')
    axes[1].legend()

    # Plot 3: Time comparison
    axes[2].bar(range(1, 7), times, color='orange', alpha=0.7)
    axes[2].set_xlabel('Curve')
    axes[2].set_ylabel('Time (seconds)')
    axes[2].set_title('Optimization Time per Curve')

    # Plot 4: Feasible solutions count
    feasible_counts = [results[i]['feasible_solutions'] if results[i] else 0 for i in range(6)]
    axes[3].bar(range(1, 7), feasible_counts, color='purple', alpha=0.7)
    axes[3].set_xlabel('Curve')
    axes[3].set_ylabel('Feasible Solutions')
    axes[3].set_title('Number of Feasible Solutions')

    # Plot 5: Distance vs Material scatter
    valid_results = [r for r in results.values() if r]
    if valid_results:
        dist_vals = [r['best_distance'] for r in valid_results]
        mat_vals = [r['best_material'] for r in valid_results]
        axes[4].scatter(dist_vals, mat_vals, s=100, alpha=0.7)
        axes[4].axhline(y=10, color='r', linestyle='--', alpha=0.5)
        axes[4].axvline(x=0.75, color='r', linestyle='--', alpha=0.5)
        axes[4].set_xlabel('Distance')
        axes[4].set_ylabel('Material')
        axes[4].set_title('Distance vs Material Trade-off')

    # Hide unused subplot
    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nValidation plots saved to: validation_results.png")

if __name__ == "__main__":
    validate_all_curves()