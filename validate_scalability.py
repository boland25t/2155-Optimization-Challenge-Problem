"""
Test scalability with different population sizes and configurations
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import time
from optimize_mechanisms import MechanismOptimizer, OptimizationConfig

def test_scalability():
    """Test how performance scales with different configurations"""

    target_curves = np.load('target_curves.npy')
    target_curve = target_curves[1]  # Use curve 2

    print("="*60)
    print("SCALABILITY VALIDATION")
    print("="*60)

    # Test different configurations
    test_configs = [
        ("Small", OptimizationConfig(n_seeds=10, nsga_pop_size=50, nsga_n_gen=50, max_refined=5)),
        ("Medium", OptimizationConfig(n_seeds=20, nsga_pop_size=100, nsga_n_gen=100, max_refined=10)),
        ("Large", OptimizationConfig(n_seeds=40, nsga_pop_size=200, nsga_n_gen=150, max_refined=20)),
    ]

    results = {}

    for name, config in test_configs:
        print(f"\n{'='*40}")
        print(f"TESTING {name.upper()} CONFIGURATION")
        print(f"{'='*40}")
        print(f"Seeds: {config.n_seeds}, Pop: {config.nsga_pop_size}, "
              f"Gen: {config.nsga_n_gen}, Refined: {config.max_refined}")

        optimizer = MechanismOptimizer(config)

        start_time = time.time()
        solutions = optimizer.optimize_for_curve(target_curve, 1)
        elapsed = time.time() - start_time

        # Analyze quality
        feasible = [s for s in solutions if s['material'] <= 10.0]
        if feasible:
            best_distance = min(s['distance'] for s in feasible)
            avg_distance = np.mean([s['distance'] for s in feasible])
        else:
            best_distance = float('inf')
            avg_distance = float('inf')

        results[name] = {
            'time': elapsed,
            'solutions': len(solutions),
            'feasible': len(feasible),
            'best_distance': best_distance,
            'avg_distance': avg_distance,
            'config': config
        }

        print(f"Time: {elapsed:.1f}s")
        print(f"Solutions: {len(solutions)} ({len(feasible)} feasible)")
        print(f"Best distance: {best_distance:.4f}")
        print(f"Avg distance: {avg_distance:.4f}")

    # Performance analysis
    print(f"\n{'='*60}")
    print("SCALABILITY ANALYSIS")
    print(f"{'='*60}")

    print(f"\nTime scaling:")
    for name, result in results.items():
        config = result['config']
        total_evals = config.n_seeds * config.nsga_pop_size * config.nsga_n_gen
        print(f"  {name:8}: {result['time']:6.1f}s for {total_evals:8,} evaluations "
              f"({total_evals/result['time']:6.0f} eval/s)")

    print(f"\nQuality scaling:")
    for name, result in results.items():
        efficiency = result['feasible'] / result['time'] if result['time'] > 0 else 0
        print(f"  {name:8}: {result['best_distance']:6.4f} best distance, "
              f"{efficiency:5.1f} feasible/second")

    # Efficiency recommendations
    print(f"\nEfficiency Analysis:")
    best_efficiency = max(results.items(), key=lambda x: x[1]['feasible']/x[1]['time'])
    best_quality = min(results.items(), key=lambda x: x[1]['best_distance'])

    print(f"  Most efficient: {best_efficiency[0]} ({best_efficiency[1]['feasible']/best_efficiency[1]['time']:.1f} feasible/s)")
    print(f"  Best quality: {best_quality[0]} (distance: {best_quality[1]['best_distance']:.4f})")

    return results

if __name__ == "__main__":
    test_scalability()