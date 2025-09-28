"""
Test script to validate the optimization pipeline on a single target curve
This will help verify each stage is working before scaling up
"""

import os
# Force CPU to avoid Metal/JAX issues
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time
from optimize_mechanisms import MechanismOptimizer, OptimizationConfig
from LINKS.Visualization import MechanismVisualizer, GAVisualizer
from LINKS.Kinematics import MechanismSolver
from LINKS.Geometry import CurveEngine


class PipelineValidator:
    """Validate and visualize the optimization pipeline"""

    def __init__(self):
        self.visualizer = MechanismVisualizer()
        self.ga_visualizer = GAVisualizer()
        self.solver = MechanismSolver(device='cpu')
        self.curve_engine = CurveEngine(device='cpu', normalize_scale=False)

        # Track metrics at each stage
        self.metrics = {
            'seeds': {'distances': [], 'materials': []},
            'nsga2': {'distances': [], 'materials': []},
            'refined': {'distances': [], 'materials': []}
        }

    def test_seed_generation(self, optimizer: MechanismOptimizer,
                           target_curve: np.ndarray, n_test_seeds: int = 10):
        """Test and visualize seed generation"""
        print("\n" + "="*60)
        print("STAGE 1: SEED GENERATION TEST")
        print("="*60)

        start_time = time.time()
        seeds = optimizer.generate_seeds(target_curve)[:n_test_seeds]
        gen_time = time.time() - start_time

        print(f"Generated {len(seeds)} seeds in {gen_time:.2f} seconds")

        # Evaluate all seeds
        for i, seed in enumerate(seeds):
            dist, mat = optimizer.evaluate_mechanism(seed, target_curve)
            self.metrics['seeds']['distances'].append(dist)
            self.metrics['seeds']['materials'].append(mat)

        # Statistics
        if not self.metrics['seeds']['distances']:
            print("\nNo seeds were successfully generated!")
            print("This likely means the material constraint is too strict.")
            print("Try adjusting the configuration or checking the randomizer.")
            return seeds

        distances = np.array(self.metrics['seeds']['distances'])
        materials = np.array(self.metrics['seeds']['materials'])

        print(f"\nSeed Statistics:")
        print(f"  Distance: min={distances.min():.4f}, mean={distances.mean():.4f}, max={distances.max():.4f}")
        print(f"  Material: min={materials.min():.4f}, mean={materials.mean():.4f}, max={materials.max():.4f}")
        print(f"  Feasible (mat≤10): {(materials <= 10).sum()}/{len(materials)}")

        # Visualize best and worst seeds
        best_idx = np.argmin(distances)
        worst_idx = np.argmax(distances)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Best seed
        axes[0].set_title(f'Best Seed (d={distances[best_idx]:.3f}, m={materials[best_idx]:.3f})')
        self.visualizer(seeds[best_idx]['x0'], seeds[best_idx]['edges'],
                       seeds[best_idx]['fixed_joints'], seeds[best_idx]['motor'], ax=axes[0])
        axes[0].axis('equal')

        # Worst seed
        axes[1].set_title(f'Worst Seed (d={distances[worst_idx]:.3f}, m={materials[worst_idx]:.3f})')
        self.visualizer(seeds[worst_idx]['x0'], seeds[worst_idx]['edges'],
                       seeds[worst_idx]['fixed_joints'], seeds[worst_idx]['motor'], ax=axes[1])
        axes[1].axis('equal')

        # Distribution plot
        axes[2].scatter(distances, materials, alpha=0.6)
        axes[2].axhline(y=10, color='r', linestyle='--', label='Material limit')
        axes[2].axvline(x=0.75, color='r', linestyle='--', label='Distance limit')
        axes[2].set_xlabel('Distance')
        axes[2].set_ylabel('Material')
        axes[2].set_title('Seed Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return seeds

    def test_nsga2_stage(self, optimizer: MechanismOptimizer,
                         seeds: List[Dict], target_curve: np.ndarray):
        """Test NSGA-2 optimization on seeds"""
        print("\n" + "="*60)
        print("STAGE 2: NSGA-2 OPTIMIZATION TEST")
        print("="*60)

        # Run NSGA-2 on first few seeds
        test_seeds = seeds[:3]  # Test on 3 seeds for speed

        nsga_results = []
        for i, seed in enumerate(test_seeds):
            print(f"\nRunning NSGA-2 on seed {i+1}/{len(test_seeds)}...")
            start_time = time.time()

            result = optimizer.run_nsga2(seed, target_curve)

            if result is not None:
                nsga_results.append(result)
                print(f"  Completed in {time.time() - start_time:.2f}s")
                print(f"  Solutions found: {result['X'].shape[0] if result['X'].ndim > 1 else 1}")
            else:
                print(f"  No valid solutions found")

        # Collect all NSGA-2 solutions
        all_nsga_solutions = []
        for res in nsga_results:
            mech = res['mech']
            if res['X'].ndim == 1:
                x0 = res['X'].reshape(mech['x0'].shape)
                dist, mat = optimizer.evaluate_mechanism({'x0': x0, **mech}, target_curve)
                all_nsga_solutions.append({'x0': x0, 'mech': mech, 'distance': dist, 'material': mat})
                self.metrics['nsga2']['distances'].append(dist)
                self.metrics['nsga2']['materials'].append(mat)
            else:
                for j in range(res['X'].shape[0]):
                    x0 = res['X'][j].reshape(mech['x0'].shape)
                    dist, mat = optimizer.evaluate_mechanism({'x0': x0, **mech}, target_curve)
                    all_nsga_solutions.append({'x0': x0, 'mech': mech, 'distance': dist, 'material': mat})
                    self.metrics['nsga2']['distances'].append(dist)
                    self.metrics['nsga2']['materials'].append(mat)

        # Compare before/after NSGA-2
        if self.metrics['nsga2']['distances']:
            print(f"\nNSGA-2 Statistics:")
            nsga_distances = np.array(self.metrics['nsga2']['distances'])
            nsga_materials = np.array(self.metrics['nsga2']['materials'])

            print(f"  Distance: min={nsga_distances.min():.4f}, mean={nsga_distances.mean():.4f}")
            print(f"  Material: min={nsga_materials.min():.4f}, mean={nsga_materials.mean():.4f}")
            print(f"  Feasible: {((nsga_distances <= 0.75) & (nsga_materials <= 10)).sum()}/{len(nsga_distances)}")

            # Improvement from seeds
            seed_best_dist = np.array(self.metrics['seeds']['distances']).min()
            nsga_best_dist = nsga_distances.min()
            print(f"\nImprovement: {seed_best_dist:.4f} → {nsga_best_dist:.4f} "
                  f"({100*(seed_best_dist - nsga_best_dist)/seed_best_dist:.1f}% better)")

        return all_nsga_solutions

    def test_gradient_refinement(self, optimizer: MechanismOptimizer,
                                nsga_solutions: List[Dict], target_curve: np.ndarray):
        """Test gradient-based refinement"""
        print("\n" + "="*60)
        print("STAGE 3: GRADIENT REFINEMENT TEST")
        print("="*60)

        # Select top solutions to refine
        nsga_solutions = sorted(nsga_solutions, key=lambda x: x['distance'])[:5]

        refined_solutions = []
        for i, sol in enumerate(nsga_solutions):
            print(f"\nRefining solution {i+1}/{len(nsga_solutions)}...")
            print(f"  Before: distance={sol['distance']:.4f}, material={sol['material']:.4f}")

            start_time = time.time()
            x0_refined, dist, mat = optimizer.gradient_refine(
                sol['mech'], sol['x0'], target_curve
            )
            refine_time = time.time() - start_time

            print(f"  After:  distance={dist:.4f}, material={mat:.4f}")
            print(f"  Time: {refine_time:.2f}s")

            refined_solutions.append({
                'mech': sol['mech'],
                'x0_before': sol['x0'],
                'x0_after': x0_refined,
                'dist_before': sol['distance'],
                'dist_after': dist,
                'mat_before': sol['material'],
                'mat_after': mat
            })

            self.metrics['refined']['distances'].append(dist)
            self.metrics['refined']['materials'].append(mat)

        return refined_solutions

    def visualize_full_pipeline(self, target_curve: np.ndarray):
        """Create comprehensive visualization of all stages"""
        print("\n" + "="*60)
        print("PIPELINE SUMMARY VISUALIZATION")
        print("="*60)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Distance progression
        stages = ['Seeds', 'NSGA-2', 'Refined']
        best_dists = []
        mean_dists = []

        for stage in ['seeds', 'nsga2', 'refined']:
            if self.metrics[stage]['distances']:
                dists = np.array(self.metrics[stage]['distances'])
                best_dists.append(dists.min())
                mean_dists.append(dists.mean())
            else:
                best_dists.append(np.nan)
                mean_dists.append(np.nan)

        x_pos = np.arange(len(stages))
        axes[0, 0].bar(x_pos - 0.2, best_dists, 0.4, label='Best', color='green', alpha=0.7)
        axes[0, 0].bar(x_pos + 0.2, mean_dists, 0.4, label='Mean', color='blue', alpha=0.7)
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(stages)
        axes[0, 0].set_ylabel('Distance')
        axes[0, 0].set_title('Distance Improvement Through Pipeline')
        axes[0, 0].legend()
        axes[0, 0].axhline(y=0.75, color='r', linestyle='--', alpha=0.5, label='Target')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Material progression
        best_mats = []
        mean_mats = []

        for stage in ['seeds', 'nsga2', 'refined']:
            if self.metrics[stage]['materials']:
                mats = np.array(self.metrics[stage]['materials'])
                best_mats.append(mats.min())
                mean_mats.append(mats.mean())
            else:
                best_mats.append(np.nan)
                mean_mats.append(np.nan)

        axes[0, 1].bar(x_pos - 0.2, best_mats, 0.4, label='Best', color='green', alpha=0.7)
        axes[0, 1].bar(x_pos + 0.2, mean_mats, 0.4, label='Mean', color='blue', alpha=0.7)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(stages)
        axes[0, 1].set_ylabel('Material')
        axes[0, 1].set_title('Material Usage Through Pipeline')
        axes[0, 1].legend()
        axes[0, 1].axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Limit')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Pareto fronts comparison
        axes[1, 0].set_title('Pareto Front Evolution')

        # Seeds
        if self.metrics['seeds']['distances']:
            axes[1, 0].scatter(self.metrics['seeds']['distances'],
                             self.metrics['seeds']['materials'],
                             alpha=0.4, s=30, label='Seeds')

        # NSGA-2
        if self.metrics['nsga2']['distances']:
            axes[1, 0].scatter(self.metrics['nsga2']['distances'],
                             self.metrics['nsga2']['materials'],
                             alpha=0.6, s=40, label='NSGA-2')

        # Refined
        if self.metrics['refined']['distances']:
            axes[1, 0].scatter(self.metrics['refined']['distances'],
                             self.metrics['refined']['materials'],
                             alpha=0.8, s=50, label='Refined', color='red')

        axes[1, 0].axhline(y=10, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=0.75, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Distance')
        axes[1, 0].set_ylabel('Material')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Feasible solutions count
        feasible_counts = []
        total_counts = []

        for stage in ['seeds', 'nsga2', 'refined']:
            if self.metrics[stage]['distances']:
                dists = np.array(self.metrics[stage]['distances'])
                mats = np.array(self.metrics[stage]['materials'])
                feasible = ((dists <= 0.75) & (mats <= 10)).sum()
                feasible_counts.append(feasible)
                total_counts.append(len(dists))
            else:
                feasible_counts.append(0)
                total_counts.append(0)

        axes[1, 1].bar(x_pos, total_counts, label='Total', alpha=0.5)
        axes[1, 1].bar(x_pos, feasible_counts, label='Feasible', alpha=0.8)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(stages)
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Solution Counts')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\nFinal Statistics:")
        if self.metrics['refined']['distances']:
            final_dists = np.array(self.metrics['refined']['distances'])
            final_mats = np.array(self.metrics['refined']['materials'])
            feasible_mask = (final_dists <= 0.75) & (final_mats <= 10)

            print(f"  Total refined solutions: {len(final_dists)}")
            print(f"  Feasible solutions: {feasible_mask.sum()}")
            print(f"  Best distance: {final_dists.min():.4f}")
            print(f"  Best feasible distance: {final_dists[feasible_mask].min():.4f}"
                  if feasible_mask.any() else "  No feasible solutions")

    def visualize_best_mechanism(self, refined_solutions: List[Dict], target_curve: np.ndarray):
        """Visualize the best refined mechanism"""
        if not refined_solutions:
            print("No refined solutions to visualize")
            return

        # Find best by distance
        best = min(refined_solutions, key=lambda x: x['dist_after'])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Before refinement
        axes[0].set_title(f"Before Refinement\n(d={best['dist_before']:.3f}, m={best['mat_before']:.3f})")
        self.visualizer(best['x0_before'], best['mech']['edges'],
                       best['mech']['fixed_joints'], best['mech']['motor'], ax=axes[0])
        axes[0].axis('equal')

        # After refinement
        axes[1].set_title(f"After Refinement\n(d={best['dist_after']:.3f}, m={best['mat_after']:.3f})")
        self.visualizer(best['x0_after'], best['mech']['edges'],
                       best['mech']['fixed_joints'], best['mech']['motor'], ax=axes[1])
        axes[1].axis('equal')

        # Curve comparison
        axes[2].set_title("Traced Curve vs Target")
        target_joint = best['mech'].get('target_joint', best['x0_after'].shape[0] - 1)
        traced_curve = self.solver(best['x0_after'], best['mech']['edges'],
                                  best['mech']['fixed_joints'], best['mech']['motor'])[target_joint]

        # Plot curves
        axes[2].plot(target_curve[:, 0], target_curve[:, 1], 'r-', linewidth=2, label='Target')
        traced_curve_np = np.array(traced_curve)
        axes[2].plot(traced_curve_np[:, 0], traced_curve_np[:, 1], 'b--', linewidth=2, label='Traced')
        axes[2].axis('equal')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def run_pipeline_test():
    """Main test function"""
    print("="*60)
    print("OPTIMIZATION PIPELINE VALIDATION TEST")
    print("="*60)

    # Load target curves
    target_curves = np.load('target_curves.npy')

    # Select easiest curve (usually curve 1 or 2)
    test_curve_idx = 1  # Test with curve 2 (index 1)
    target_curve = target_curves[test_curve_idx]

    print(f"\nTesting with Target Curve {test_curve_idx + 1}")
    print("This is a small-scale test to validate the pipeline")

    # Create optimizer with small test configuration
    # Based on our testing, 6-8 nodes work best for material < 10
    config = OptimizationConfig(
        n_seeds=15,        # Small number for testing
        min_nodes=6,
        max_nodes=8,       # Focus on smaller mechanisms that meet material constraint
        nsga_pop_size=50,  # Smaller population for speed
        nsga_n_gen=50,     # Fewer generations for testing
        max_refined=5,     # Refine just top 5
        material_threshold=10.0,
        lbfgs_maxiter=100  # Fewer iterations for testing
    )

    optimizer = MechanismOptimizer(config)
    validator = PipelineValidator()

    # Test each stage
    print("\nStarting pipeline validation...")

    # Stage 1: Seed generation
    seeds = validator.test_seed_generation(optimizer, target_curve, n_test_seeds=10)

    # Stage 2: NSGA-2
    nsga_solutions = validator.test_nsga2_stage(optimizer, seeds, target_curve)

    # Stage 3: Gradient refinement
    if nsga_solutions:
        refined = validator.test_gradient_refinement(optimizer, nsga_solutions, target_curve)

        # Final visualizations
        validator.visualize_full_pipeline(target_curve)
        validator.visualize_best_mechanism(refined, target_curve)
    else:
        print("\nNo NSGA-2 solutions found - check seed quality or NSGA parameters")

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("\nIf results look good, you can scale up by increasing:")
    print("  - n_seeds (try 50-100)")
    print("  - nsga_pop_size (try 100-200)")
    print("  - nsga_n_gen (try 100-300)")
    print("  - max_nodes (try 15-20)")


if __name__ == "__main__":
    run_pipeline_test()