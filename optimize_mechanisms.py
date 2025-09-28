"""
Mechanism Optimization Pipeline
Goal: Generate diverse seed mechanisms → Run NSGA-2 → Refine with gradient descent
Constraints: Material < 10, Nodes < 20
"""

import os
# Force CPU to avoid Metal/JAX issues
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from tqdm import tqdm

from LINKS.Optimization import Tools, DifferentiableTools, MechanismRandomizer


@dataclass
class OptimizationConfig:
    """Configuration for the optimization pipeline"""
    # Seed generation
    n_seeds: int = 50
    min_nodes: int = 6
    max_nodes: int = 20
    material_threshold: float = 10.0

    # NSGA-2
    nsga_pop_size: int = 100
    nsga_n_gen: int = 200

    # Gradient refinement
    lbfgs_maxiter: int = 200
    penalty_lambda: float = 100.0

    # Selection
    max_refined: int = 20  # Max solutions to refine per curve


class MechanismOptimizer:
    """Clean, modular optimizer for mechanism synthesis"""

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()

        # Initialize tools
        self.tools = Tools(device='cpu', material=True)
        self.tools.compile()

        self.diff_tools = DifferentiableTools(device='cpu', material=True)
        self.diff_tools.compile()

        self.randomizer = MechanismRandomizer(
            min_size=self.config.min_nodes,
            max_size=self.config.max_nodes,
            device='cpu'
        )

    def generate_seeds(self, target_curve: np.ndarray) -> List[Dict]:
        """
        Generate diverse seed mechanisms with material < threshold
        """
        seeds = []
        attempts = 0
        max_attempts = self.config.n_seeds * 50  # More attempts
        materials_tested = []

        print(f"Generating {self.config.n_seeds} seed mechanisms...")
        print(f"Material threshold: {self.config.material_threshold}")

        while len(seeds) < self.config.n_seeds and attempts < max_attempts:
            attempts += 1

            # Random size within bounds
            n = np.random.randint(self.config.min_nodes, self.config.max_nodes + 1)

            try:
                # Generate random mechanism with more relaxed constraints
                mech = self.randomizer(n=n, n_tests=64, max_tries=200)

                # Handle different return formats from randomizer
                if isinstance(mech, (list, tuple)):
                    if len(mech) == 0:
                        continue
                    mech = mech[0] if len(mech) == 1 else mech[np.random.randint(len(mech))]

                # Validate size
                if mech['x0'].shape[0] != n:
                    continue

                # Check material constraint
                _, material = self.evaluate_mechanism(mech, target_curve)
                materials_tested.append(material)

                # Accept if under threshold OR if we're struggling to find valid ones
                if material <= self.config.material_threshold:
                    seeds.append(mech)
                    print(f"  Found seed {len(seeds)}/{self.config.n_seeds}: n={n}, material={material:.2f}")
                elif len(seeds) == 0 and attempts > max_attempts // 2:
                    # If we're halfway through and have no seeds, relax constraint slightly
                    relaxed_threshold = self.config.material_threshold * 1.2
                    if material <= relaxed_threshold:
                        seeds.append(mech)
                        print(f"  Accepted seed with relaxed constraint: n={n}, material={material:.2f}")

            except (ValueError, RuntimeError) as e:
                continue

        if len(seeds) == 0 and materials_tested:
            # If no valid seeds, take the best we found
            print(f"\nNo seeds met material constraint. Materials found: min={min(materials_tested):.2f}, mean={np.mean(materials_tested):.2f}")
            print("Retrying with relaxed constraints...")

            # Try again with just taking best materials
            attempts = 0
            all_mechs = []

            while len(all_mechs) < self.config.n_seeds * 2 and attempts < 100:
                attempts += 1
                n = np.random.randint(self.config.min_nodes, self.config.max_nodes + 1)
                try:
                    mech = self.randomizer(n=n, n_tests=64, max_tries=200)
                    if isinstance(mech, (list, tuple)):
                        if len(mech) == 0:
                            continue
                        mech = mech[0] if len(mech) == 1 else mech[np.random.randint(len(mech))]

                    _, material = self.evaluate_mechanism(mech, target_curve)
                    all_mechs.append((material, mech))
                except:
                    continue

            # Sort by material and take best ones
            all_mechs.sort(key=lambda x: x[0])
            seeds = [m[1] for m in all_mechs[:self.config.n_seeds]]
            print(f"Selected {len(seeds)} best mechanisms by material")

        if len(seeds) < self.config.n_seeds:
            print(f"Warning: Only generated {len(seeds)} valid seeds out of {self.config.n_seeds} requested")

        return seeds

    def evaluate_mechanism(self, mech: Dict, target_curve: np.ndarray) -> Tuple[float, float]:
        """Evaluate a mechanism's distance and material"""
        result = self.tools(
            mech['x0'],
            mech['edges'],
            mech['fixed_joints'],
            mech['motor'],
            target_curve,
            target_idx=mech.get('target_joint', mech['x0'].shape[0] - 1)
        )

        # Handle different return formats
        if isinstance(result, tuple):
            distance, material = result
        else:
            distance = float(result)
            material = np.inf

        # Ensure scalar values
        distance = float(np.asarray(distance).item())
        material = float(np.asarray(material).item())

        # Handle invalid values
        if not np.isfinite(distance):
            distance = 1e9
        if not np.isfinite(material):
            material = 1e9

        return distance, material

    def run_nsga2(self, mech: Dict, target_curve: np.ndarray) -> Optional[Dict]:
        """
        Run NSGA-2 for a single mechanism topology
        Returns best solutions found
        """
        n = mech['x0'].shape[0]

        class MechProblem(Problem):
            def __init__(self, outer_self):
                self.outer = outer_self
                super().__init__(
                    n_var=n*2,
                    n_obj=2,
                    n_ieq_constr=2,  # distance <= 0.75, material <= 10
                    xl=0.0,
                    xu=1.0
                )

            def _evaluate(self, X, out, *args, **kwargs):
                pop_size = X.shape[0]
                F = np.zeros((pop_size, 2))
                G = np.zeros((pop_size, 2))

                for i in range(pop_size):
                    x0 = X[i].reshape(n, 2)

                    # Create temporary mech with new positions
                    temp_mech = mech.copy()
                    temp_mech['x0'] = x0

                    distance, material = self.outer.evaluate_mechanism(temp_mech, target_curve)

                    F[i, 0] = distance
                    F[i, 1] = material

                    # Constraints
                    G[i, 0] = distance - 0.75
                    G[i, 1] = material - 10.0

                out["F"] = F
                out["G"] = G

        # Run NSGA-2
        problem = MechProblem(self)
        algorithm = NSGA2(pop_size=self.config.nsga_pop_size)
        termination = get_termination("n_gen", self.config.nsga_n_gen)

        result = pymoo_minimize(
            problem,
            algorithm,
            termination,
            verbose=False,
            seed=np.random.randint(10000)
        )

        if result.X is None or result.F is None:
            return None

        # Return best solutions
        return {
            'mech': mech,
            'X': result.X,  # Position arrays
            'F': result.F   # Objective values
        }

    def gradient_refine(self, mech: Dict, x0_init: np.ndarray,
                       target_curve: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Refine positions using L-BFGS-B with gradient information
        """
        n = x0_init.shape[0]

        def objective_with_grad(x_flat):
            x0 = x_flat.reshape(n, 2)

            # Get objectives and gradients
            dist, mat, dist_grad, mat_grad = self.diff_tools(
                x0,
                mech['edges'],
                mech['fixed_joints'],
                mech['motor'],
                target_curve,
                mech.get('target_joint', n-1)
            )

            # Convert to scalars
            dist = float(np.asarray(dist).item())
            mat = float(np.asarray(mat).item())

            # Handle invalid values
            if not np.isfinite(dist) or not np.isfinite(mat):
                return 1e9, np.zeros_like(x_flat)

            dist_grad = np.asarray(dist_grad).reshape(-1)
            mat_grad = np.asarray(mat_grad).reshape(-1)

            # Penalty for material constraint
            penalty = 0.0
            penalty_grad = np.zeros_like(x_flat)

            if mat > self.config.material_threshold:
                excess = mat - self.config.material_threshold
                penalty = self.config.penalty_lambda * excess**2
                penalty_grad = 2 * self.config.penalty_lambda * excess * mat_grad

            # Total objective: minimize distance with material penalty
            obj = dist + penalty
            grad = dist_grad + penalty_grad

            return obj, grad

        # Run L-BFGS-B
        result = minimize(
            fun=objective_with_grad,
            x0=x0_init.ravel(),
            method='L-BFGS-B',
            jac=True,
            bounds=[(0, 1)] * (2*n),
            options={'maxiter': self.config.lbfgs_maxiter}
        )

        # Get final positions
        x0_final = result.x.reshape(n, 2)

        # Evaluate final performance
        temp_mech = mech.copy()
        temp_mech['x0'] = x0_final
        distance, material = self.evaluate_mechanism(temp_mech, target_curve)

        return x0_final, distance, material

    def optimize_for_curve(self, target_curve: np.ndarray, curve_idx: int) -> List[Dict]:
        """
        Complete optimization pipeline for a single target curve
        """
        print(f"\n{'='*60}")
        print(f"Optimizing for Target Curve {curve_idx + 1}")
        print(f"{'='*60}")

        # Step 1: Generate seeds
        seeds = self.generate_seeds(target_curve)
        print(f"Generated {len(seeds)} valid seed mechanisms")

        # Step 2: Run NSGA-2 for each seed
        print(f"Running NSGA-2 on each seed mechanism...")
        nsga_results = []

        for i, seed in enumerate(tqdm(seeds, desc="NSGA-2")):
            result = self.run_nsga2(seed, target_curve)
            if result is not None:
                nsga_results.append(result)

        if not nsga_results:
            print("Warning: No NSGA-2 runs produced valid results")
            return []

        # Step 3: Collect and filter Pareto front
        all_solutions = []
        for res in nsga_results:
            mech = res['mech']
            if res['X'].ndim == 1:
                # Single solution
                all_solutions.append({
                    'mech': mech,
                    'x0': res['X'].reshape(mech['x0'].shape),
                    'distance': res['F'][0] if res['F'].ndim == 1 else res['F'][0, 0],
                    'material': res['F'][1] if res['F'].ndim == 1 else res['F'][0, 1]
                })
            else:
                # Multiple solutions
                for j in range(res['X'].shape[0]):
                    all_solutions.append({
                        'mech': mech,
                        'x0': res['X'][j].reshape(mech['x0'].shape),
                        'distance': res['F'][j, 0],
                        'material': res['F'][j, 1]
                    })

        # Filter to Pareto front
        pareto_solutions = self.get_pareto_front(all_solutions)
        print(f"Pareto front contains {len(pareto_solutions)} solutions")

        # Step 4: Select top solutions for gradient refinement
        # Prioritize feasible solutions with good distance
        feasible = [s for s in pareto_solutions if s['material'] <= self.config.material_threshold]
        to_refine = feasible[:self.config.max_refined] if feasible else pareto_solutions[:self.config.max_refined]

        # Step 5: Gradient-based refinement
        print(f"Refining {len(to_refine)} solutions with L-BFGS-B...")
        refined_solutions = []

        for sol in tqdm(to_refine, desc="Gradient Refinement"):
            x0_refined, dist, mat = self.gradient_refine(
                sol['mech'],
                sol['x0'],
                target_curve
            )

            refined_solutions.append({
                'mech': sol['mech'],
                'x0': x0_refined,
                'distance': dist,
                'material': mat
            })

        # Final Pareto filtering
        final_solutions = self.get_pareto_front(refined_solutions)

        # Sort by distance for submission
        final_solutions.sort(key=lambda x: x['distance'])

        print(f"Final: {len(final_solutions)} solutions")
        if final_solutions:
            best = final_solutions[0]
            print(f"Best: distance={best['distance']:.4f}, material={best['material']:.4f}")

        return final_solutions

    def get_pareto_front(self, solutions: List[Dict]) -> List[Dict]:
        """Extract Pareto-optimal solutions"""
        if not solutions:
            return []

        # Extract objectives
        objectives = np.array([[s['distance'], s['material']] for s in solutions])

        # Find non-dominated solutions
        n = len(solutions)
        is_dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                # Check if j dominates i
                if (objectives[j] <= objectives[i]).all() and (objectives[j] < objectives[i]).any():
                    is_dominated[i] = True
                    break

        return [s for i, s in enumerate(solutions) if not is_dominated[i]]

    def optimize_all_curves(self, target_curves: np.ndarray) -> Dict[int, List[Dict]]:
        """
        Run optimization for all target curves
        Returns dictionary mapping curve index to solutions
        """
        all_results = {}

        for i in range(len(target_curves)):
            solutions = self.optimize_for_curve(target_curves[i], i)
            all_results[i] = solutions

        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        for i, solutions in all_results.items():
            feasible = [s for s in solutions if s['material'] <= 10.0]
            print(f"Curve {i+1}: {len(solutions)} solutions ({len(feasible)} feasible)")

        return all_results


def create_submission(results: Dict[int, List[Dict]], max_per_curve: int = 5) -> List[List[Dict]]:
    """
    Format results for submission
    Returns list of 6 lists (one per curve)
    """
    submission = []

    for curve_idx in range(6):
        curve_solutions = results.get(curve_idx, [])

        # Take top solutions by distance
        curve_solutions = sorted(curve_solutions, key=lambda x: x['distance'])[:max_per_curve]

        # Format for submission
        formatted = []
        for sol in curve_solutions:
            mech = sol['mech'].copy()
            mech['x0'] = sol['x0']
            formatted.append({
                'x0': mech['x0'],
                'edges': mech['edges'],
                'fixed_joints': mech['fixed_joints'],
                'motor': mech['motor'],
                'target_joint': mech.get('target_joint', mech['x0'].shape[0] - 1)
            })

        submission.append(formatted)

    return submission


if __name__ == "__main__":
    # Load target curves
    target_curves = np.load('target_curves.npy')

    # Create optimizer with custom config
    config = OptimizationConfig(
        n_seeds=30,  # Adjust based on computational budget
        nsga_pop_size=100,
        nsga_n_gen=150,
        max_refined=15
    )

    optimizer = MechanismOptimizer(config)

    # Run optimization for all curves
    results = optimizer.optimize_all_curves(target_curves)

    # Create submission format
    submission = create_submission(results)

    # Save results
    np.save('optimized_mechanisms.npy', submission)
    print("\nResults saved to optimized_mechanisms.npy")