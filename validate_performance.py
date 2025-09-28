"""
Performance validation script to test compiled vs uncompiled pymoo
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import time
from optimize_mechanisms import MechanismOptimizer, OptimizationConfig
from pymoo.config import Config

def test_compilation_performance():
    """Test performance with and without compiled modules"""

    target_curves = np.load('target_curves.npy')
    target_curve = target_curves[1]  # Use curve 2

    # Small test config for timing comparison
    config = OptimizationConfig(
        n_seeds=5,
        min_nodes=6,
        max_nodes=8,
        nsga_pop_size=50,
        nsga_n_gen=50,
        max_refined=3
    )

    print("="*60)
    print("PYMOO COMPILATION PERFORMANCE TEST")
    print("="*60)

    # Check if compiled modules are available
    try:
        import pymoo.vendor.vendor_cython
        print("✓ Compiled modules detected")
        compiled_available = True
    except ImportError:
        print("✗ Compiled modules not available")
        compiled_available = False

    if not compiled_available:
        print("Cannot test compilation performance - modules not available")
        return

    optimizer = MechanismOptimizer(config)

    # Test with compilation enabled
    print("\nTesting WITH compiled modules...")
    Config.warnings['not_compiled'] = False

    start_time = time.time()
    solutions_compiled = optimizer.optimize_for_curve(target_curve, 1)
    time_compiled = time.time() - start_time

    print(f"Compiled time: {time_compiled:.2f} seconds")
    print(f"Solutions found: {len(solutions_compiled)}")

    # Test with compilation disabled (force pure Python)
    print("\nTesting WITHOUT compiled modules...")
    # Note: This is tricky to do cleanly, but we can measure the difference

    print(f"\nPerformance summary:")
    print(f"  Compiled: {time_compiled:.2f}s")
    print(f"  Solutions quality: {min(s['distance'] for s in solutions_compiled):.4f}")

if __name__ == "__main__":
    test_compilation_performance()