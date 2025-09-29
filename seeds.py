# seeds.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Callable
import numpy as np

from LINKS.Optimization._MechanismRandomizer import MechanismRandomizer as _Randomizer
from LINKS.Optimization._Tools import Tools as _Tools

@dataclass
class SeederConfig:
    # How many mechanisms to return per curve
    n_seeds: int = 64

    # Randomizer settings
    min_nodes: int = 6
    max_nodes: int = 20
    fixed_probability: float = 0.15
    timesteps: int = 200
    device: str = "gpu"          # 'cpu' or 'gpu'

    # Random search per mechanism:
    n_tests_per_mech: int = 32   # how many x0 trials per skeleton
    max_tries_per_mech: int = 100

    # Optional pre-evaluation of seeds (for logging / triage, doesn’t filter)
    preevaluate: bool = True
    scaled_distance: bool = False  # pass through to Tools
    include_material: bool = True  # pass through to Tools

    # Reproducibility
    seed: Optional[int] = 42

    # JIT warmup for the randomizer (helps first-call latency on GPU)
    warmup_randomizer: bool = True
    warmup_batch_size: int = 32
    warmup_n_tests: int = 16


class MechanismSeeder:
    """
    Thin wrapper around LINKS MechanismRandomizer that:
      * generates valid random mechanisms (diverse sizes),
      * (optionally) warms up JIT for smoother first-run timing,
      * (optionally) pre-evaluates distance/material for quick logging,
      * returns a simple list[dict] compatible with the grader’s schema.
    """

    def __init__(
        self,
        cfg: SeederConfig,
        randomizer_cls: Callable[..., Any] = _Randomizer,
        tools_cls: Callable[..., Any] = _Tools,
    ):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()
        self._randomizer = randomizer_cls(
            min_size=cfg.min_nodes,
            max_size=cfg.max_nodes,
            fixed_probability=cfg.fixed_probability,
            timesteps=cfg.timesteps,
            device=cfg.device,
        )
        self._tools = None
        if cfg.preevaluate:
            self._tools = tools_cls(
                timesteps=cfg.timesteps,
                max_size=cfg.max_nodes,
                material=cfg.include_material,
                scaled=cfg.scaled_distance,
                device=cfg.device,
            )

        # Optional JIT warmup (highly recommended on GPU)
        if cfg.warmup_randomizer:
            try:
                self._randomizer.full_batch_compile(
                    target_batch_size=cfg.warmup_batch_size,
                    target_n_tests=cfg.warmup_n_tests,
                )
            except Exception:
                # Warmup is a nice-to-have; don’t fail the run if unavailable
                pass

        # JIT compile distance/material if requested
        if self._tools is not None:
            try:
                self._tools.compile()
            except Exception:
                pass

    def _gen_one(self) -> Dict[str, Any]:
        """
        Generate a single valid mechanism using the library's rejection-sampling
        (tries up to cfg.max_tries_per_mech with cfg.n_tests_per_mech x0 candidates).
        """
        # Important: use numpy's global random state for the library calls.
        # The library itself calls np.random.*; we set the bitgen for determinism.
        state = np.random.get_state()
        try:
            # bridge our Generator seed into legacy np.random.* used internally
            # CRITICAL: Generate a fresh random seed for each mechanism to ensure diversity
            np.random.seed(self._rng.integers(0, 2**32 - 1, dtype=np.uint32))
            mech = self._randomizer(
                n=None,  # let it pick uniformly in [min_nodes, max_nodes]
                n_tests=self.cfg.n_tests_per_mech,
                max_tries=self.cfg.max_tries_per_mech,
            )
        finally:
            # restore global state to avoid side-effects outside the seeder
            np.random.set_state(state)
        return mech  # dict with keys: x0, edges, fixed_joints, motor

    def _eval_batch(
        self,
        mechanisms: List[Dict[str, Any]],
        target_curve: np.ndarray,
        target_joint: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fast objective eval using Tools (no gradients).
        Returns:
          distances: (N,)
          materials: (N,) or None if cfg.include_material=False
        """
        X = [m["x0"] for m in mechanisms]
        E = [m["edges"] for m in mechanisms]
        FJ = [m["fixed_joints"] for m in mechanisms]
        M = [m["motor"] for m in mechanisms]

        out = self._tools(X, E, FJ, M, target_curve, target_idx=None)
        if self.cfg.include_material:
            distances, materials = out
            return np.asarray(distances), np.asarray(materials)
        else:
            distances = out
            return np.asarray(distances), None

    def generate_for_curve(
        self,
        target_curve: np.ndarray,
        n_seeds: Optional[int] = None,
        target_joint: Optional[int] = None,
        annotate_metrics: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Main entry point.
        Returns a list of mechanisms (length = n_seeds), each a dict:
            {
              "x0": (n,2) float in [0,1],
              "edges": (m,2) int,
              "fixed_joints": (k,) int,
              "motor": (2,) int,
              "target_joint": int,     # added here for downstream consistency
              # (optional) "distance": float,
              # (optional) "material": float,
            }
        """
        n = n_seeds or self.cfg.n_seeds

        # 1) Generate
        mechs: List[Dict[str, Any]] = [self._gen_one() for _ in range(n)]
        # Attach target_joint now to make downstream code trivial
        for m in mechs:
            m["target_joint"] = target_joint if target_joint is not None else int(m["x0"].shape[0] - 1)

        # 2) Optional quick evaluation (for logging/triage/plots)
        if self.cfg.preevaluate and annotate_metrics and len(mechs) > 0:
            d, mat = self._eval_batch(mechs, target_curve, target_joint)
            for i, m in enumerate(mechs):
                m["distance"] = float(d[i])
                if mat is not None:
                    m["material"] = float(mat[i])

        return mechs

    def generate_pool(self, n_total: int) -> list[dict]:
        """
        Curve-agnostic superset of mechanisms. No pre-eval, no target binding.
        Returns a list of mechanism dicts as produced by _gen_one().
        """
        import copy

        made = []
        # Temporarily disable any curve-specific plumbing if your class keeps tools cached
        prev_tools = getattr(self, "_tools", None)
        if hasattr(self, "_tools"):
            self._tools = None
        try:
            tries = 0
            while len(made) < n_total:
                tries += 1
                try:
                    m = self._gen_one()
                    # ensure it's detached from any internal buffers
                    made.append(copy.deepcopy(m))
                except (RecursionError, RuntimeError, ValueError):
                    # keep going; pool generation should be resilient
                    continue
                # optional: hard cap on pathological loops
                if tries > n_total * 100:
                    break
            return made
        finally:
            if hasattr(self, "_tools"):
                self._tools = prev_tools

    def select_for_curve(
        self,
        pool: list[dict],
        target_curve,
        k: int,
        strategy: str = "diverse_by_nodes",
        annotate_metrics: bool = False,
        rng_seed: int | None = None,
    ) -> list[dict]:
        """
        Choose k mechanisms from the global pool to seed a given curve.
        - Sets per-curve fields (e.g., target_joint) per mechanism.
        - Optionally annotates quick metrics via a lightweight batch eval.
        - strategy: "random" | "diverse_by_nodes"
        """
        import numpy as np, copy

        if not pool:
            return []

        rng = np.random.default_rng(rng_seed)
        # Work on copies so we never mutate the shared pool
        candidates = [copy.deepcopy(m) for m in pool]

        if strategy == "random":
            idx = rng.choice(
                len(candidates), size=min(k, len(candidates)), replace=False
            )
            chosen = [candidates[i] for i in idx]

        elif strategy == "diverse_by_nodes":
            # round-robin across node-count buckets to keep topological variety
            from collections import defaultdict, deque

            buckets = defaultdict(list)
            for m in candidates:
                n_nodes = int(
                    m["x0"].shape[0]
                )  # adapt if your dict stores size differently
                buckets[n_nodes].append(m)
            # shuffle each bucket
            for kk in list(buckets.keys()):
                rng.shuffle(buckets[kk])
                buckets[kk] = deque(buckets[kk])

            chosen = []
            order = deque(sorted(buckets.keys()))
            while len(chosen) < k and order:
                key = order[0]
                if buckets[key]:
                    chosen.append(buckets[key].popleft())
                    order.rotate(-1)
                else:
                    order.popleft()

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Per-curve attachment (minimal): pick an end joint as target unless you have a rule
        for m in chosen:
            n_nodes = int(m["x0"].shape[0])
            m["target_joint"] = n_nodes - 1

        # (Optional) annotate quick metrics so you can log/triage seeds before NSGA
        if (
            annotate_metrics
            and hasattr(self, "_tools")
            and self._tools is not None
            and len(chosen) > 0
        ):
            try:
                dists, mats = self._eval_batch(chosen, target_curve, target_joint=None)
                for i, m in enumerate(chosen):
                    m["distance"] = float(dists[i])
                    if mats is not None:
                        m["material"] = float(mats[i])
            except Exception:
                # metrics are best-effort; swallow errors to avoid blocking seeding
                pass

        return chosen
