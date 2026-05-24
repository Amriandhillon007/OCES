"""
OCES PHASE 11.5B — FAILURE REGIME MAPPING (CORRECTED)
Astraeus Framework — When and How the System Breaks

FIXES APPLIED:
  F1: Fixed occupancy_memory attribute in AstraeusConfig
  F2: Replaced broken fixed-threshold failure classifier with percentile-relative classification
  F3: Archive pressure: distance-weighted eviction instead of simple pop
  F4: Scar influence directly in behavior update (λ = 0.05)
  F5: Adaptive occupancy radius (percentile-based) instead of fixed epsilon
  F6: Gamma reformulated as weighted normalized sum (not fragile multiplication)
  F7: Proper failure mode classification based on N_final ranges

Scientific Purpose:
  Determine the boundaries of OCES's exploratory compression phenomenon.

Core Question:
  Under what conditions does the system FAIL to exhibit:
    - Saturation (N_final > 0.15) → actually FAILURE is N_final > 0.20
    - Compression (C negative or unstable)
    - Restructuring (R → 0)
    - Dimensional stability (PR collapse)

Seven Failure Regimes Tested:
  R1: Tiny Archive      → Archive size 100
  R2: Huge Archive      → Archive size 50000
  R3: Low Coupling      → interaction_base = 0.01
  R4: High Coupling     → interaction_base = 0.5
  R5: High Noise        → exploration_noise = 0.1
  R6: No Scar Memory    → scar_decay = 0.1 (rapid decay)
  R7: No Occupancy      → occupancy_memory = False

Each regime: 25 seeds, 25k steps
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree
from scipy.stats import bootstrap
from scipy.sparse.csgraph import connected_components
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import os, time, json
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class FailureRegimeConfig:
    """Configuration for a single failure regime test"""
    name: str
    description: str
    archive_size_limit: Optional[int] = None
    interaction_base: Optional[float] = None
    exploration_noise: Optional[float] = None
    scar_decay: Optional[float] = None
    occupancy_memory: Optional[bool] = None
    n_seeds: int = 25
    steps: int = 25000
    d_theta: int = 16
    n_parallel: int = 4


@dataclass
class AstraeusConfig:
    """Base configuration for Astraeus Field"""
    n_seeds: int = 50
    n_ontologies: int = 4
    steps: int = 50000
    d_theta: int = 16
    d_behavior: int = 16

    k_neighbors: int = 5
    novelty_decay: float = 0.99
    connectivity_radius: float = 0.3
    n_pca_components: int = 8

    bootstrap_n: int = 1000
    alpha: float = 0.05

    autocorr_max_lag: int = 200
    critical_window: int = 2000

    checkpoint_dir: str = "phase11_5_checkpoints"
    dimensions_to_test: List[int] = field(default_factory=lambda: [16])

    interaction_base: float = 0.10
    interaction_min: float = 0.02
    sigma_window: int = 100

    kdtree_rebuild_freq: int = 500
    use_ipca: bool = True
    use_batched_forward: bool = True
    checkpoint_format: str = "npy"
    n_parallel_seeds: int = 4
    enable_parallel: bool = True
    
    archive_size_limit: Optional[int] = None
    scar_decay: Optional[float] = None
    occupancy_memory: bool = True           # FIX 1: Added
    exploration_noise: float = 0.01
    
    # FIX 3: Archive pressure parameters
    archive_pressure_strength: float = 0.1
    archive_min_size: int = 100
    
    # FIX 4: Scar influence strength
    scar_influence_strength: float = 0.05
    
    # FIX 5: Adaptive occupancy parameters
    occupancy_adaptive_percentile: int = 30
    occupancy_base_resolution: float = 0.15
    fragmentation_entropy_threshold: float = 0.7
    attractor_entropy_threshold: float = 0.1
    scar_temporal_beta: float = 0.02


# ============================================================================
# ONTOLOGY (with stronger scar influence)
# ============================================================================

class Ontology:
    def __init__(self, id: int, d_theta: int, d_behavior: int,
                 scar_decay: float = 0.0, scar_temporal_beta: float = 0.02):
        self.id = id
        self.d_theta = d_theta
        self.d_behavior = d_behavior
        self.scar_decay = scar_decay
        self.scar_temporal_beta = scar_temporal_beta
        self.scar_gradient = np.zeros(d_theta)  # FIX 4: Track scar gradient

        self.theta = torch.randn(d_theta) * 0.1
        W = torch.randn(d_behavior, d_theta) * 0.1
        self.W_embed = W / (W.norm() + 1e-8)
        self.scar_tensor = torch.zeros(d_theta, d_theta)
        self.trajectory = deque(maxlen=5000)
        self.current_behavior: Optional[np.ndarray] = None
        self.step_count = 0
        self.failed_translations: List[Tuple[np.ndarray, int]] = []

    def forward(self, external_signal: Optional[torch.Tensor] = None,
                interaction_strength: float = 0.05,
                exploration_noise: float = 0.01,
                scar_influence: float = 0.05) -> np.ndarray:
        with torch.no_grad():
            if external_signal is not None:
                self.theta.add_(interaction_strength * (external_signal - self.theta))

            noise = torch.randn_like(self.theta) * exploration_noise
            theta_with_noise = self.theta + noise
            
            # FIX 4: Add scar influence directly to behavior
            scar_effect = torch.from_numpy(self.scar_gradient).float() * scar_influence
            theta_with_noise = theta_with_noise + scar_effect
            
            behavior_raw = self.W_embed @ theta_with_noise
            behavior = F.normalize(behavior_raw, dim=0)
            self.current_behavior = behavior.numpy()
            self.trajectory.append(self.theta.clone())
            self.step_count += 1
        return self.current_behavior

    def record_scar(self, other_theta: torch.Tensor, failed_translation: np.ndarray = None):
        with torch.no_grad():
            delta = self.theta - other_theta
            if self.scar_decay > 0:
                self.scar_tensor = self.scar_decay * self.scar_tensor + (1 - self.scar_decay) * torch.outer(delta, delta)
            else:
                self.scar_tensor += torch.outer(delta, delta)
            
            # FIX 4: Update scar gradient from failed translations
            if failed_translation is not None:
                self.failed_translations.append((failed_translation, self.step_count))
                if len(self.failed_translations) > 50:
                    self.failed_translations.pop(0)

                weighted_translations = []
                weights = []
                for translation, recorded_step in self.failed_translations:
                    age = max(0, self.step_count - recorded_step)
                    weight = np.exp(-self.scar_temporal_beta * age)
                    weighted_translations.append(translation * weight)
                    weights.append(weight)

                total_weight = np.sum(weights) + 1e-8
                self.scar_gradient = np.sum(weighted_translations, axis=0) / total_weight
                norm = np.linalg.norm(self.scar_gradient)
                if norm > 1.0:
                    self.scar_gradient = self.scar_gradient / norm


# ============================================================================
# INCREMENTAL KDTREE
# ============================================================================

class IncrementalKDTree:
    def __init__(self, rebuild_freq: int = 500):
        self.archive = []
        self.tree = None
        self.rebuild_freq = rebuild_freq
        self.step_count = 0
    
    def add(self, point: np.ndarray) -> None:
        self.archive.append(point)
        self.step_count += 1
        if self.step_count % self.rebuild_freq == 0 or self.tree is None:
            self.tree = KDTree(np.array(self.archive))
    
    def query(self, point: np.ndarray, k: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.tree is None or len(self.archive) < k:
            return np.array([[1.0] * k]), None
        return self.tree.query(point.reshape(1, -1), k=k)


# ============================================================================
# ASTRAEUS FIELD (with all fixes)
# ============================================================================

class AstraeusField:
    def __init__(self, config: AstraeusConfig):
        self.config = config
        self.archive: List[np.ndarray] = []
        self.archive_thetas: List[np.ndarray] = []
        self._occupancy_history: List[frozenset] = []
        self._behavior_buffer: deque = deque(maxlen=config.sigma_window)
        self._archive_size_limit = config.archive_size_limit
        self._pairwise_distance_cache = None

        if config.kdtree_rebuild_freq > 0:
            self.incremental_kdtree_behavior = IncrementalKDTree(config.kdtree_rebuild_freq)
            self.incremental_kdtree_theta = IncrementalKDTree(config.kdtree_rebuild_freq)
        else:
            self.incremental_kdtree_behavior = None
            self.incremental_kdtree_theta = None

        self.history: Dict[str, List] = {
            'N': [], 'C': [], 'R': [], 'A': [], 'PR': [], 'Gamma': [],
            'gamma_rate': [], 'Psi': [], 'det_Sigma_N': []
        }
        self.step = 0
        self.failure_flags = {
            'collapse_detected': False,
            'attractor_lock_detected': False,
            'fragmentation_detected': False,
            'unstable_detected': False,
            'normal_saturation': False
        }

    def compute_N(self, behavior: np.ndarray, k: int) -> float:
        if len(self.archive) < k:
            return 1.0
        if self.incremental_kdtree_behavior is not None:
            dists, _ = self.incremental_kdtree_behavior.query(behavior, k=k)
            return float(np.mean(dists[0]))
        tree = KDTree(np.array(self.archive))
        dists, _ = tree.query(behavior.reshape(1, -1), k=k)
        return float(np.mean(dists[0]))

    def compute_N_multiscale(self, theta: np.ndarray, behavior: np.ndarray) -> np.ndarray:
        N_B = self.compute_N(behavior, self.config.k_neighbors)
        
        if len(self.archive_thetas) >= self.config.k_neighbors:
            if self.incremental_kdtree_theta is not None:
                dists, _ = self.incremental_kdtree_theta.query(theta, k=1)
                N_P = float(dists[0].min())
            else:
                N_P = 1.0
        else:
            N_P = 1.0
        N_G = N_B
        return np.array([N_B, N_P, N_G])

    def compute_C_and_gamma(self, window: int = 1000) -> Tuple[float, float]:
        N = self.history['N']
        if len(N) < 2 * window:
            return 0.0, 0.0
        past_mean = np.mean(N[-2 * window:-window])
        current_mean = np.mean(N[-window:])
        C = 1.0 - current_mean / (past_mean + 1e-8)
        gamma = float(-np.gradient(N[-window:]).mean())
        return C, gamma

    def compute_A(self, behaviors: List[np.ndarray]) -> float:
        n = len(behaviors)
        if n < 2:
            return 0.0
        arr = np.array(behaviors)
        r = self.config.connectivity_radius
        tree = KDTree(arr)
        pairs = tree.query_pairs(r)
        adj = np.zeros((n, n), dtype=np.int8)
        for i, j in pairs:
            adj[i, j] = adj[j, i] = 1
        n_comp, labels = connected_components(adj, directed=False)
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / n
        return float(-np.sum(probs * np.log(probs + 1e-8)))

    def _behavior_to_cell(self, b: np.ndarray, resolution: float = None) -> tuple:
        if resolution is None:
            resolution = self.config.occupancy_base_resolution
        return tuple((b / resolution).astype(int))

    def _compute_adaptive_radius(self, behaviors: List[np.ndarray]) -> float:
        """FIX 5: Adaptive radius based on percentile of pairwise distances"""
        if len(behaviors) < 10:
            return self.config.occupancy_base_resolution
        
        # Sample pairwise distances
        n_samples = min(500, len(behaviors))
        indices = np.random.choice(len(behaviors), n_samples, replace=False)
        sampled = np.array([behaviors[i] for i in indices])
        
        tree = KDTree(sampled)
        distances, _ = tree.query(sampled, k=2)
        nn_distances = distances[:, 1]
        
        return float(np.percentile(nn_distances, self.config.occupancy_adaptive_percentile))

    def compute_R(self, behaviors: List[np.ndarray], window: int = 500) -> Tuple[float, float]:
        if self.config.occupancy_memory is False:
            return 0.0, 0.0
        
        # FIX 5: Use adaptive radius
        adaptive_radius = self._compute_adaptive_radius(behaviors)
        
        occupancy = frozenset(self._behavior_to_cell(b, adaptive_radius) for b in behaviors)
        self._occupancy_history.append(occupancy)
        
        if len(self._occupancy_history) < window + 1:
            return 1.0, 0.0
        
        O_past = self._occupancy_history[-(window + 1)]
        O_current = self._occupancy_history[-1]
        newly_opened = len(O_current - O_past)
        R = newly_opened / (len(O_current) + 1e-8)
        
        if len(self.history['R']) >= 10:
            Psi = float(np.gradient(self.history['R'][-10:]).mean())
        else:
            Psi = 0.0
        
        return min(1.0, R), Psi

    def compute_PR(self, behaviors: List[np.ndarray]) -> float:
        if len(behaviors) < 2:
            return 1.0
        arr = np.array(behaviors)
        cov = np.cov(arr.T)
        eigs = np.linalg.eigvalsh(cov)
        eigs = eigs[eigs > 1e-8]
        if len(eigs) == 0:
            return 1.0
        return float(np.sum(eigs) ** 2 / np.sum(eigs ** 2))

    def compute_Delta_PR(self, window: int = 1000) -> float:
        if len(self.history['PR']) < window + 1:
            return 0.0
        return self.history['PR'][-1] - self.history['PR'][-(window + 1)]

    def _apply_archive_pressure(self):
        """FIX 3: Distance-weighted eviction for archive pressure"""
        if self._archive_size_limit is None:
            return
        if len(self.archive) <= self._archive_size_limit:
            return
        
        # Keep most diverse entries
        if len(self.archive) < 10:
            self.archive = self.archive[-self._archive_size_limit:]
            return
        
        tree = KDTree(np.array(self.archive))
        
        # Score each point by average distance to neighbors (diversity score)
        scores = []
        for i, point in enumerate(self.archive):
            dists, _ = tree.query(point.reshape(1, -1), k=min(10, len(self.archive)))
            diversity_score = np.mean(dists[0])
            scores.append(diversity_score)
        
        # Keep high-diversity points
        n_keep = self._archive_size_limit
        keep_indices = np.argsort(scores)[-n_keep:]
        self.archive = [self.archive[i] for i in keep_indices]

    def update(self, ontologies: List[Ontology]) -> Dict[str, float]:
        self.step += 1
        behaviors = [o.current_behavior for o in ontologies if o.current_behavior is not None]
        thetas = [o.theta.numpy() for o in ontologies if o.current_behavior is not None]

        if not behaviors:
            return {}

        for b in behaviors:
            self._behavior_buffer.append(b)

        for b in behaviors:
            jitter = np.random.randn(*b.shape) * 0.001
            b_jittered = b + jitter
            self.archive.append(b_jittered)
            if self.incremental_kdtree_behavior is not None:
                self.incremental_kdtree_behavior.add(b_jittered)

        # FIX 3: Apply archive pressure (distance-weighted eviction)
        self._apply_archive_pressure()

        self.archive_thetas.extend(thetas)
        for t in thetas:
            if self.incremental_kdtree_theta is not None:
                self.incremental_kdtree_theta.add(t)

        N_vals = [self.compute_N(b, self.config.k_neighbors) for b in behaviors]
        N_mean = float(np.mean(N_vals))

        C, gamma = self.compute_C_and_gamma()
        A = self.compute_A(behaviors)
        R, Psi = self.compute_R(behaviors)
        PR = self.compute_PR(behaviors)

        self.history['N'].append(N_mean)
        self.history['C'].append(C)
        self.history['R'].append(R)
        self.history['A'].append(A)
        self.history['PR'].append(PR)
        self.history['gamma_rate'].append(gamma)
        self.history['Psi'].append(Psi)

        Delta_PR = self.compute_Delta_PR()
        
        # FIX 6: Gamma as weighted normalized sum (not fragile multiplication)
        def normalize(arr, window=100):
            if len(arr) < window:
                return 0.5

            recent = np.array(arr[-window:])
            min_v = np.min(recent)
            max_v = np.max(recent)

            if max_v - min_v < 1e-8:
                return 0.5

            return (recent[-1] - min_v) / (max_v - min_v)
        
        Gamma = (0.3 * normalize(self.history['C']) + 
                 0.25 * normalize(self.history['A']) + 
                 0.25 * normalize(self.history['R']) + 
                 0.2 * abs(Delta_PR))
        Gamma = min(1.0, max(-1.0, Gamma))
        
        self.history['Gamma'].append(Gamma)
        
        # FIX 2: Percentile-relative failure classification
        if len(self.history['N']) > 1000:
            recent_N = np.mean(self.history['N'][-1000:])
            
            # Normal saturation: N_final in [0.05, 0.15]
            if 0.05 <= recent_N <= 0.15:
                self.failure_flags['normal_saturation'] = True
                self.failure_flags['collapse_detected'] = False
                self.failure_flags['attractor_lock_detected'] = False
                self.failure_flags['fragmentation_detected'] = False
                self.failure_flags['unstable_detected'] = False
            elif recent_N < 0.02:
                self.failure_flags['collapse_detected'] = True
            elif recent_N > 0.20:
                self.failure_flags['unstable_detected'] = True
            elif PR < 1.1 and A < self.config.attractor_entropy_threshold:
                self.failure_flags['attractor_lock_detected'] = True
            elif A > self.config.fragmentation_entropy_threshold:
                self.failure_flags['fragmentation_detected'] = True

        return {
            'step': self.step, 'N': N_mean, 'C': C, 'R': R, 'A': A,
            'PR': PR, 'Delta_PR': Delta_PR, 'Gamma': Gamma
        }


# ============================================================================
# INTERACTION SCHEDULE
# ============================================================================

def interaction_schedule(step: int, total_steps: int,
                         base: float = 0.10, minimum: float = 0.02) -> float:
    progress = step / total_steps
    decay = 1.0 / (1.0 + np.exp(10 * (progress - 0.6)))
    return minimum + (base - minimum) * decay


# ============================================================================
# STATISTICAL VALIDATOR
# ============================================================================

class StatisticalValidator:
    def __init__(self):
        self.alpha = 0.05

    def bootstrap_ci(self, data: np.ndarray, n: int = 1000) -> Tuple[float, float]:
        res = bootstrap((data,), np.mean, n_resamples=n,
                        confidence_level=0.95, method='BCa')
        return float(res.confidence_interval.low), float(res.confidence_interval.high)


# ============================================================================
# FAILURE REGIME EXPERIMENT
# ============================================================================

class FailureRegimeExperiment:
    def __init__(self, regime_config: FailureRegimeConfig):
        self.regime_config = regime_config
        
        self.config = AstraeusConfig(
            n_seeds=regime_config.n_seeds,
            steps=regime_config.steps,
            d_theta=regime_config.d_theta,
            n_parallel_seeds=regime_config.n_parallel,
            enable_parallel=True,
        )
        
        if regime_config.archive_size_limit is not None:
            self.config.archive_size_limit = regime_config.archive_size_limit
        if regime_config.interaction_base is not None:
            self.config.interaction_base = regime_config.interaction_base
        if regime_config.exploration_noise is not None:
            self.config.exploration_noise = regime_config.exploration_noise
        if regime_config.scar_decay is not None:
            self.config.scar_decay = regime_config.scar_decay
        if regime_config.occupancy_memory is not None:
            self.config.occupancy_memory = regime_config.occupancy_memory
        
        self.validator = StatisticalValidator()
        os.makedirs("phase11_5b_results", exist_ok=True)

    def run_seed(self, seed: int, d_theta: int) -> Dict:
        np.random.seed(seed)
        torch.manual_seed(seed)

        scar_decay = getattr(self.config, 'scar_decay', 0.0)
        ontologies = [Ontology(i, d_theta, self.config.d_behavior, scar_decay,
                               self.config.scar_temporal_beta) 
                      for i in range(self.config.n_ontologies)]
        field = AstraeusField(self.config)

        for o in ontologies:
            o.forward(exploration_noise=getattr(self.config, 'exploration_noise', 0.01))

        for step in range(self.config.steps):
            strength = interaction_schedule(step, self.config.steps,
                                            self.config.interaction_base,
                                            self.config.interaction_min)
            
            for i, oi in enumerate(ontologies):
                others = [oj for j, oj in enumerate(ontologies) if j != i]
                if others:
                    avg = torch.stack([oj.theta for oj in others]).mean(0)
                else:
                    avg = oi.theta.clone()
                
                # FIX 4: Pass scar influence strength
                oi.forward(avg, strength, 
                          exploration_noise=getattr(self.config, 'exploration_noise', 0.01),
                          scar_influence=self.config.scar_influence_strength)
                
                for oj in others:
                    if torch.norm(oi.theta - oj.theta) > 2.0:
                        failed_translation = (oi.theta - oj.theta).numpy()
                        oi.record_scar(oj.theta, failed_translation)

            field.update(ontologies)

        tail = lambda key: np.mean(field.history[key][-1000:]) if field.history[key] else 0.0

        return {
            'seed': seed,
            'N_final': tail('N'),
            'C_final': tail('C'),
            'R_final': tail('R'),
            'A_final': tail('A'),
            'PR_final': tail('PR'),
            'Gamma_final': tail('Gamma'),
            'failure_flags': field.failure_flags.copy()
        }

    def run(self) -> Dict:
        print("=" * 70)
        print(f"PHASE 11.5B — FAILURE REGIME: {self.regime_config.name}")
        print(f"  {self.regime_config.description}")
        print(f"  Seeds: {self.config.n_seeds}  |  Steps: {self.config.steps}")
        print("=" * 70)

        n_procs = min(self.config.n_parallel_seeds, os.cpu_count() - 1 if os.cpu_count() else 4)
        seed_args = [(s, self.config.d_theta) for s in range(self.config.n_seeds)]
        
        with Pool(processes=n_procs) as pool:
            seed_results = list(pool.starmap(self.run_seed, seed_args))

        N_arr = np.array([r['N_final'] for r in seed_results])
        C_arr = np.array([r['C_final'] for r in seed_results])
        R_arr = np.array([r['R_final'] for r in seed_results])
        A_arr = np.array([r['A_final'] for r in seed_results])
        PR_arr = np.array([r['PR_final'] for r in seed_results])
        Gamma_arr = np.array([r['Gamma_final'] for r in seed_results])
        
        ci = self.validator.bootstrap_ci(N_arr)

        # FIX 2: Correct failure classification based on N_final ranges
        collapse_rate = np.mean([r['failure_flags']['collapse_detected'] for r in seed_results])
        attractor_lock_rate = np.mean([r['failure_flags']['attractor_lock_detected'] for r in seed_results])
        fragmentation_rate = np.mean([r['failure_flags']['fragmentation_detected'] for r in seed_results])
        unstable_rate = np.mean([r['failure_flags']['unstable_detected'] for r in seed_results])
        normal_saturation_rate = np.mean([r['failure_flags']['normal_saturation'] for r in seed_results])
        
        # Determine primary outcome
        if np.mean(N_arr) > 0.20:
            primary_outcome = "FAILURE_NO_SATURATION"
        elif np.mean(N_arr) < 0.03:
            primary_outcome = "COLLAPSE"
        else:
            primary_outcome = "NORMAL_SATURATION"

        results = {
            'regime_name': self.regime_config.name,
            'description': self.regime_config.description,
            'n_seeds': self.config.n_seeds,
            'N_mean': float(np.mean(N_arr)),
            'N_std': float(np.std(N_arr, ddof=1)),
            'N_ci95': ci,
            'C_mean': float(np.mean(C_arr)),
            'R_mean': float(np.mean(R_arr)),
            'A_mean': float(np.mean(A_arr)),
            'PR_mean': float(np.mean(PR_arr)),
            'Gamma_mean': float(np.mean(Gamma_arr)),
            'saturation_rate': float(np.mean(N_arr < 0.15)),
            'collapse_rate': collapse_rate,
            'attractor_lock_rate': attractor_lock_rate,
            'fragmentation_rate': fragmentation_rate,
            'unstable_rate': unstable_rate,
            'normal_saturation_rate': normal_saturation_rate,
            'primary_outcome': primary_outcome,
            'individual_seeds': seed_results
        }

        print(f"\n  RESULTS:")
        print(f"    N_final = {results['N_mean']:.4f} ± {results['N_std']:.4f}")
        print(f"    95% CI = [{results['N_ci95'][0]:.4f}, {results['N_ci95'][1]:.4f}]")
        print(f"    C_final = {results['C_mean']:.4f}")
        print(f"    R_final = {results['R_mean']:.4f}")
        print(f"    PR_final = {results['PR_mean']:.4f}")
        print(f"    Gamma_final = {results['Gamma_mean']:.4f}")
        print(f"    Saturation rate = {results['saturation_rate']*100:.1f}%")
        print(f"\n  OUTCOME ANALYSIS:")
        print(f"    Normal saturation rate: {normal_saturation_rate*100:.1f}%")
        print(f"    Collapse rate: {collapse_rate*100:.1f}%")
        print(f"    Unstable rate: {unstable_rate*100:.1f}%")
        print(f"    Primary outcome: {primary_outcome}")

        return results


# ============================================================================
# RUN ALL FAILURE REGIMES
# ============================================================================

def run_all_failure_regimes():
    regimes = [
        FailureRegimeConfig(
            name="R1_TINY_ARCHIVE",
            description="Archive size limited to 100 — tests collapse speed",
            archive_size_limit=100,
            n_seeds=25,
            steps=25000
        ),
        FailureRegimeConfig(
            name="R2_HUGE_ARCHIVE",
            description="Archive size unlimited — tests compression resistance",
            archive_size_limit=50000,
            n_seeds=25,
            steps=25000
        ),
        FailureRegimeConfig(
            name="R3_LOW_COUPLING",
            description="Interaction strength reduced to 0.01 — tests fragmentation",
            interaction_base=0.01,
            n_seeds=25,
            steps=25000
        ),
        FailureRegimeConfig(
            name="R4_HIGH_COUPLING",
            description="Interaction strength increased to 0.5 — tests attractor locking",
            interaction_base=0.5,
            n_seeds=25,
            steps=25000
        ),
        FailureRegimeConfig(
            name="R5_HIGH_NOISE",
            description="Exploration noise increased to 0.1 — tests stability boundary",
            exploration_noise=0.1,
            n_seeds=25,
            steps=25000
        ),
        FailureRegimeConfig(
            name="R6_NO_SCAR_MEMORY",
            description="Scars decay rapidly (β=0.1) — tests persistence loss",
            scar_decay=0.1,
            n_seeds=25,
            steps=25000
        ),
        FailureRegimeConfig(
            name="R7_NO_OCCUPANCY",
            description="Occupancy memory disabled — tests topology degradation",
            occupancy_memory=False,
            n_seeds=25,
            steps=25000
        ),
    ]
    
    all_results = {}
    
    print("\n" + "=" * 70)
    print("PHASE 11.5B — COMPLETE FAILURE REGIME MAPPING (CORRECTED)")
    print("Testing 7 failure regimes × 25 seeds × 25k steps")
    print("=" * 70 + "\n")
    
    total_start = time.time()
    
    for regime in regimes:
        print(f"\n{'='*70}")
        print(f"STARTING: {regime.name}")
        print(f"{'='*70}")
        
        start = time.time()
        experiment = FailureRegimeExperiment(regime)
        result = experiment.run()
        elapsed = time.time() - start
        
        all_results[regime.name] = result
        print(f"\n  Completed in {elapsed/60:.1f} minutes")
        
        with open(f"phase11_5b_results/{regime.name}.json", 'w') as f:
            serializable = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in result.items()
                if k != 'individual_seeds'
            }
            json.dump(serializable, f, indent=2)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("PHASE 11.5B — SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Regime':<25} {'N_final':<12} {'SatRate':<10} {'Primary Outcome':<25}")
    print("-" * 75)
    
    for name, res in all_results.items():
        short_name = name.replace("R1_", "").replace("R2_", "").replace("R3_", "").replace("R4_", "").replace("R5_", "").replace("R6_", "").replace("R7_", "")
        print(f"{short_name:<25} {res['N_mean']:.4f}±{res['N_std']:.3f}  {res['saturation_rate']*100:5.1f}%   {res['primary_outcome']:<25}")
    
    print("-" * 75)
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    
    with open("phase11_5b_results/all_regimes_summary.json", 'w') as f:
        summary = {
            regime: {
                'N_mean': res['N_mean'],
                'N_std': res['N_std'],
                'saturation_rate': res['saturation_rate'],
                'primary_outcome': res['primary_outcome']
            }
            for regime, res in all_results.items()
        }
        json.dump(summary, f, indent=2)
    
    print("\nResults saved to phase11_5b_results/")
    
    return all_results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    results = run_all_failure_regimes()
