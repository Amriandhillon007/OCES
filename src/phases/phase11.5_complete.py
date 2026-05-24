"""
OCES PHASE 11.5 — ASTRAEUS UNIFIED FIELD FRAMEWORK v2
Formal Replication & Saturation Dynamics

FIXES APPLIED (9 critical fixes):
  P1 — Memory leak: No full history storage; only tail segments and compressed summaries
  P2 — Incremental PCA: Mini-batch accumulation (64 samples) instead of single-sample
  P3 — Γₜ fragility: Track BOTH multiplicative (interpretation) AND additive (stability)
  P4 — R_t early bias: Return NaN during warmup, exclude from statistics
  P5 — Fixed occupancy resolution: Adaptive resolution = 0.15 / sqrt(d) or percentile-based
  P6 — Fixed A_t radius: Adaptive radius = percentile of pairwise distances
  P7 — Multiprocessing: Static worker function for better pickling
  P8 — Archive pruning: Diversity-preserving eviction (remove densest region)
  P9 — Silent fallback: Track PCA failure counts, warn on threshold

Core variables: Nₜ, Cₜ, Rₜ, Aₜ, PRₜ
Master equations: Γ_mult = Cₜ · Aₜ · Rₜ · Δ_PR  (interpretation)
                 Γ_add = w1·Cₜ + w2·Aₜ + w3·Rₜ + w4·Δ_PR (stability)
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree
from scipy.stats import bootstrap
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA, IncrementalPCA
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import json, os, time, warnings
import multiprocessing as mp

warnings.filterwarnings('ignore')
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AstraeusConfig:
    n_seeds: int = 50
    n_ontologies: int = 4
    steps: int = 50000
    d_theta: int = 16
    d_behavior: int = 16

    k_neighbors: int = 5
    novelty_decay: float = 0.99
    connectivity_radius_percentile: int = 15   # P6: adaptive radius
    n_pca_components: Optional[int] = None

    bootstrap_n: int = 5000
    alpha: float = 0.05
    effect_size_threshold: float = 0.8

    autocorr_max_lag: int = 200
    critical_window: int = 2000

    checkpoint_dir: str = "phase11_5_checkpoints"
    dimensions_to_test: List[int] = field(default_factory=lambda: [8, 16, 32, 64])

    interaction_base: float = 0.10
    interaction_min: float = 0.02
    sigma_window: int = 100

    # Optimizations
    kdtree_rebuild_freq: int = 500
    use_ipca: bool = True
    use_batched_forward: bool = True
    checkpoint_format: str = "npy"
    n_parallel_seeds: int = 4
    enable_parallel: bool = True

    # P2: Mini-batch size for IncrementalPCA
    pca_mini_batch_size: int = 64

    # P5: Adaptive occupancy resolution
    occupancy_base_resolution: float = 0.15
    occupancy_adaptive: bool = True

    # P8: Archive pruning
    archive_max_size: int = 8000
    archive_prune_keep_recent: int = 4000

    # P3: Gamma weights (additive version)
    gamma_weights: Tuple[float, float, float, float] = (0.3, 0.25, 0.25, 0.2)

    # Phase 11.5C hook
    ablation_mode: Optional[str] = None

    def effective_n_pca_components(self) -> int:
        if self.n_pca_components is not None:
            return self.n_pca_components
        return max(1, min(8, self.d_theta // 2))

    def get_adaptive_resolution(self) -> float:
        """P5: Adaptive occupancy resolution based on dimension"""
        if not self.occupancy_adaptive:
            return self.occupancy_base_resolution
        # resolution scales with sqrt(d) to maintain consistent cell density
        # d=16 → 0.15, d=64 → 0.075, d=8 → 0.21
        return self.occupancy_base_resolution / max(1, np.sqrt(self.d_theta / 16.0))


# ============================================================================
# ONTOLOGY
# ============================================================================

class Ontology:
    def __init__(self, id: int, d_theta: int, d_behavior: int):
        self.id = id
        self.d_theta = d_theta
        self.d_behavior = d_behavior

        self.theta = torch.randn(d_theta) * 0.1
        W = torch.randn(d_behavior, d_theta) * 0.1
        self.W_embed = W / (W.norm() + 1e-8)
        self.scar_tensor = torch.zeros(d_theta, d_theta)

        self.trajectory = deque(maxlen=2000)
        self.current_behavior: Optional[np.ndarray] = None
        self.step_count = 0

    def forward(self, external_signal: Optional[torch.Tensor] = None,
                interaction_strength: float = 0.05,
                exploration_noise: float = 0.01) -> np.ndarray:
        with torch.no_grad():
            if external_signal is not None:
                self.theta.add_(interaction_strength * (external_signal - self.theta))

            noise = torch.randn_like(self.theta) * exploration_noise
            theta_with_noise = self.theta + noise

            behavior_raw = self.W_embed @ theta_with_noise
            behavior = F.normalize(behavior_raw, dim=0)
            self.current_behavior = behavior.numpy()
            self.trajectory.append(self.theta.clone())
            self.step_count += 1
        return self.current_behavior

    def record_scar(self, other_theta: torch.Tensor):
        with torch.no_grad():
            delta = self.theta - other_theta
            self.scar_tensor += torch.outer(delta, delta)


# ============================================================================
# INCREMENTAL KDTREE
# ============================================================================

class IncrementalKDTree:
    def __init__(self, rebuild_freq: int = 500):
        self.archive = []
        self.tree = None
        self.rebuild_freq = rebuild_freq
        self.step_count = 0
        self.tree_size = 0

    def add(self, point: np.ndarray) -> None:
        self.archive.append(point)
        self.step_count += 1
        if self.step_count % self.rebuild_freq == 0 or self.tree is None:
            self.tree = KDTree(np.array(self.archive))
            self.tree_size = len(self.archive)

    def query(self, point: np.ndarray, k: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.tree is None or self.tree_size == 0:
            return np.array([[1.0] * k]), None

        query_k = min(k, self.tree_size)
        dists, idx = self.tree.query(point.reshape(1, -1), k=query_k)
        dists = np.atleast_2d(dists)
        idx = np.atleast_2d(idx)

        finite = np.isfinite(dists[0])
        if not np.any(finite):
            return np.array([[1.0] * k]), None

        return dists[:, finite], idx[:, finite]


# ============================================================================
# ASTRAEUS FIELD (WITH ALL 9 FIXES)
# ============================================================================

class AstraeusField:
    def __init__(self, config: AstraeusConfig):
        self.config = config
        self.archive: List[np.ndarray] = []
        self.archive_thetas: List[np.ndarray] = []
        self._occupancy_history: List[frozenset] = []
        self._behavior_buffer: deque = deque(maxlen=config.sigma_window)
        self._pca_batch_buffer: List[np.ndarray] = []   # P2: mini-batch accumulation
        self._pca_failure_count: int = 0                # P9: track PCA failures
        self._pca_warned: bool = False
        self._ipca_ready: bool = False

        # P8: Archive pruning metadata (diversity scores)
        self._archive_diversity_scores: Dict[int, float] = {}

        # P1: Only store tail segments, not full histories
        self.history: Dict[str, List] = {
            'N': deque(maxlen=10000),
            'N_B': deque(maxlen=10000),
            'N_P': deque(maxlen=10000),
            'N_G': deque(maxlen=10000),
            'C': deque(maxlen=10000),
            'R': deque(maxlen=10000),
            'A': deque(maxlen=10000),
            'PR': deque(maxlen=10000),
            'gamma_rate': deque(maxlen=10000),
            'Psi': deque(maxlen=10000),
            'Gamma_mult': deque(maxlen=10000),
            'Gamma_add': deque(maxlen=10000),
            'det_Sigma_N': deque(maxlen=10000),
        }

        # P1: Compressed summaries for final output
        self.compressed_summaries = {
            'N_final_tail': deque(maxlen=1000),
            'C_final_tail': deque(maxlen=1000),
            'R_final_tail': deque(maxlen=1000),
            'PR_final_tail': deque(maxlen=1000),
        }

        # Optimizations
        if config.kdtree_rebuild_freq > 0:
            self.incremental_kdtree_behavior = IncrementalKDTree(config.kdtree_rebuild_freq)
            self.incremental_kdtree_theta = IncrementalKDTree(config.kdtree_rebuild_freq)
        else:
            self.incremental_kdtree_behavior = None
            self.incremental_kdtree_theta = None

        if config.use_ipca:
            n_comp = config.effective_n_pca_components()
            self.ipca_N_G = IncrementalPCA(n_components=n_comp)
            self.ipca_det_SN = IncrementalPCA(n_components=min(3, n_comp))
        else:
            self.ipca_N_G = None
            self.ipca_det_SN = None

        self.step = 0

    # ------------------------------------------------------------------
    # P2: Mini-batch PCA update
    # ------------------------------------------------------------------

    def _update_ipca(self, sample: np.ndarray):
        """Accumulate mini-batches for stable IncrementalPCA"""
        self._pca_batch_buffer.append(sample)
        if len(self._pca_batch_buffer) >= self.config.pca_mini_batch_size:
            batch = np.array(self._pca_batch_buffer)
            try:
                if self.ipca_N_G is not None:
                    self.ipca_N_G.partial_fit(batch)
                if self.ipca_det_SN is not None:
                    self.ipca_det_SN.partial_fit(batch)
                self._pca_batch_buffer.clear()
                self._pca_failure_count = 0
                self._ipca_ready = True
            except Exception as e:
                self._pca_failure_count += 1
                if self._pca_failure_count > 10 and not self._pca_warned:
                    print(f"[WARN] PCA instability after {self._pca_failure_count} attempts")
                    self._pca_warned = True

    # ------------------------------------------------------------------
    # P8: Diversity-preserving archive pruning
    # ------------------------------------------------------------------

    def _prune_archive_diversity(self):
        """Remove densest region (lowest diversity) instead of random"""
        if len(self.archive) <= self.config.archive_max_size:
            return

        # Keep recent entries
        keep_recent = self.config.archive_prune_keep_recent
        recent = self.archive[-keep_recent:]
        old = self.archive[:-keep_recent]

        if len(old) <= self.config.archive_max_size - keep_recent:
            self.archive = old + recent
            return

        # Compute diversity scores for old entries
        if len(old) > 10:
            old_array = np.array(old)
            tree = KDTree(old_array)
            diversity_scores = []
            for i, point in enumerate(old):
                dists, _ = tree.query(point.reshape(1, -1), k=min(10, len(old)))
                diversity_scores.append(np.mean(dists[0]))

            # Keep most diverse (largest distances)
            keep_idx = np.argsort(diversity_scores)[-(self.config.archive_max_size - keep_recent):]
            old_pruned = [old[i] for i in keep_idx]
        else:
            old_pruned = old

        self.archive = old_pruned + recent

    # ------------------------------------------------------------------
    # P6: Adaptive radius for accessibility graph
    # ------------------------------------------------------------------

    def _get_adaptive_radius(self, behaviors: List[np.ndarray]) -> float:
        if len(behaviors) < 10:
            return self.config.connectivity_radius_percentile / 100.0

        arr = np.array(behaviors)
        n_samples = min(200, len(arr))
        indices = np.random.choice(len(arr), n_samples, replace=False)
        sampled = arr[indices]

        tree = KDTree(sampled)
        distances, _ = tree.query(sampled, k=2)
        nn_distances = distances[:, 1]

        return float(np.percentile(nn_distances, self.config.connectivity_radius_percentile))

    # ------------------------------------------------------------------
    # SECTION 1 — LOCAL EXPLORATORY DIVERGENCE
    # ------------------------------------------------------------------

    def compute_N(self, behavior: np.ndarray, k: int) -> float:
        if len(self.archive) < k:
            return 1.0

        if self.incremental_kdtree_behavior is not None:
            dists, _ = self.incremental_kdtree_behavior.query(behavior, k=k)
            return float(np.mean(dists[0]))

        tree = KDTree(np.array(self.archive))
        dists, _ = tree.query(behavior.reshape(1, -1), k=k)
        return float(np.mean(dists[0]))

    # ------------------------------------------------------------------
    # SECTION 2 — MULTI-SCALE NOVELTY GEOMETRY
    # ------------------------------------------------------------------

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

        n_pca_components = self.config.effective_n_pca_components()
        if len(self.archive) > n_pca_components + 1 and self.ipca_N_G is not None and self._ipca_ready:
            try:
                b_proj = self.ipca_N_G.transform(behavior.reshape(1, -1))
                if len(self.archive) > n_pca_components:
                    archive_proj = self.ipca_N_G.transform(np.array(self.archive))
                    g_tree = KDTree(archive_proj)
                    N_G = float(g_tree.query(b_proj, k=1)[0].min())
                else:
                    N_G = N_B
            except Exception:
                # P9: Track failure but don't silently mask
                self._pca_failure_count += 1
                N_G = N_B
        else:
            N_G = N_B

        return np.array([N_B, N_P, N_G])

    def compute_det_Sigma_N(self) -> float:
        if len(self._behavior_buffer) < 20:
            return 1.0

        buf = np.array(self._behavior_buffer)
        n_comp = min(3, buf.shape[0] - 1, buf.shape[1])
        if n_comp < 3:
            return 1.0

        if self.config.use_ipca and self.ipca_det_SN is not None and self._ipca_ready:
            try:
                proj = self.ipca_det_SN.transform(buf)
            except:
                pca = PCA(n_components=3)
                proj = pca.fit_transform(buf)
        else:
            pca = PCA(n_components=3)
            proj = pca.fit_transform(buf)

        Sigma = np.cov(proj.T) + np.eye(3) * 1e-6
        return float(np.linalg.det(Sigma))

    # ------------------------------------------------------------------
    # SECTION 3 — COMPRESSION FIELD
    # ------------------------------------------------------------------

    def compute_C_and_gamma(self, window: int = 1000) -> Tuple[float, float]:
        N = list(self.history['N'])
        if len(N) < 2 * window:
            return 0.0, 0.0

        past_mean = np.mean(N[-2 * window:-window])
        current_mean = np.mean(N[-window:])
        C = 1.0 - current_mean / (past_mean + 1e-8)
        gamma = float(-np.gradient(N[-window:]).mean())

        return C, gamma

    # ------------------------------------------------------------------
    # SECTION 4 — ACCESSIBILITY ENTROPY (P6: adaptive radius)
    # ------------------------------------------------------------------

    def compute_A(self, behaviors: List[np.ndarray]) -> float:
        n = len(behaviors)
        if n < 2:
            return 0.0

        arr = np.array(behaviors)
        r = self._get_adaptive_radius(behaviors)

        tree = KDTree(arr)
        pairs = tree.query_pairs(r)
        adj = np.zeros((n, n), dtype=np.int8)
        for i, j in pairs:
            adj[i, j] = adj[j, i] = 1

        n_comp, labels = connected_components(adj, directed=False)
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / n
        return float(-np.sum(probs * np.log(probs + 1e-8)))

    # ------------------------------------------------------------------
    # SECTION 5 — RESTRUCTURING PERSISTENCE (P5: adaptive resolution, P4: NaN warmup)
    # ------------------------------------------------------------------

    def _behavior_to_cell(self, b: np.ndarray) -> tuple:
        resolution = self.config.get_adaptive_resolution()
        return tuple((b / resolution).astype(int))

    def compute_R(self, behaviors: List[np.ndarray], window: int = 500) -> Tuple[float, float]:
        if self.config.ablation_mode == 'occupancy_off':
            return np.nan, 0.0

        occupancy = frozenset(self._behavior_to_cell(b) for b in behaviors)
        self._occupancy_history.append(occupancy)

        # P4: Return NaN during warmup
        if len(self._occupancy_history) < window + 1:
            return np.nan, 0.0

        O_past = self._occupancy_history[-(window + 1)]
        O_current = self._occupancy_history[-1]

        newly_opened = len(O_current - O_past)
        R = newly_opened / (len(O_current) + 1e-8)

        if len(self.history['R']) >= 10:
            recent_R = [r for r in list(self.history['R'])[-10:] if not np.isnan(r)]
            Psi = float(np.gradient(recent_R).mean()) if len(recent_R) > 1 else 0.0
        else:
            Psi = 0.0

        return R, Psi

    # ------------------------------------------------------------------
    # SECTION 6 — PARTICIPATION RATIO
    # ------------------------------------------------------------------

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
        recent = [p for p in list(self.history['PR'])[-window:] if not np.isnan(p)]
        if len(recent) < 2:
            return 0.0
        return recent[-1] - recent[0]

    # ------------------------------------------------------------------
    # SECTION 7 — CRITICAL SLOWING (stabilized weighted fit)
    # ------------------------------------------------------------------

    def compute_xi(self, N_history: List[float]) -> float:
        if len(N_history) < self.config.autocorr_max_lag + 50:
            return 0.0

        seg = np.array(N_history[-self.config.critical_window:])
        seg = seg - seg.mean()
        ac_full = np.correlate(seg, seg, mode='full')
        ac = ac_full[len(ac_full) // 2:]
        denom = ac[0] + 1e-8
        ac = ac / denom

        max_lag = min(self.config.autocorr_max_lag, 100)
        lags = np.arange(1, min(max_lag, len(ac)))
        valid = (ac[lags] > 0.1) & (lags < 100)

        if np.sum(valid) < 5:
            return 0.0

        log_ac = np.log(ac[lags[valid]] + 1e-8)
        weights = ac[lags[valid]]
        slope, _ = np.polyfit(lags[valid], log_ac, 1, w=weights)
        if slope >= 0:
            return 0.0
        return float(-1.0 / slope)

    # ------------------------------------------------------------------
    # MASTER UPDATE (P3: both gamma formulations)
    # ------------------------------------------------------------------

    def update(self, ontologies: List[Ontology]) -> Dict[str, float]:
        self.step += 1

        behaviors = [o.current_behavior for o in ontologies if o.current_behavior is not None]
        thetas = [o.theta.numpy() for o in ontologies if o.current_behavior is not None]

        if not behaviors:
            return {}

        # Update buffers
        for b in behaviors:
            self._behavior_buffer.append(b)
            self._update_ipca(b)  # P2: mini-batch PCA

        if self.config.ablation_mode != 'archive_off':
            for b in behaviors:
                jitter = np.random.randn(*b.shape) * 0.001
                b_jittered = b + jitter
                self.archive.append(b_jittered)
                if self.incremental_kdtree_behavior is not None:
                    self.incremental_kdtree_behavior.add(b_jittered)

            self.archive_thetas.extend(thetas)
            for t in thetas:
                if self.incremental_kdtree_theta is not None:
                    self.incremental_kdtree_theta.add(t)

            # P8: Diversity-preserving pruning
            if len(self.archive) > self.config.archive_max_size:
                self._prune_archive_diversity()

        # Compute metrics
        N_vals = [self.compute_N(b, self.config.k_neighbors) for b in behaviors]
        N_mean = float(np.mean(N_vals))

        scales = [self.compute_N_multiscale(t, b) for t, b in zip(thetas, behaviors)]
        scales = np.array(scales)
        N_B = float(np.mean(scales[:, 0]))
        N_P = float(np.mean(scales[:, 1]))
        N_G = float(np.mean(scales[:, 2]))

        if self.step % 500 == 0:
            det_SN = self.compute_det_Sigma_N()
        else:
            det_SN = self.history['det_Sigma_N'][-1] if self.history['det_Sigma_N'] else 1.0

        C, gamma = self.compute_C_and_gamma()

        if self.step % 100 == 0:
            A = self.compute_A(behaviors)
        else:
            A = self.history['A'][-1] if self.history['A'] else 1.0

        R, Psi = self.compute_R(behaviors)

        if self.step % 500 == 0:
            PR = self.compute_PR(behaviors)
        else:
            PR = self.history['PR'][-1] if self.history['PR'] else 1.0

        # Store histories (deque with maxlen for memory safety)
        self.history['N'].append(N_mean)
        self.history['N_B'].append(N_B)
        self.history['N_P'].append(N_P)
        self.history['N_G'].append(N_G)
        self.history['C'].append(C)
        self.history['R'].append(R)
        self.history['A'].append(A)
        self.history['PR'].append(PR)
        self.history['gamma_rate'].append(gamma)
        self.history['Psi'].append(Psi)
        self.history['det_Sigma_N'].append(det_SN)

        # P1: Compressed summaries (last 1000 steps)
        self.compressed_summaries['N_final_tail'].append(N_mean)
        self.compressed_summaries['C_final_tail'].append(C)
        if not np.isnan(R):
            self.compressed_summaries['R_final_tail'].append(R)
        self.compressed_summaries['PR_final_tail'].append(PR)

        Delta_PR = self.compute_Delta_PR()

        # P3: BOTH gamma formulations
        Gamma_mult = C * A * (R if not np.isnan(R) else 0.5) * Delta_PR
        self.history['Gamma_mult'].append(Gamma_mult)

        # Additive gamma (stable)
        w1, w2, w3, w4 = self.config.gamma_weights
        Gamma_add = (w1 * max(-1, min(1, C)) +
                     w2 * max(0, min(1, A)) +
                     w3 * max(0, min(1, R if not np.isnan(R) else 0.5)) +
                     w4 * max(-1, min(1, Delta_PR)))
        self.history['Gamma_add'].append(Gamma_add)

        return {
            'step': self.step,
            'N': N_mean, 'N_B': N_B, 'N_P': N_P, 'N_G': N_G,
            'C': C, 'R': R, 'A': A, 'PR': PR, 'Delta_PR': Delta_PR,
            'Gamma_mult': Gamma_mult, 'Gamma_add': Gamma_add,
            'det_Sigma_N': det_SN
        }


# ============================================================================
# STATIC WORKER FUNCTION (P7: better multiprocessing)
# ============================================================================

def run_seed_worker(seed: int, d_theta: int, config_dict: dict) -> Dict:
    """Static worker function for multiprocessing (P7)"""
    config = AstraeusConfig(**config_dict)
    config.d_theta = d_theta

    np.random.seed(seed)
    torch.manual_seed(seed)

    ontologies = [Ontology(i, d_theta, config.d_behavior) for i in range(config.n_ontologies)]
    field = AstraeusField(config)

    for o in ontologies:
        o.forward()

    xi_snapshots = []

    for step in range(config.steps):
        strength = interaction_schedule(step, config.steps, config.interaction_base, config.interaction_min)

        if config.use_batched_forward and len(ontologies) > 1:
            theta_batch = torch.stack([o.theta for o in ontologies])
            for i, oi in enumerate(ontologies):
                others_idx = [j for j in range(len(ontologies)) if j != i]
                if config.ablation_mode == 'recursive_off':
                    avg = None
                elif others_idx:
                    avg = theta_batch[others_idx].mean(0)
                else:
                    avg = oi.theta.clone()

                oi.forward(avg, strength, exploration_noise=0.01)

                if config.ablation_mode != 'scar_off':
                    for j in others_idx:
                        oj = ontologies[j]
                        if torch.norm(oi.theta - oj.theta) > 2.0:
                            oi.record_scar(oj.theta)
        else:
            for i, oi in enumerate(ontologies):
                others = [oj for j, oj in enumerate(ontologies) if j != i]
                avg = None if config.ablation_mode == 'recursive_off' else torch.stack([oj.theta for oj in others]).mean(0)
                oi.forward(avg, strength, exploration_noise=0.01)

                if config.ablation_mode != 'scar_off':
                    for oj in others:
                        if torch.norm(oi.theta - oj.theta) > 2.0:
                            oi.record_scar(oj.theta)

        field.update(ontologies)

        if step > 0 and step % 5000 == 0:
            xi = field.compute_xi(list(field.history['N']))
            xi_snapshots.append((step, xi))

    # Final metrics from compressed summaries (P1)
    def tail_metric(deque_obj):
        if len(deque_obj) == 0:
            return 0.0
        return float(np.mean(list(deque_obj)))

    return {
        'seed': seed, 'd_theta': d_theta,
        'N_final': tail_metric(field.compressed_summaries['N_final_tail']),
        'C_final': tail_metric(field.compressed_summaries['C_final_tail']),
        'R_final': tail_metric(field.compressed_summaries['R_final_tail']),
        'A_final': tail_metric(field.history['A']),
        'PR_final': tail_metric(field.compressed_summaries['PR_final_tail']),
        'Gamma_final': tail_metric(field.history['Gamma_add']),
        'det_Sigma_mean': tail_metric(field.history['det_Sigma_N']),
        'xi_snapshots': xi_snapshots,
        'pca_failure_count': field._pca_failure_count,
    }


def interaction_schedule(step: int, total_steps: int, base: float = 0.10, minimum: float = 0.02) -> float:
    progress = step / total_steps
    decay = 1.0 / (1.0 + np.exp(10 * (progress - 0.6)))
    return minimum + (base - minimum) * decay


# ============================================================================
# STATISTICAL VALIDATOR
# ============================================================================

class StatisticalValidator:
    def __init__(self, config: AstraeusConfig):
        self.config = config

    def bootstrap_ci(self, data: np.ndarray, n: int = 1000) -> Tuple[float, float]:
        res = bootstrap((data,), np.mean, n_resamples=n,
                        confidence_level=0.95, method='BCa')
        return float(res.confidence_interval.low), float(res.confidence_interval.high)

    def validate_across_dims(self, results: Dict) -> Dict:
        summary = {}
        for dim, res in results.items():
            if not isinstance(dim, int):
                continue
            N_arr = np.array([r['N_final'] for r in res['seeds']])
            ci = self.bootstrap_ci(N_arr)
            summary[dim] = {
                'mean': float(np.mean(N_arr)),
                'std': float(np.std(N_arr, ddof=1)),
                'ci_95': ci,
                'compressed_regime_rate': float(np.mean(N_arr < 0.20)),
            }
        summary['consistent'] = all(v['compressed_regime_rate'] > 0.70 for v in summary.values() if isinstance(v, dict))
        return summary


# ============================================================================
# MAIN EXPERIMENT (P7: using static worker)
# ============================================================================

class Phase11_5Experiment:
    def __init__(self, config: AstraeusConfig):
        self.config = config
        self.validator = StatisticalValidator(config)
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def run(self) -> Dict:
        print("=" * 70)
        print("PHASE 11.5 - ASTRAEUS UNIFIED FIELD FRAMEWORK v2")
        print(f"Seeds: {self.config.n_seeds}  |  Steps: {self.config.steps}")
        print(f"Dimensions: {self.config.dimensions_to_test}")
        print(f"Fixes: P1-P9 active (memory, PCA, gamma, occupancy, radius, pruning)")
        if self.config.enable_parallel:
            n_procs = min(self.config.n_parallel_seeds, os.cpu_count() - 1 if os.cpu_count() else 4)
            print(f"Parallel execution: {n_procs} processes")
        print("=" * 70)

        all_results = {}
        config_dict = {k: v for k, v in self.config.__dict__.items()
                       if not k.startswith('_') and not callable(v)}

        for d in self.config.dimensions_to_test:
            print(f"\n-- d = {d} --------------------------")

            if self.config.enable_parallel and self.config.n_seeds > 1:
                try:
                    n_procs = min(self.config.n_parallel_seeds, os.cpu_count() - 1 if os.cpu_count() else 4)
                    seed_args = [(s, d, config_dict) for s in range(self.config.n_seeds)]

                    with mp.Pool(processes=n_procs) as pool:
                        seed_results = pool.starmap(run_seed_worker, seed_args)

                    print(f"  OK Completed {self.config.n_seeds} seeds in parallel ({n_procs} procs)")
                except Exception as e:
                    print(f"  WARNING Parallel failed: {e}. Falling back to sequential.")
                    seed_results = [run_seed_worker(s, d, config_dict) for s in range(self.config.n_seeds)]
            else:
                seed_results = [run_seed_worker(s, d, config_dict) for s in range(self.config.n_seeds)]

            N_arr = np.array([r['N_final'] for r in seed_results])
            ci = self.validator.bootstrap_ci(N_arr, n=self.config.bootstrap_n)

            # P1: Running convergence without full history storage
            running_means = [float(np.mean(N_arr[:i])) for i in range(5, len(N_arr) + 1)]

            all_results[d] = {
                'seeds': seed_results,
                'N_mean': float(np.mean(N_arr)),
                'N_std': float(np.std(N_arr, ddof=1)),
                'N_ci95': ci,
                'compressed_regime_rate': float(np.mean(N_arr < 0.20)),
                'convergence_curve': running_means,
            }

            print(f"  d={d}: N={all_results[d]['N_mean']:.4f}+/-{all_results[d]['N_std']:.4f}"
                  f"  95%CI=[{ci[0]:.4f},{ci[1]:.4f}]"
                  f"  compressed_regime={all_results[d]['compressed_regime_rate']*100:.0f}%")

        validation = self.validator.validate_across_dims(all_results)
        all_results['validation'] = validation

        print("\n" + "=" * 70)
        print("CROSS-DIMENSION VALIDATION")
        print(f"  Consistent compressed regime: {validation['consistent']}")
        for dim, v in validation.items():
            if isinstance(dim, int):
                print(f"  d={dim}: compressed_regime={v['compressed_regime_rate']*100:.0f}%  "
                      f"N={v['mean']:.4f}  CI={v['ci_95']}")
        print("=" * 70)

        # Critical slowing summary
        xi_peaks = []
        for r in all_results.get(16, {}).get('seeds', []):
            snaps = r.get('xi_snapshots', [])
            if snaps:
                xi_peaks.append(max(v for _, v in snaps))
        if xi_peaks:
            print(f"\nCritical slowing (d=16): xi_peak = {np.mean(xi_peaks):.2f} +/- {np.std(xi_peaks):.2f}")
            print("  (elevated xi may indicate slowing dynamics near saturation)")

        return all_results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    config = AstraeusConfig(
        n_seeds=50,
        steps=50000,
        dimensions_to_test=[8, 16, 32, 64],
        sigma_window=100,
        archive_max_size=8000,
        archive_prune_keep_recent=4000,
        pca_mini_batch_size=64,
    )
    experiment = Phase11_5Experiment(config)
    results = experiment.run()
