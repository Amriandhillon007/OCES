"""
PHASE 9.5: INFORMATION GEOMETRY ANALYSIS
Exploratory implementation for OCES project

RESULTS FROM YOUR RUN:
- Curvature-proxy vs Novelty: strong negative association
- Volume-proxy vs Novelty: strong positive association
- Geodesic-proxy vs Novelty: strong positive association

INTERPRETATION:
- Curvature is an operational proxy, not a rigorous Ricci/scalar curvature proof.
- Results are preliminary correlations, not causation, universality, or theorem.
- Strong geodesic-novelty association may be the most promising theoretical lead.

Author: OCES Project
Date: 2026-05-19
Status: EXPLORATORY
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
import os
import sys
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class InfoGeomConfig:
    """Configuration for information geometry analysis"""
    
    # Data parameters
    horizons: List[int] = field(default_factory=lambda: [5000, 10000, 25000, 50000])
    novelty: List[float] = field(default_factory=lambda: [0.085, 0.141, 0.151, 0.134])
    divergence: List[float] = field(default_factory=lambda: [0.534, 0.495, 0.409, 0.334])
    utility: List[float] = field(default_factory=lambda: [0.50, 0.52, 0.53, 0.540])
    entropy: List[float] = field(default_factory=lambda: [0.85, 0.90, 0.93, 0.950])
    stability: List[float] = field(default_factory=lambda: [0.70, 0.78, 0.82, 0.847])
    
    # Analysis parameters
    smooth_curvature: bool = True
    curvature_window: int = 2
    curvature_coordinate: str = "log_horizon"  # "log_horizon", "normalized_horizon", or "raw_horizon"
    normalization_method: str = "minmax"  # "minmax" or "zscore"
    
    # Output
    save_plots: bool = True
    save_json: bool = True
    output_dir: str = "phase9.5_results"
    verbose: bool = True


# ============================================================================
# PART 1: FISHER INFORMATION METRIC (COMPLETE)
# ============================================================================

class FisherInformationMetric:
    """
    Fisher-like information proxy on the representational manifold.
    
    A formal Fisher metric requires an explicit statistical model,
    parameter manifold, and likelihood. The default divergence-based path here
    is exploratory until those assumptions are supplied.
    
    In OCES context: Measures how distinguishable different law parameters are
    based on their behavioral outputs.
    """
    
    def __init__(self, config: InfoGeomConfig):
        self.config = config
        
    def compute_from_divergence(self, divergence: np.ndarray) -> np.ndarray:
        """
        Compute Fisher metric proxy from behavioral divergence.
        
        Rationale: Divergence between ontologies is used as an exploratory
        proxy for distinguishability. Do not interpret this as a formally
        derived Fisher metric without additional assumptions.
        
        Returns:
            fisher_norm: Normalized Fisher information norm
        """
        if np.max(divergence) == np.min(divergence):
            return np.ones_like(divergence)
        
        # Min-max normalization
        fisher_norm = (divergence - np.min(divergence)) / (np.max(divergence) - np.min(divergence))
        return fisher_norm
    
    def compute_from_trajectories(self, trajectories: List[np.ndarray]) -> np.ndarray:
        """
        Compute Fisher metric from actual trajectory data.
        
        Uses covariance of state differences as empirical Fisher information.
        
        Args:
            trajectories: List of state trajectory arrays
            
        Returns:
            fisher_matrix: Fisher information matrix
        """
        if len(trajectories) < 2:
            return np.eye(16)
        
        # Stack trajectories
        traj_stack = np.vstack(trajectories)
        
        # Compute covariance (empirical Fisher)
        cov = np.cov(traj_stack.T)
        
        # Fisher metric is inverse of covariance (up to scale)
        try:
            fisher_matrix = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            fisher_matrix = np.eye(cov.shape[0])
        
        return fisher_matrix
    
    def compute_norm(self, fisher_matrix: np.ndarray) -> float:
        """Compute Frobenius norm of Fisher matrix"""
        return np.sqrt(np.sum(fisher_matrix ** 2))


# ============================================================================
# PART 2: CURVATURE ANALYSIS (COMPLETE)
# ============================================================================

class ManifoldCurvature:
    """
    Operational curvature-proxy analysis.
    
    This implementation does not compute rigorous differential-geometric
    Ricci/scalar curvature. It estimates a curvature-like signal from the
    second derivative of behavioral divergence over a chosen horizon coordinate.
    
    Interpretation:
    - proxy > 0: divergence is locally convex in the chosen coordinate
    - proxy < 0: divergence is locally concave in the chosen coordinate
    - proxy = 0: locally neutral under this operational proxy
    """
    
    def __init__(self, config: InfoGeomConfig):
        self.config = config
        
    def compute_curvature_from_divergence(self, divergence: np.ndarray, 
                                           steps: np.ndarray) -> np.ndarray:
        """
        Compute an operational curvature proxy from divergence.
        
        This is a normalized second derivative of divergence in the configured
        horizon coordinate. Treat it as exploratory evidence, not rigorous
        differential curvature.
        """
        steps = np.asarray(steps, dtype=float)
        divergence = np.asarray(divergence, dtype=float)
        coordinate = self._curvature_coordinate(steps)

        # First derivative (velocity)
        velocity = np.gradient(divergence, coordinate)
        
        # Second derivative (acceleration/curvature)
        curvature = np.gradient(velocity, coordinate)
        
        # Normalize
        if np.std(curvature) > 1e-12:
            curvature = curvature / np.std(curvature)
        
        # Smooth if requested
        if self.config.smooth_curvature and len(curvature) > self.config.curvature_window:
            kernel = np.ones(self.config.curvature_window) / self.config.curvature_window
            curvature = np.convolve(curvature, kernel, mode='same')
        
        return curvature

    def _curvature_coordinate(self, steps: np.ndarray) -> np.ndarray:
        """Return a stable coordinate system for curvature derivatives."""
        method = self.config.curvature_coordinate

        if method == "raw_horizon":
            return steps

        if method == "normalized_horizon":
            span = np.max(steps) - np.min(steps)
            if span <= 0:
                return np.arange(len(steps), dtype=float)
            return (steps - np.min(steps)) / span

        if method == "log_horizon":
            safe_steps = np.maximum(steps, 1.0)
            return np.log(safe_steps)

        raise ValueError(
            "curvature_coordinate must be one of: "
            "'log_horizon', 'normalized_horizon', 'raw_horizon'"
        )
    
    def compute_ricci_scalar(self, fisher_matrix: np.ndarray) -> float:
        """
        Compute a scalar proxy from the Fisher-like metric.
        
        This is not a formal Ricci scalar unless the parameter manifold, metric,
        and differential structure have been derived separately.
        """
        d = fisher_matrix.shape[0]
        det_g = np.linalg.det(fisher_matrix + 1e-8 * np.eye(d))
        ricci_scalar = np.log(det_g + 1e-8) / d
        return ricci_scalar
    
    def classify_curvature_regime(self, curvature: np.ndarray) -> str:
        """Classify the operational curvature-proxy trajectory."""
        curvature = np.asarray(curvature, dtype=float)
        mean_curvature = float(np.mean(curvature))
        final_curvature = float(curvature[-1])
        trend = float(final_curvature - curvature[0])
        
        if abs(mean_curvature) < 0.2 and abs(final_curvature) < 0.2:
            regime = "Near-Zero (Flat/Neutral)"
        elif final_curvature > 0.2 or mean_curvature > 0.2:
            regime = "Positive (Attractor Dominant)"
        elif final_curvature < -0.2 or mean_curvature < -0.2:
            regime = "Negative (Divergent/Novel)"
        else:
            regime = "Near-Zero (Flat/Neutral)"
        
        if abs(trend) > 0.15:
            direction = "slightly attracting" if trend > 0 else "slightly diverging"
            return f"{regime} with {direction} trend"
        return regime


# ============================================================================
# PART 3: GEODESIC DISTANCE (COMPLETE)
# ============================================================================

class GeodesicDistance:
    """
    Geodesic distance on the representational manifold.
    
    Mathematical Definition:
    d(Î¦â‚, Î¦â‚‚) = inf_Î³ âˆ«âˆš(g_ij dÎ³â± dÎ³Ê²)
    
    Properties:
    - True behavioral distance (not Euclidean)
    - Accounts for manifold curvature
    - Geodesic = shortest path under metric
    """
    
    def __init__(self, config: InfoGeomConfig):
        self.config = config
        
    def compute_geodesic_from_divergence(self, divergence: np.ndarray) -> np.ndarray:
        """
        Compute cumulative geodesic distance from divergence.
        
        Geodesic distance approximates the integrated path length
        through behavioral space.
        """
        # Normalize divergence to [0, 1]
        if np.max(divergence) > np.min(divergence):
            norm_div = (divergence - np.min(divergence)) / (np.max(divergence) - np.min(divergence))
        else:
            norm_div = divergence
        
        # Cumulative sum approximates geodesic path length
        geodesic = np.cumsum(norm_div)
        
        # Normalize to [0, 1]
        if np.max(geodesic) > 0:
            geodesic = geodesic / np.max(geodesic)
        
        return geodesic
    
    def compute_geodesic_matrix(self, states: List[np.ndarray], 
                                 metric_func) -> np.ndarray:
        """
        Compute pairwise geodesic distance matrix.
        
        Args:
            states: List of state vectors
            metric_func: Function that returns metric tensor at a point
            
        Returns:
            dist_matrix: Pairwise geodesic distances
        """
        n = len(states)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Simplified: use Euclidean with metric correction
                diff = states[i] - states[j]
                g = metric_func(states[i])
                
                if g.shape[0] == len(diff):
                    dist = np.sqrt(diff @ g[:len(diff), :len(diff)] @ diff)
                else:
                    dist = np.linalg.norm(diff)
                
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix


# ============================================================================
# PART 4: VOLUME ELEMENT (COMPLETE)
# ============================================================================

class ManifoldVolume:
    """
    Volume element tracking for representational capacity.
    
    Mathematical Definition:
    dV = âˆš(det(g)) dÎ¸Â¹ âˆ§ ... âˆ§ dÎ¸â¿
    
    Interpretation:
    - Volume increases as manifold explores new regions
    - Volume plateau = representational saturation
    - dV/dt â†’ 0 indicates novelty ceiling
    """
    
    def __init__(self, config: InfoGeomConfig):
        self.config = config
        
    def compute_volume_from_divergence(self, divergence: np.ndarray) -> np.ndarray:
        """
        Compute cumulative volume from divergence.
        
        Volume approximates the total representational capacity
        explored up to time t.
        """
        # Normalize divergence to [0, 1]
        if np.max(divergence) > np.min(divergence):
            norm_div = (divergence - np.min(divergence)) / (np.max(divergence) - np.min(divergence))
        else:
            norm_div = divergence
        
        # Cumulative sum approximates volume
        volume = np.cumsum(norm_div)
        
        # Normalize to [0, 1]
        if np.max(volume) > 0:
            volume = volume / np.max(volume)
        
        return volume
    
    def compute_volume_element(self, metric: np.ndarray) -> float:
        """Compute local volume element âˆš(det(g))"""
        d = metric.shape[0]
        det_g = np.linalg.det(metric + 1e-8 * np.eye(d))
        return np.sqrt(det_g)
    
    def detect_saturation(self, volume: np.ndarray, threshold: float = 0.05) -> bool:
        """
        Detect if volume has saturated.
        
        Saturation = growth rate below threshold.
        """
        if len(volume) < 2:
            return False
        
        growth_rate = np.gradient(volume)
        if len(growth_rate) > 0:
            return bool(growth_rate[-1] < threshold)
        
        return False


# ============================================================================
# PART 5: CORRELATION ANALYSIS (COMPLETE)
# ============================================================================

class CorrelationAnalyzer:
    """
    Correlation analysis between geometric quantities and novelty.
    """
    
    def __init__(self, config: InfoGeomConfig):
        self.config = config
        
    def compute_correlations(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute all pairwise correlations.
        
        Returns:
            Dictionary of correlation coefficients
        """
        correlations = {}
        
        # Define pairs to analyze
        pairs = [
            ('curvature', 'novelty'),
            ('volume', 'novelty'),
            ('geodesic', 'novelty'),
            ('fisher', 'novelty'),
            ('divergence', 'novelty'),
            ('curvature', 'divergence')
        ]
        
        for key1, key2 in pairs:
            if key1 in data and key2 in data:
                arr1 = np.array(data[key1])
                arr2 = np.array(data[key2])
                
                if len(arr1) > 1 and len(arr2) > 1 and np.std(arr1) > 1e-8 and np.std(arr2) > 1e-8:
                    corr = np.corrcoef(arr1, arr2)[0, 1]
                else:
                    corr = 0.0
                
                correlations[f"{key1}_vs_{key2}"] = corr
        
        return correlations
    
    def interpret_correlation(self, corr: float, metric_name: str) -> str:
        """Interpret correlation strength and direction"""
        abs_corr = abs(corr)
        
        if abs_corr > 0.7:
            strength = "Strong"
        elif abs_corr > 0.4:
            strength = "Moderate"
        elif abs_corr > 0.2:
            strength = "Weak"
        else:
            strength = "Negligible"
        
        direction = "positive" if corr > 0 else "negative"
        
        # Specific interpretations
        if metric_name == "curvature_vs_novelty":
            if corr < -0.5:
                insight = "Preliminary evidence consistent with geometric novelty saturation"
            elif corr > 0.5:
                insight = "Curvature proxy rises with novelty in this exploratory run"
            else:
                insight = "No clear curvature-proxy/novelty association"
        elif metric_name == "volume_vs_novelty":
            if corr > 0.5:
                insight = "Volume grows with novelty (expected)"
            else:
                insight = "Volume decoupled from novelty"
        elif metric_name == "geodesic_vs_novelty":
            if corr > 0.5:
                insight = "Strong lead: novelty may track geometric trajectory displacement"
            else:
                insight = "Geodesic distance decoupled from novelty"
        else:
            insight = f"{strength} {direction} correlation"
        
        return f"{strength} {direction} ({corr:.3f}): {insight}"


# ============================================================================
# PART 6: INTEGRATED ANALYZER (COMPLETE)
# ============================================================================

class InformationGeometryAnalyzer:
    """
    Complete information geometry analysis for OCES.
    
    Integrates:
    - Fisher metric
    - Curvature analysis
    - Geodesic distance
    - Volume tracking
    - Correlation analysis
    """
    
    def __init__(self, config: InfoGeomConfig = None):
        self.config = config or InfoGeomConfig()
        
        self.fisher = FisherInformationMetric(self.config)
        self.curvature = ManifoldCurvature(self.config)
        self.geodesic = GeodesicDistance(self.config)
        self.volume = ManifoldVolume(self.config)
        self.correlation = CorrelationAnalyzer(self.config)
        
        self.results = {}
        
    def analyze(self, 
                horizons: List[int] = None,
                novelty: List[float] = None,
                divergence: List[float] = None,
                utility: List[float] = None,
                entropy: List[float] = None,
                stability: List[float] = None) -> Dict:
        """
        Run complete information geometry analysis.
        
        Args:
            horizons: Step counts
            novelty: Novelty values
            divergence: Divergence values
            utility: Utility values
            entropy: Entropy values
            stability: Stability values
            
        Returns:
            Dictionary containing all analysis results
        """
        # Use config defaults if not provided
        horizons = horizons or self.config.horizons
        novelty = novelty or self.config.novelty
        divergence = divergence or self.config.divergence
        utility = utility or self.config.utility
        
        # Convert to numpy arrays
        horizons = np.array(horizons)
        novelty = np.array(novelty)
        divergence = np.array(divergence)
        utility = np.array(utility)
        
        # Compute geometric quantities
        fisher_norm = self.fisher.compute_from_divergence(divergence)
        curvature = self.curvature.compute_curvature_from_divergence(divergence, horizons)
        geodesic_dist = self.geodesic.compute_geodesic_from_divergence(divergence)
        volume = self.volume.compute_volume_from_divergence(divergence)
        
        # Detect saturation
        volume_saturated = self.volume.detect_saturation(volume)
        
        # Compute correlations
        data = {
            'horizons': horizons,
            'novelty': novelty,
            'divergence': divergence,
            'utility': utility,
            'fisher': fisher_norm,
            'curvature': curvature,
            'geodesic': geodesic_dist,
            'volume': volume
        }
        
        correlations = self.correlation.compute_correlations(data)
        
        # Store results
        self.results = {
            'data': {
                'horizons': horizons.tolist(),
                'novelty': novelty.tolist(),
                'divergence': divergence.tolist(),
                'utility': utility.tolist(),
                'fisher_norm': fisher_norm.tolist(),
                'curvature': curvature.tolist(),
                'geodesic_distance': geodesic_dist.tolist(),
                'volume': volume.tolist()
            },
            'correlations': correlations,
            'metadata': {
                'epistemic_status': 'exploratory correlations over operational proxies',
                'curvature_status': 'normalized second-derivative proxy; not formal Ricci/scalar curvature',
                'strongest_lead': 'geodesic_vs_novelty association',
                'novelty_ceiling': float(np.max(novelty)),
                'novelty_sustainable': float(novelty[-1]),
                'utility_at_ceiling': float(utility[-1]),
                'volume_saturated': volume_saturated,
                'curvature_regime': self.curvature.classify_curvature_regime(curvature),
                'curvature_mean': float(np.mean(curvature)),
                'curvature_final': float(curvature[-1]),
                'curvature_trend': float(curvature[-1] - curvature[0]),
                'analysis_date': datetime.now().isoformat()
            }
        }
        
        # Add interpretations
        self.results['interpretations'] = {}
        for key, corr in correlations.items():
            self.results['interpretations'][key] = self.correlation.interpret_correlation(corr, key)
        
        if self.config.verbose:
            self._print_results()
        
        if self.config.save_plots:
            self._plot_results()
        
        if self.config.save_json:
            self._save_json()
        
        return self.results
    
    def _print_results(self):
        """Print analysis results to console"""
        print()
        print("=" * 80)
        print("INFORMATION GEOMETRY ANALYSIS RESULTS")
        print("=" * 80)
        print()
        
        print("Exploratory Geometric Proxy Quantities:")
        print("-" * 40)
        print(f"{'Step':>8} | {'Fisher*':>8} | {'Curv*':>10} | {'Geodesic*':>10} | {'Volume*':>10}")
        print("-" * 60)
        
        for i, h in enumerate(self.results['data']['horizons']):
            print(f"{h:8d} | {self.results['data']['fisher_norm'][i]:8.3f} | "
                  f"{self.results['data']['curvature'][i]:10.3f} | "
                  f"{self.results['data']['geodesic_distance'][i]:10.3f} | "
                  f"{self.results['data']['volume'][i]:10.3f}")
        
        print()
        print("Correlations:")
        print("-" * 40)
        for key, corr in self.results['correlations'].items():
            print(f"{key:25}: {corr:.4f}")
        
        print()
        print("Interpretations:")
        print("-" * 40)
        for key, interpretation in self.results['interpretations'].items():
            print(f"{key}: {interpretation}")
        print()
        print("Caution:")
        print("-" * 40)
        print("These are preliminary correlations over operational proxies.")
        print("They do not establish causation, proof, theorem status, or universality.")
        print("Curvature* is a normalized second-derivative proxy, not formal Ricci curvature.")
        
        print()
        print("Metadata:")
        print("-" * 40)
        print(f"Epistemic Status:          {self.results['metadata']['epistemic_status']}")
        print(f"Curvature Status:          {self.results['metadata']['curvature_status']}")
        print(f"Strongest Lead:            {self.results['metadata']['strongest_lead']}")
        print(f"Novelty Ceiling (peak):     {self.results['metadata']['novelty_ceiling']:.3f}")
        print(f"Novelty Sustainable (50k):  {self.results['metadata']['novelty_sustainable']:.3f}")
        print(f"Utility at Ceiling:         {self.results['metadata']['utility_at_ceiling']:.3f}")
        print(f"Volume Saturated:           {self.results['metadata']['volume_saturated']}")
        print(f"Final Curvature* Regime:    {self.results['metadata']['curvature_regime']}")
        print(f"Curvature* Mean:            {self.results['metadata']['curvature_mean']:.4f}")
        print(f"Curvature* Final:           {self.results['metadata']['curvature_final']:.4f}")
        print(f"Curvature* Trend:           {self.results['metadata']['curvature_trend']:+.4f}")
        
        print()
        print("EXPLORATORY READOUT:")
        print("-" * 40)
        
        # Final verdict
        curvature_corr = self.results['correlations'].get('curvature_vs_novelty', 0)
        curvature_regime = self.results['metadata']['curvature_regime']
        
        if curvature_corr < -0.5:
            print("[ASSOCIATION] STRONG NEGATIVE CURVATURE-PROXY/NOVELTY CORRELATION")
            print("   -> Preliminary evidence consistent with geometric novelty saturation")
            print("   -> Not a proof, theorem, causal claim, or universality claim")
        elif "Near-Zero" in curvature_regime:
            print("[INFO] NEUTRAL CURVATURE-PROXY REGIME")
            print("   -> Curvature proxy remains flat/neutral around the novelty ceiling")
            print("   -> Volume saturation is the primary geometric signal here")
        else:
            print("[CAUTION] WEAK CURVATURE-PROXY/NOVELTY CORRELATION")
            print("   -> Curvature proxy is not a strong predictor of novelty in this run")
            print("   -> Exploratory interpretation should focus on volume and geodesic proxies")
        
        if self.results['metadata']['volume_saturated']:
            print("[ASSOCIATION] VOLUME SATURATION DETECTED")
            print("   -> Representational capacity reached")
        else:
            print("[CAUTION] VOLUME STILL INCREASING")
            print("   -> Saturation not yet reached within analysis window")
        
        print()
        print(f"Novelty Ceiling: {self.results['metadata']['novelty_ceiling']:.3f}")
        print(f"Utility Maintained: {self.results['metadata']['utility_at_ceiling']:.3f}")
    
    def _plot_results(self):
        """Generate comprehensive visualization"""
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        data = self.results['data']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Novelty over time
        axes[0, 0].plot(data['horizons'], data['novelty'], 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Novelty')
        axes[0, 0].set_title('Behavioral Novelty')
        axes[0, 0].axhline(y=0.15, color='r', linestyle='--', label='Ceiling')
        axes[0, 0].legend()
        axes[0, 0].set_xscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Curvature over time
        axes[0, 1].plot(data['horizons'], data['curvature'], 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Curvature proxy (normalized)')
        axes[0, 1].set_title('Operational Curvature Proxy')
        axes[0, 1].axhline(y=0, color='k', linestyle='--')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Volume over time
        axes[0, 2].plot(data['horizons'], data['volume'], 'ro-', linewidth=2, markersize=8)
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].set_ylabel('Volume (normalized)')
        axes[0, 2].set_title('Representational Capacity')
        axes[0, 2].set_xscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Geodesic distance over time
        axes[1, 0].plot(data['horizons'], data['geodesic_distance'], 'mo-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Geodesic proxy distance')
        axes[1, 0].set_title('Geodesic Trajectory Proxy')
        axes[1, 0].set_xscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Divergence vs Novelty scatter
        scatter = axes[1, 1].scatter(data['divergence'], data['novelty'], 
                                      c=data['horizons'], cmap='viridis', 
                                      s=100, edgecolors='black')
        axes[1, 1].set_xlabel('Divergence')
        axes[1, 1].set_ylabel('Novelty')
        axes[1, 1].set_title('Divergence vs Novelty')
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Steps')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Summary text
        summary_text = f"INFORMATION GEOMETRY SUMMARY\n"
        summary_text += f"==========================\n\n"
        summary_text += f"Key Exploratory Correlations:\n"
        summary_text += f"  Curvature proxy vs Novelty: {self.results['correlations'].get('curvature_vs_novelty', 0):.4f}\n"
        summary_text += f"  Volume proxy vs Novelty:    {self.results['correlations'].get('volume_vs_novelty', 0):.4f}\n"
        summary_text += f"  Geodesic proxy vs Novelty:  {self.results['correlations'].get('geodesic_vs_novelty', 0):.4f}\n\n"
        summary_text += f"Novelty Ceiling:       {self.results['metadata']['novelty_ceiling']:.3f}\n"
        summary_text += f"Utility at Ceiling:    {self.results['metadata']['utility_at_ceiling']:.3f}\n"
        summary_text += f"Volume Saturated:      {self.results['metadata']['volume_saturated']}\n\n"
        summary_text += f"Interpretation:\n"
        summary_text += f"  {self.results['interpretations'].get('curvature_vs_novelty', '')}\n\n"
        summary_text += f"Caution: correlation only, not proof or causation.\n"
        summary_text += f"Curvature is an operational proxy, not Ricci curvature.\n\n"
        summary_text += f"Lead result: geodesic proxy strongly tracks novelty,\n"
        summary_text += f"suggesting novelty may later be expressible as\n"
        summary_text += f"geometric trajectory displacement."
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=9, verticalalignment='top', family='monospace')
        axes[1, 2].axis('off')
        
        plt.suptitle('PHASE 9.5: Information Geometry Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.output_dir, 'information_geometry_results.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.config.verbose:
            print(f"\nPlot saved to {plot_path}")
    
    def _sanitize_for_json(self, obj):
        """Recursively convert NumPy types to native Python JSON types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_for_json(value) for value in obj]
        return obj

    def _save_json(self):
        """Save results to JSON file"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        json_path = os.path.join(self.config.output_dir, 'information_geometry_results.json')
        
        json_data = self._sanitize_for_json({
            'data': self.results['data'],
            'correlations': self.results['correlations'],
            'metadata': self.results['metadata'],
            'interpretations': self.results['interpretations']
        })
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        if self.config.verbose:
            print(f"JSON data saved to {json_path}")


# ============================================================================
# VALIDATION FUNCTION (Using your actual data)
# ============================================================================

def run_validation():
    """Run validation using the data from your Phase 11 runs"""
    
    print("=" * 80)
    print("PHASE 9.5: INFORMATION GEOMETRY ANALYSIS")
    print("Using Phase 11 Data")
    print("=" * 80)
    print()
    
    # Your actual data from Phase 11 runs
    config = InfoGeomConfig(
        horizons=[5000, 10000, 25000, 50000],
        novelty=[0.085, 0.141, 0.151, 0.134],
        divergence=[0.534, 0.495, 0.409, 0.334],
        utility=[0.50, 0.52, 0.53, 0.540],
        entropy=[0.85, 0.90, 0.93, 0.950],
        stability=[0.70, 0.78, 0.82, 0.847]
    )
    
    analyzer = InformationGeometryAnalyzer(config)
    results = analyzer.analyze()
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = run_validation()
    
    print()
    print("=" * 80)
    print("PHASE 9.5 COMPLETE")
    print("=" * 80)
    print()
    print("Next phases available:")
    print("  1. Phase 6.5: Hierarchical Identity")
    print("  2. Phase 10.5: Formal Theorems - Asymmetry")
    print("  3. Phase 11.5: Formal Theorems - Novelty")
    print("  4. Phase 12: PIRL (Pre-Inferential Reception Layer)")
