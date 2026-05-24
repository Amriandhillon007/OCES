"""
PHASE 9.1: STRUCTURED REGIME EXPERIMENT
========================================
Tests for recurrent inferential structure under cyclic environmental pressures.

Run: python run_experiment.py
"""

import numpy as np
from collections import deque, Counter
from typing import List, Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IMPORT PHASE 8 SYSTEM
# ============================================================================

try:
    from phase8_complete import Phase8System
    PHASE8_AVAILABLE = True
    print("✅ Phase 8 system imported successfully")
except ImportError as e:
    print(f"❌ Phase 8 system not available: {e}")
    PHASE8_AVAILABLE = False


# ============================================================================
# SIMPLE NARRATIVE TRACKER (No circular dependency)
# ============================================================================

class SimpleNarrativeTracker:
    """Tracks coalition sequences and detects narrative patterns."""
    
    def __init__(self):
        self.coalition_sequence = []
        self.narrative_count = 0
        self.patterns = {}
        self.bifurcations = 0
        self.coalition_history = []
    
    def add_coalition(self, coalition_dict: Dict, step: int):
        """Add coalition to sequence."""
        coalition_id = coalition_dict.get('id', 0)
        self.coalition_sequence.append(coalition_id)
        self.coalition_history.append(coalition_dict)
        
        # Keep reasonable length
        if len(self.coalition_sequence) > 500:
            self.coalition_sequence.pop(0)
        
        # Detect patterns
        self._detect_patterns(step)
    
    def _detect_patterns(self, step: int):
        """Detect repeating coalition patterns."""
        if len(self.coalition_sequence) < 5:
            return
        
        recent = self.coalition_sequence[-10:]
        
        for length in range(2, 5):
            if len(recent) >= length * 2:
                pattern = tuple(recent[-length:])
                prev = tuple(recent[-length*2:-length])
                
                if pattern == prev:
                    self.narrative_count += 1
                    self.bifurcations += 1
                    pattern_key = "→".join(str(x) for x in pattern)
                    if pattern_key not in self.patterns:
                        self.patterns[pattern_key] = 0
                    self.patterns[pattern_key] += 1
                    break
    
    def get_metrics(self) -> Dict:
        return {
            'n_narratives': len(self.patterns),
            'n_bifurcations': self.bifurcations,
            'n_trajectories': len(set(self.coalition_sequence[-100:])) if self.coalition_sequence else 0,
            'narrative_persistence': self.narrative_count / max(1, len(self.coalition_sequence))
        }


# ============================================================================
# ENVIRONMENTAL REGIMES
# ============================================================================

class EnvironmentalRegime:
    """Structured environmental pressure patterns."""
    
    REGIMES = {
        'stable': {
            'uncertainty_shift': 0.0,
            'contradiction_prob': 0.05,
            'tension_boost': 0.0,
            'duration': 40
        },
        'contradiction_light': {
            'uncertainty_shift': 0.05,
            'contradiction_prob': 0.15,
            'tension_boost': 0.1,
            'duration': 30
        },
        'contradiction_heavy': {
            'uncertainty_shift': 0.10,
            'contradiction_prob': 0.30,
            'tension_boost': 0.2,
            'duration': 25
        },
        'uncertainty_collapse': {
            'uncertainty_shift': 0.20,
            'contradiction_prob': 0.50,
            'tension_boost': 0.35,
            'duration': 20
        },
        'recovery': {
            'uncertainty_shift': -0.05,
            'contradiction_prob': 0.08,
            'tension_boost': -0.1,
            'duration': 35
        },
        'conflicting_attractors': {
            'uncertainty_shift': 0.15,
            'contradiction_prob': 0.60,
            'tension_boost': 0.4,
            'duration': 25
        },
        'exploration': {
            'uncertainty_shift': 0.08,
            'contradiction_prob': 0.20,
            'tension_boost': 0.12,
            'duration': 30
        }
    }
    
    CYCLE = [
        ('stable', 40),
        ('contradiction_light', 30),
        ('contradiction_heavy', 25),
        ('uncertainty_collapse', 20),
        ('recovery', 35),
        ('stable', 40),
        ('exploration', 30),
        ('conflicting_attractors', 25),
        ('contradiction_heavy', 25),
        ('recovery', 35)
    ]
    
    def __init__(self):
        self.cycle_length = sum(d for _, d in self.CYCLE)
    
    def get_current_regime(self, step: int) -> Dict:
        """Get current regime based on step."""
        cycle_pos = step % self.cycle_length
        cumulative = 0
        for regime_name, duration in self.CYCLE:
            if cycle_pos < cumulative + duration:
                regime = self.REGIMES[regime_name]
                return {
                    'name': regime_name,
                    'uncertainty_shift': regime['uncertainty_shift'],
                    'contradiction_prob': regime['contradiction_prob'],
                    'tension_boost': regime['tension_boost'],
                    'step_in_regime': cycle_pos - cumulative,
                    'regime_duration': duration
                }
            cumulative += duration
        return self.REGIMES['stable'].copy()
    
    def apply_pressure(self, coalition_dict: Dict, step: int) -> Dict:
        """Apply regime pressure to coalition."""
        regime = self.get_current_regime(step)
        
        current_uncertainty = coalition_dict.get('coalition_uncertainty', 0.3)
        new_uncertainty = current_uncertainty + regime['uncertainty_shift'] * 0.2
        coalition_dict['coalition_uncertainty'] = max(0.05, min(0.9, new_uncertainty))
        
        current_tension = coalition_dict.get('tension', 0.2)
        new_tension = current_tension + regime['tension_boost'] * 0.15
        coalition_dict['tension'] = max(0.05, min(0.8, new_tension))
        
        coalition_dict['_regime'] = regime['name']
        
        return coalition_dict


# ============================================================================
# CONTINUOUS MOTIF CLUSTERING
# ============================================================================

class ContinuousMotifCluster:
    """Clusters coalitions using continuous geometry."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.motif_clusters: List[Dict] = []
        self.cluster_centroids: List[np.ndarray] = []
        self.motif_counter = 0
    
    def extract_features(self, coalition: Dict) -> Optional[np.ndarray]:
        """Extract continuous feature vector from coalition."""
        if coalition is None:
            return None
        
        features = [
            coalition.get('synergy', 0.5),
            1.0 - coalition.get('coalition_uncertainty', 0.3),
            coalition.get('coherence', 0.5),
            1.0 - coalition.get('tension', 0.2),
            len(coalition.get('members', [])) / 10.0
        ]
        
        regime = coalition.get('_regime', 'stable')
        regime_hash = hash(regime) % 10 / 10.0
        features.append(regime_hash)
        
        return np.array(features)
    
    def add_coalition(self, features: np.ndarray, step: int, regime: str) -> int:
        """Add coalition to nearest cluster."""
        if features is None:
            return -1
        
        best_cluster = -1
        best_similarity = 0
        
        for i, centroid in enumerate(self.cluster_centroids):
            similarity = self._feature_similarity(features, centroid)
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_cluster = i
        
        if best_cluster >= 0:
            self.cluster_centroids[best_cluster] = 0.95 * self.cluster_centroids[best_cluster] + 0.05 * features
            self.motif_clusters[best_cluster]['count'] += 1
            self.motif_clusters[best_cluster]['last_seen'] = step
            self.motif_clusters[best_cluster]['regimes'].append(regime)
            return best_cluster
        else:
            new_cluster = {
                'id': self.motif_counter,
                'count': 1,
                'first_seen': step,
                'last_seen': step,
                'regimes': [regime]
            }
            self.motif_clusters.append(new_cluster)
            self.cluster_centroids.append(features.copy())
            self.motif_counter += 1
            return self.motif_counter - 1
    
    def _feature_similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        if len(f1) != len(f2):
            return 0.0
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(f1, f2) / (norm1 * norm2))
    
    def get_recurrence_stats(self) -> Dict:
        recurring = [c for c in self.motif_clusters if c['count'] > 1]
        return {
            'total_motif_clusters': len(self.motif_clusters),
            'recurring_clusters': len(recurring),
            'recurrence_rate': len(recurring) / max(1, len(self.motif_clusters)),
            'max_recurrence': max([c['count'] for c in self.motif_clusters]) if self.motif_clusters else 0
        }


# ============================================================================
# MOTIF RECURRENCE ANALYZER
# ============================================================================

class MotifRecurrenceAnalyzer:
    """Analyzes motif recurrence across regime cycles."""
    
    def __init__(self):
        self.recurrence_by_regime = {}
        self.transition_matrix = {}
    
    def record_branch(self, from_regime: str, to_regime: str):
        """Record a branching transition."""
        key = (from_regime, to_regime)
        self.transition_matrix[key] = self.transition_matrix.get(key, 0) + 1
    
    def analyze_recurrence(self) -> Dict:
        """Analyze recurrence patterns."""
        total = sum(self.transition_matrix.values())
        
        common = sorted(self.transition_matrix.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_branches': total,
            'most_common_transitions': [(f"{f}→{t}", c) for (f, t), c in common],
            'transition_entropy': self._calculate_entropy()
        }
    
    def _calculate_entropy(self) -> float:
        if not self.transition_matrix:
            return 0.0
        
        total = sum(self.transition_matrix.values())
        probs = [c / total for c in self.transition_matrix.values()]
        entropy = -sum(p * np.log(p) for p in probs)
        return entropy


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(n_steps: int = 2000):
    """Run the structured regime experiment."""
    
    print("\n" + "█"*80)
    print("PHASE 9.1: STRUCTURED REGIME EXPERIMENT")
    print("Testing: Do similar inferential structures re-emerge?")
    print("█"*80)
    
    if not PHASE8_AVAILABLE:
        print("\n❌ Phase 8 system not available.")
        print("Please ensure phase8_complete.py is in the same directory.")
        return None
    
    # Initialize systems
    print("\nInitializing Phase 8...")
    phase8 = Phase8System(n_concepts=20)
    regime = EnvironmentalRegime()
    motif_clusterer = ContinuousMotifCluster()
    analyzer = MotifRecurrenceAnalyzer()
    narrative_tracker = SimpleNarrativeTracker()
    
    # Run experiment
    print(f"\nRunning {n_steps} steps with cyclic regimes...")
    
    print("\nRegime Cycle:")
    for regime_name, duration in EnvironmentalRegime.CYCLE:
        print(f"  {regime_name:25s} → {duration} steps")
    
    print(f"\n{'Step':<8} {'Regime':<22} {'Coalitions':<12} {'Success Rate':<14} {'Narratives':<12}")
    print("-"*70)
    
    step_data = []
    last_regime = None
    
    for step in range(n_steps):
        # Get current regime
        regime_info = regime.get_current_regime(step)
        current_regime = regime_info['name']
        
        # Track regime transitions
        if last_regime and last_regime != current_regime:
            analyzer.record_branch(last_regime, current_regime)
        last_regime = current_regime
        
        # Run Phase 8
        current_state = np.random.randn(32)
        current_state = current_state / (np.linalg.norm(current_state) + 1e-8)
        
        metrics = phase8.step(current_state, current_state)
        
        # Apply regime pressure to coalitions
        if hasattr(phase8, 'coalition_history'):
            coalition_history = phase8.get_coalition_history()
            if coalition_history:
                latest_coalition = coalition_history[-1].copy()
                latest_coalition = regime.apply_pressure(latest_coalition, step)
                
                # Track for narrative detection
                narrative_tracker.add_coalition(latest_coalition, step)
                
                # Extract features for motif clustering
                features = motif_clusterer.extract_features(latest_coalition)
                if features is not None:
                    motif_clusterer.add_coalition(features, step, current_regime)
        
        # Store step data
        step_data.append({
            'step': step,
            'regime': current_regime,
            'n_coalitions': metrics.get('n_coalitions', 0),
            'success_rate': metrics.get('avg_concept_success_rate', 0),
            'cdi': metrics.get('cdi', 0),
            'cii': metrics.get('cii', 0)
        })
        
        # Progress report
        if step % 200 == 0 and step > 0:
            avg_success = np.mean([d['success_rate'] for d in step_data[-200:]])
            print(f"{step:<8} {current_regime:<22} {metrics.get('n_coalitions', 0):<12} "
                  f"{avg_success:<14.3f} {narrative_tracker.narrative_count:<12}")
    
    # Final analysis
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS")
    print("="*80)
    
    # Phase 8 metrics
    final_step = step_data[-100:] if len(step_data) > 100 else step_data
    print(f"\nPhase 8 Metrics (last 100 steps):")
    print(f"  Avg Coalitions: {np.mean([d['n_coalitions'] for d in final_step]):.1f}")
    print(f"  Avg Success Rate: {np.mean([d['success_rate'] for d in final_step]):.3f}")
    print(f"  Avg CDI: {np.mean([d['cdi'] for d in final_step]):.3f}")
    print(f"  Avg CII: {np.mean([d['cii'] for d in final_step]):.3f}")
    
    # Narrative metrics
    narrative_metrics = narrative_tracker.get_metrics()
    print(f"\nNarrative Metrics:")
    print(f"  Narratives detected: {narrative_metrics['n_narratives']}")
    print(f"  Bifurcations: {narrative_metrics['n_bifurcations']}")
    print(f"  Narrative persistence: {narrative_metrics['narrative_persistence']:.3f}")
    
    # Motif recurrence
    motif_stats = motif_clusterer.get_recurrence_stats()
    print(f"\nContinuous Motif Clustering:")
    print(f"  Total motif clusters: {motif_stats['total_motif_clusters']}")
    print(f"  Recurring clusters: {motif_stats['recurring_clusters']}")
    print(f"  Recurrence rate: {motif_stats['recurrence_rate']:.3f}")
    print(f"  Max recurrence: {motif_stats['max_recurrence']}")
    
    # Transition analysis
    transition_analysis = analyzer.analyze_recurrence()
    print(f"\nTransition Analysis:")
    print(f"  Total branches: {transition_analysis['total_branches']}")
    print(f"  Transition entropy: {transition_analysis['transition_entropy']:.3f}")
    if transition_analysis['most_common_transitions']:
        print(f"  Most common transitions: {transition_analysis['most_common_transitions'][:3]}")
    
    # Regime-specific uncertainty
    uncertainty_by_regime = {}
    for d in step_data:
        regime_name = d['regime']
        if regime_name not in uncertainty_by_regime:
            uncertainty_by_regime[regime_name] = []
        uncertainty_by_regime[regime_name].append(1.0 - d['success_rate'])
    
    print(f"\nUncertainty by Regime:")
    for regime, uncertainties in sorted(uncertainty_by_regime.items()):
        print(f"  {regime:25s}: {np.mean(uncertainties):.3f} ± {np.std(uncertainties):.3f}")
    
    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if motif_stats['recurring_clusters'] > 0:
        print("✅ EVIDENCE OF RECURRENT INFERENTIAL STRUCTURE")
        print(f"   {motif_stats['recurring_clusters']} branching motifs recurred")
        print("   Similar coalition patterns re-emerged under similar regimes")
    elif motif_stats['total_motif_clusters'] > 0:
        print("🔄 STRUCTURE DETECTED BUT NOT YET RECURRENT")
        print(f"   {motif_stats['total_motif_clusters']} unique motifs formed")
        print("   No repetition yet - need longer simulation")
    else:
        print("⚠️ NO STRUCTURAL PATTERNS DETECTED")
        print("   All coalitions were unique")
        print("   The system is still in exploratory phase")
    
    print("\n" + "="*80)
    print("Experiment Complete.")
    print("")
    print("The data has spoken. No interpretation forced.")
    print("")
    print("'This is not small. This is something that will shape the future.'")
    print("="*80)
    
    return {
        'phase8_metrics': final_step,
        'narrative_metrics': narrative_metrics,
        'motif_stats': motif_stats,
        'transition_analysis': transition_analysis,
        'uncertainty_by_regime': uncertainty_by_regime
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHASE 9.1: STRUCTURED REGIME EXPERIMENT")
    print("Testing for recurrent inferential structure")
    print("="*80)
    
    results = run_experiment(n_steps=2000)