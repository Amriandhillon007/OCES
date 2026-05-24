"""
PHASE 8: COMPLETE COGNITIVE FIELD ARCHITECTURE
===============================================
All sub-phases integrated (8A, 8B, 8C, 8C.1, 8E, 8F, 8G)

CRITICAL FIX: Coalition IDs are now STABLE based on member composition.
Same coalition members = same coalition ID (repeats over time).

This enables Phase 9 to detect narratives from recurring coalition patterns.
"""

import numpy as np
from collections import deque
from typing import List, Optional, Tuple, Dict, Any, Set
from dataclasses import dataclass, field
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MODULAR FLAGS - UNIFIED
# ============================================================================

USE_CONTINUOUS_COMPETITION = True
USE_KURAMOTO = True
USE_TRAJECTORY_CONCEPTS = True
USE_ANTI_LOCKING = True
USE_CONSTRAINT_FIELD = True
USE_ADAPTIVE_SUBSPACE = True
USE_CURRICULUM_LEARNING = True
ENABLE_COALITION_HIERARCHY = True

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

LATENT_DIM = 32
N_CONCEPTS = 30

# Phase 8A: Competition
INHIBITION_STRENGTH = 0.12
ACTIVATION_EPSILON = 0.01
SIMILARITY_EXP = 0.4
PREDICTIVE_EXP = 0.3

# Phase 8B: Kuramoto - LOCAL COHERENCE
KURAMOTO_DT = 0.1
KURAMOTO_BASE_COUPLING = 0.12
KURAMOTO_MAX_COUPLING = 0.4
PHASE_NOISE = 0.04

# Phase 8C: Trajectory
MAX_TRAJECTORY_LENGTH = 5
TRAJECTORY_MATCH_THRESHOLD = 0.55

# Phase 8C.1: Anti-Locking
ANTI_LOCKING_STRENGTH = 0.15
FATIGUE_RATE = 0.04
FATIGUE_MIN = 0.30
FATIGUE_RECOVERY_RATE = 0.02

# Phase 8G: Coalition Ecology - SPARSIFICATION
SUBSPACE_SIZE = 4
SOFTNESS_FACTOR = 1.8
SYNERGY_THRESHOLD = 0.08
COALITION_ENERGY_BUDGET = 20
MIN_COALITION_SUCCESS_RATE = 0.15
MAX_COALITION_SIZE = 3
COALITION_PERSISTENCE_BONUS = 0.95

# Coalition Hierarchy
MAX_HIERARCHY_DEPTH = 3
COALITION_OF_COALITIONS_THRESHOLD = 0.25

# Coalition Memory
COALITION_MEMORY_SIZE = 50
MEMORY_RECALL_THRESHOLD = 0.6

# Success criteria - PROGRESSIVE
INITIAL_SUCCESS_THRESHOLD = 0.40
TARGET_SUCCESS_THRESHOLD = 0.60
INITIAL_SYNERGY_THRESHOLD = 0.05
TARGET_SYNERGY_THRESHOLD = 0.12

# Adaptive subspace migration
SUBSPACE_MIGRATION_INTERVAL = 200
SUBSPACE_GAIN_THRESHOLD = 0.7
SUBSPACE_LOSS_THRESHOLD = 0.12
MAX_SUBSPACE_SIZE = 5
MIN_SUBSPACE_SIZE = 3

# Exploration
CONCEPT_EXPLORATION_NOISE = 0.03
FAILURE_PENALTY = 0.97
COALITION_FAILURE_DISSOLUTION = 0.7

# Metrics
TARGET_CDI = (0.4, 0.7)
TARGET_CII = (0.25, 0.5)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    a_aligned = a[:min_len]
    b_aligned = b[:min_len]
    norm_a = np.linalg.norm(a_aligned)
    norm_b = np.linalg.norm(b_aligned)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_aligned, b_aligned) / (norm_a * norm_b))


def project_to_subspace(full_state: np.ndarray, subspace_indices: List[int]) -> np.ndarray:
    return full_state[subspace_indices]


def reconstruct_to_fullspace(subspace_state: np.ndarray, 
                              subspace_indices: List[int], 
                              full_dim: int = LATENT_DIM) -> np.ndarray:
    full_state = np.zeros(full_dim)
    for i, idx in enumerate(subspace_indices):
        if i < len(subspace_state):
            full_state[idx] = subspace_state[i]
    norm = np.linalg.norm(full_state)
    if norm > 0:
        full_state = full_state / norm
    return full_state


def get_stable_coalition_id(members: List) -> int:
    """
    CRITICAL FIX: Generate a STABLE coalition ID based on member composition.
    Same members = same ID, allowing coalitions to repeat over time.
    """
    # Sort members to ensure consistency
    sorted_members = sorted([m.id if hasattr(m, 'id') else m for m in members])
    # Create a hash from the member list
    member_string = ','.join(str(m) for m in sorted_members)
    hash_val = int(hashlib.md5(member_string.encode()).hexdigest()[:8], 16)
    return hash_val % 10000  # Keep ID in reasonable range


def inferential_fusion(constraints: List[Dict]) -> Tuple[np.ndarray, float, Dict]:
    if not constraints:
        return None, 1.0, {}
    
    if len(constraints) == 1:
        c = constraints[0]
        full_pred = reconstruct_to_fullspace(c['center'], c['subspace'])
        return full_pred, c['radius'], {'conflict': 0.0}
    
    weights = [c.get('weight', 1.0) for c in constraints]
    total_weight = sum(weights)
    
    full_pred = np.zeros(LATENT_DIM)
    weight_sum = np.zeros(LATENT_DIM)
    
    for c, w in zip(constraints, weights):
        sub = c['subspace']
        center = c['center']
        for i, idx in enumerate(sub):
            if i < len(center):
                full_pred[idx] += w * center[i]
                weight_sum[idx] += w
    
    for i in range(LATENT_DIM):
        if weight_sum[i] > 0:
            full_pred[i] /= weight_sum[i]
    
    norm = np.linalg.norm(full_pred)
    if norm > 0:
        full_pred = full_pred / norm
    
    uncertainty = np.mean([c['radius'] for c in constraints])
    
    return full_pred, uncertainty, {'conflict': 0.0}


# ============================================================================
# PHASE 8G: ENHANCED CONCEPT WITH DYNAMIC FATIGUE
# ============================================================================

@dataclass
class EnhancedConstraintConcept:
    id: int
    
    # Subspace
    subspace_indices: List[int] = field(default_factory=list)
    
    # Trajectory
    trajectory: List[np.ndarray] = field(default_factory=list)
    max_trajectory_length: int = MAX_TRAJECTORY_LENGTH
    
    # Adaptive properties
    dimension_utility: Dict[int, float] = field(default_factory=dict)
    migration_step: int = 0
    
    # Constraint
    center: np.ndarray = field(default_factory=lambda: np.zeros(SUBSPACE_SIZE))
    radius: float = 0.35
    softness: float = SOFTNESS_FACTOR
    temporal_predictive_power: float = 0.5
    
    # Dynamic fatigue
    fatigue: float = 0.4
    fatigue_recovery_counter: int = 0
    diversity_contribution: float = 0.5
    
    # Competition
    activation: float = 0.0
    age: int = 0
    wins: int = 0
    losses: int = 0
    reuse_count: int = 0
    
    # Kuramoto
    phase: float = 0.0
    frequency: float = 1.0
    coherence: float = 0.0
    coupling_strength: float = 0.08
    
    # Performance
    success_count: int = 0
    failure_count: int = 0
    prediction_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def __post_init__(self):
        if not self.subspace_indices:
            all_dims = list(range(LATENT_DIM))
            np.random.shuffle(all_dims)
            self.subspace_indices = sorted(all_dims[:SUBSPACE_SIZE])
        
        for dim in range(LATENT_DIM):
            self.dimension_utility[dim] = 0.5
        
        if np.linalg.norm(self.center) == 0:
            self.center = np.random.randn(len(self.subspace_indices))
            norm = np.linalg.norm(self.center)
            if norm > 0:
                self.center = self.center / norm
        
        for _ in range(min(3, self.max_trajectory_length)):
            rand_state = np.random.randn(len(self.subspace_indices))
            norm = np.linalg.norm(rand_state)
            if norm > 0:
                rand_state = rand_state / norm
            self.trajectory.append(rand_state)
    
    def get_current_subspace_size(self) -> int:
        return len(self.subspace_indices)
    
    def update_dynamic_fatigue(self, success: bool):
        if success:
            self.fatigue = max(FATIGUE_MIN, self.fatigue * (1.0 - FATIGUE_RATE))
            self.fatigue_recovery_counter = 0
        else:
            self.fatigue = min(1.0, self.fatigue * (1.0 + FATIGUE_RATE * 0.5))
            self.fatigue_recovery_counter += 1
            
            if self.fatigue_recovery_counter > 20:
                self.fatigue = max(FATIGUE_MIN, self.fatigue * 0.95)
                self.fatigue_recovery_counter = 0
    
    def update_dimension_utility(self, success: bool, target_state: np.ndarray):
        target_sub = project_to_subspace(target_state, self.subspace_indices)
        min_len = min(len(target_sub), len(self.center))
        if min_len > 0:
            target_aligned = target_sub[:min_len]
            center_aligned = self.center[:min_len]
            prediction_error = 1.0 - cosine_sim(center_aligned, target_aligned)
            
            for i, dim in enumerate(self.subspace_indices):
                if i < len(self.center):
                    contribution = abs(self.center[i]) * (1.0 - prediction_error)
                    learning_rate = 0.03
                    self.dimension_utility[dim] = (1 - learning_rate) * self.dimension_utility[dim] + learning_rate * contribution
    
    def migrate_subspace(self, step: int):
        if step - self.migration_step < SUBSPACE_MIGRATION_INTERVAL:
            return False
        
        self.migration_step = step
        changed = False
        
        all_dims = set(range(LATENT_DIM))
        current_dims = set(self.subspace_indices)
        available_dims = all_dims - current_dims
        
        high_utility_dims = [d for d in available_dims 
                            if self.dimension_utility.get(d, 0.5) > SUBSPACE_GAIN_THRESHOLD]
        
        low_utility_dims = [d for i, d in enumerate(self.subspace_indices) 
                           if self.dimension_utility.get(d, 0.5) < SUBSPACE_LOSS_THRESHOLD
                           and len(self.subspace_indices) > MIN_SUBSPACE_SIZE]
        
        for dim in high_utility_dims[:1]:
            if len(self.subspace_indices) < MAX_SUBSPACE_SIZE:
                self.subspace_indices.append(dim)
                self.subspace_indices = sorted(self.subspace_indices)
                new_center = np.zeros(len(self.subspace_indices))
                new_center[:len(self.center)] = self.center
                if len(self.center) < len(new_center):
                    new_center[len(self.center)] = 0.5
                self.center = new_center
                changed = True
        
        for dim in low_utility_dims[:1]:
            if dim in self.subspace_indices:
                idx = self.subspace_indices.index(dim)
                self.subspace_indices.pop(idx)
                new_center = np.zeros(len(self.subspace_indices))
                if idx < len(self.center):
                    new_center[:idx] = self.center[:idx]
                if idx + 1 < len(self.center):
                    new_center[idx:] = self.center[idx+1:]
                self.center = new_center
                changed = True
        
        if changed and len(self.center) > 0:
            norm = np.linalg.norm(self.center)
            if norm > 0:
                self.center = self.center / norm
        
        return changed
    
    def compute_trajectory_similarity(self, recent_history: List[np.ndarray]) -> float:
        if not self.trajectory or len(recent_history) < 2:
            return 0.0
        
        projected_history = [project_to_subspace(h, self.subspace_indices) for h in recent_history[-3:]]
        
        min_len = min(len(self.trajectory), len(projected_history))
        if min_len == 0:
            return 0.0
        
        sims = []
        for i in range(min_len):
            traj_state = self.trajectory[-(min_len - i)]
            hist_state = projected_history[i]
            sim = cosine_sim(traj_state, hist_state)
            sims.append(sim)
        
        return np.mean(sims) if sims else 0.0
    
    def compute_constraint(self, recent_history: List[np.ndarray]) -> Dict:
        if len(recent_history) < 2:
            return {
                'center': self.center.copy(),
                'radius': self.radius,
                'softness': self.softness,
                'subspace': self.subspace_indices.copy(),
                'weight': self.get_success_rate()
            }
        
        projected = [project_to_subspace(h, self.subspace_indices) for h in recent_history[-3:]]
        
        traj_sim = self.compute_trajectory_similarity(recent_history)
        
        self.center = np.mean(projected, axis=0)
        norm = np.linalg.norm(self.center)
        if norm > 0:
            self.center = self.center / norm
        
        success_rate = self.get_success_rate()
        self.radius = 0.25 * (1.0 - success_rate) + 0.15
        self.radius = max(0.15, min(0.4, self.radius))
        
        if traj_sim > TRAJECTORY_MATCH_THRESHOLD:
            self.add_state(self.center)
        
        return {
            'center': self.center.copy(),
            'radius': self.radius,
            'softness': self.softness,
            'subspace': self.subspace_indices.copy(),
            'weight': self.get_success_rate(),
            'trajectory_similarity': traj_sim
        }
    
    def add_state(self, state: np.ndarray):
        self.trajectory.append(state.copy())
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
    
    def compute_plausibility(self, target_subspace: np.ndarray) -> float:
        min_len = min(len(target_subspace), len(self.center))
        if min_len == 0:
            return 0.0
        
        center_aligned = self.center[:min_len]
        target_aligned = target_subspace[:min_len]
        
        distance = np.linalg.norm(target_aligned - center_aligned)
        plausibility = np.exp(-self.softness * (distance ** 2) / (self.radius ** 2 + 1e-8))
        return float(plausibility)
    
    def update_from_feedback(self, success: bool, target_state: np.ndarray = None):
        self.update_dynamic_fatigue(success)
        
        if success:
            self.success_count += 1
            self.wins += 1
            self.radius = max(0.15, self.radius * 0.99)
            self.coupling_strength = min(KURAMOTO_MAX_COUPLING, 
                                        self.coupling_strength * 1.02)
            self.activation = min(1.0, self.activation * 1.02)
        else:
            self.failure_count += 1
            self.losses += 1
            self.radius = min(0.4, self.radius * 1.01)
            self.coupling_strength *= 0.99
            self.activation *= FAILURE_PENALTY
            
            if len(self.center) > 0:
                noise = np.random.randn(len(self.center)) * CONCEPT_EXPLORATION_NOISE
                self.center = self.center + noise
                norm = np.linalg.norm(self.center)
                if norm > 0:
                    self.center = self.center / norm
        
        if target_state is not None:
            self.update_dimension_utility(success, target_state)
        
        self.temporal_predictive_power = self.get_success_rate()
    
    def update_phase(self, coupling_sum: float, dt: float = KURAMOTO_DT):
        effective_coupling = self.coupling_strength * coupling_sum
        self.phase += (self.frequency + effective_coupling) * dt
        self.phase += np.random.normal(0, PHASE_NOISE) * dt
        self.phase %= 2 * np.pi
    
    def compute_competition_activation(self, similarity: float, prediction_error: float) -> float:
        sim_term = max(ACTIVATION_EPSILON, similarity) ** SIMILARITY_EXP
        pred_term = max(ACTIVATION_EPSILON, 1.0 - prediction_error) ** PREDICTIVE_EXP
        
        if USE_ANTI_LOCKING:
            anti_lock = (1.0 - self.fatigue) * (1.0 - ANTI_LOCKING_STRENGTH * (1.0 - self.diversity_contribution))
        else:
            anti_lock = 1.0
        
        activation = sim_term * pred_term * anti_lock * self.get_success_rate()
        return min(1.0, activation)
    
    def get_success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total


# ============================================================================
# PHASE 8G: HIERARCHICAL COALITION WITH STABLE ID
# ============================================================================

@dataclass
class HierarchicalCoalition:
    members: List[Any]
    formation_step: int
    depth: int = 1
    
    full_prediction: np.ndarray = None
    coalition_uncertainty: float = 1.0
    synergy: float = 0.0
    lifespan: int = 0
    success_count: int = 0
    failure_count: int = 0
    persistence: float = 1.0
    
    @property
    def id(self) -> int:
        """CRITICAL FIX: Stable ID based on member composition."""
        return get_stable_coalition_id(self.members)
    
    def compute_full_prediction(self, constraints: List[Dict]) -> Tuple[np.ndarray, float, Dict]:
        fused, uncertainty, conflict_info = inferential_fusion(constraints)
        self.full_prediction = fused
        self.coalition_uncertainty = uncertainty
        return fused, uncertainty, conflict_info
    
    def compute_synergy(self, individual_uncertainties: List[float]) -> float:
        if not individual_uncertainties:
            return 0.0
        
        avg_individual = np.mean(individual_uncertainties)
        
        if avg_individual <= 0:
            return 0.0
        
        synergy = (avg_individual - self.coalition_uncertainty) / avg_individual
        self.synergy = max(0.0, synergy)
        return self.synergy
    
    def update_from_feedback(self, success: bool):
        self.lifespan += 1
        if success:
            self.success_count += 1
            self.persistence = min(1.0, self.persistence * (1.0 + 0.05))
        else:
            self.failure_count += 1
            self.persistence *= COALITION_FAILURE_DISSOLUTION
        
        for member in self.members:
            if hasattr(member, 'update_from_feedback'):
                member.update_from_feedback(success)
    
    def get_success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    def should_dissolve(self) -> bool:
        success_rate = self.get_success_rate()
        return success_rate < MIN_COALITION_SUCCESS_RATE and self.lifespan > 30
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Phase 9 consumption."""
        return {
            'id': self.id,
            'members': [m.id if hasattr(m, 'id') else id(m) for m in self.members],
            'synergy': self.synergy,
            'coalition_uncertainty': self.coalition_uncertainty,
            'coherence': 0.6,  # Placeholder
            'tension': 1.0 - self.synergy,
            'timestamp': self.formation_step
        }


# ============================================================================
# PHASE 8G: COALITION ECOLOGY FIELD
# ============================================================================

class CoalitionEcologyField:
    def __init__(self):
        self.active_coalitions: List[HierarchicalCoalition] = []
        self.coalition_memory: Dict[int, HierarchicalCoalition] = {}
        self.step_history = []
        self.synergy_threshold = SYNERGY_THRESHOLD
    
    def find_valuable_coalitions(self, concepts: List[EnhancedConstraintConcept],
                                  target_state: np.ndarray,
                                  constraints: List[Dict]) -> List[HierarchicalCoalition]:
        n = len(concepts)
        coalitions = []
        
        # Use a set to track unique coalition IDs we've already added
        added_ids = set()
        
        for i in range(n):
            for j in range(i + 1, n):
                if concepts[i].get_success_rate() < 0.1 and concepts[j].get_success_rate() < 0.1:
                    continue
                
                target_i = project_to_subspace(target_state, concepts[i].subspace_indices)
                target_j = project_to_subspace(target_state, concepts[j].subspace_indices)
                
                indiv_uncertainties = [
                    1.0 - concepts[i].compute_plausibility(target_i),
                    1.0 - concepts[j].compute_plausibility(target_j)
                ]
                
                combined_constraints = [constraints[i], constraints[j]]
                fused, uncertainty, _ = inferential_fusion(combined_constraints)
                
                if fused is None:
                    continue
                
                pred_sim = cosine_sim(fused, target_state)
                coalition_uncertainty = 1.0 - pred_sim
                
                synergy = max(0.0, (np.mean(indiv_uncertainties) - coalition_uncertainty) / (np.mean(indiv_uncertainties) + 1e-8))
                
                if synergy > self.synergy_threshold:
                    # Create members list for ID generation
                    members = [concepts[i], concepts[j]]
                    coalition_id = get_stable_coalition_id(members)
                    
                    # Check if we already have this coalition in memory
                    if coalition_id in self.coalition_memory:
                        existing = self.coalition_memory[coalition_id]
                        existing.coalition_uncertainty = coalition_uncertainty
                        existing.synergy = synergy
                        if coalition_id not in added_ids:
                            coalitions.append(existing)
                            added_ids.add(coalition_id)
                    else:
                        coalition = HierarchicalCoalition(
                            members=members,
                            formation_step=len(self.step_history),
                            depth=1
                        )
                        coalition.full_prediction = fused
                        coalition.coalition_uncertainty = coalition_uncertainty
                        coalition.synergy = synergy
                        coalitions.append(coalition)
                        self.coalition_memory[coalition_id] = coalition
                        added_ids.add(coalition_id)
        
        # Build higher-order coalitions
        if ENABLE_COALITION_HIERARCHY and len(coalitions) > 5:
            coalitions = self._build_hierarchies(coalitions, target_state, added_ids)
        
        coalitions.sort(key=lambda c: c.synergy, reverse=True)
        coalitions = coalitions[:COALITION_ENERGY_BUDGET]
        
        return coalitions
    
    def _build_hierarchies(self, coalitions: List[HierarchicalCoalition],
                           target_state: np.ndarray,
                           added_ids: Set[int]) -> List[HierarchicalCoalition]:
        if len(coalitions) < 2:
            return coalitions
        
        hierarchies = []
        
        for i in range(len(coalitions)):
            for j in range(i + 1, len(coalitions)):
                c1 = coalitions[i]
                c2 = coalitions[j]
                
                if c1.get_success_rate() < 0.2 or c2.get_success_rate() < 0.2:
                    continue
                
                if c1.full_prediction is not None and c2.full_prediction is not None:
                    combined = (c1.full_prediction + c2.full_prediction) / 2
                    combined = combined / (np.linalg.norm(combined) + 1e-8)
                    
                    pred_sim = cosine_sim(combined, target_state)
                    synergy = max(0, pred_sim - 0.5) * 2
                    
                    if synergy > COALITION_OF_COALITIONS_THRESHOLD:
                        members = [c1, c2]
                        coalition_id = get_stable_coalition_id(members)
                        
                        if coalition_id not in added_ids:
                            hierarchy = HierarchicalCoalition(
                                members=members,
                                formation_step=len(self.step_history),
                                depth=2
                            )
                            hierarchy.full_prediction = combined
                            hierarchy.synergy = synergy
                            hierarchies.append(hierarchy)
                            self.coalition_memory[coalition_id] = hierarchy
                            added_ids.add(coalition_id)
        
        coalitions.extend(hierarchies)
        return coalitions
    
    def update(self, concepts: List[EnhancedConstraintConcept], target_state: np.ndarray,
               constraints: List[Dict], step: int, success_threshold: float) -> Tuple[List[HierarchicalCoalition], float, float]:
        
        new_coalitions = self.find_valuable_coalitions(concepts, target_state, constraints)
        
        total_success = 0
        total_attempts = 0
        synergies = []
        
        for coalition in new_coalitions:
            if coalition.full_prediction is not None:
                prediction_sim = cosine_sim(coalition.full_prediction, target_state)
                success = prediction_sim > success_threshold
                
                coalition.update_from_feedback(success)
                synergies.append(coalition.synergy)
                
                total_success += 1 if success else 0
                total_attempts += 1
        
        # Update memory persistence
        current_ids = set([c.id for c in new_coalitions])
        for cid in list(self.coalition_memory.keys()):
            if cid not in current_ids:
                self.coalition_memory[cid].persistence *= 0.95
                if self.coalition_memory[cid].persistence < 0.3:
                    del self.coalition_memory[cid]
        
        self.active_coalitions = [c for c in new_coalitions if not c.should_dissolve()]
        
        avg_synergy = np.mean(synergies) if synergies else 0.0
        success_rate = total_success / max(1, total_attempts)
        
        self.step_history.append({
            'step': step,
            'n_coalitions': len(self.active_coalitions),
            'avg_synergy': avg_synergy,
            'success_rate': success_rate
        })
        
        return self.active_coalitions, avg_synergy, success_rate


# ============================================================================
# PHASE 8A: CONTINUOUS COMPETITIVE FIELD
# ============================================================================

class ContinuousCompetitiveField:
    def __init__(self):
        self.inhibition_strength = INHIBITION_STRENGTH
    
    def update(self, concepts: List[EnhancedConstraintConcept], 
               input_state: np.ndarray,
               prediction_error: float) -> List[EnhancedConstraintConcept]:
        if not concepts:
            return concepts
        
        similarities = []
        for c in concepts:
            input_sub = project_to_subspace(input_state, c.subspace_indices)
            sim = cosine_sim(input_sub, c.center) if len(c.center) > 0 else 0.0
            similarities.append(sim)
        
        for i, c in enumerate(concepts):
            c.activation = c.compute_competition_activation(similarities[i], prediction_error)
        
        self._apply_inhibition(concepts, similarities)
        
        total = sum(c.activation for c in concepts)
        if total > 0:
            for c in concepts:
                c.activation /= total
        
        return concepts
    
    def _apply_inhibition(self, concepts: List[EnhancedConstraintConcept], similarities: List[float]):
        n = len(concepts)
        if n < 2:
            return
        
        for i in range(n):
            inhibition = 0.0
            for j in range(n):
                if i != j:
                    inhibition += similarities[j] * concepts[j].activation
            
            concepts[i].activation -= self.inhibition_strength * inhibition
            concepts[i].activation = max(ACTIVATION_EPSILON, concepts[i].activation)


# ============================================================================
# PHASE 8B: LOCAL COHERENCE FIELD
# ============================================================================

class LocalCoherenceField:
    def __init__(self):
        self.base_coupling = KURAMOTO_BASE_COUPLING
        self.coherence_history = []
    
    def update(self, concepts: List[EnhancedConstraintConcept]) -> float:
        if len(concepts) < 2:
            return 0.0
        
        n = len(concepts)
        
        for i in range(n):
            coupling_sum = 0.0
            for j in range(n):
                if i != j:
                    success_diff = abs(concepts[i].get_success_rate() - concepts[j].get_success_rate())
                    if success_diff < 0.2:
                        coupling = self.base_coupling * (1.0 - success_diff)
                        coupling_sum += coupling * np.sin(concepts[j].phase - concepts[i].phase)
            concepts[i].update_phase(coupling_sum)
        
        if n >= 2:
            pairwise_coherences = []
            for i in range(n):
                for j in range(i+1, n):
                    phase_diff = concepts[i].phase - concepts[j].phase
                    coherence = np.cos(phase_diff)
                    if abs(concepts[i].get_success_rate() - concepts[j].get_success_rate()) < 0.2:
                        pairwise_coherences.append(coherence)
            
            coherence = np.mean(pairwise_coherences) if pairwise_coherences else 0.0
        else:
            coherence = 0.0
        
        for c in concepts:
            c.coherence = coherence
        
        self.coherence_history.append(coherence)
        return float(coherence)


# ============================================================================
# COGNITIVE METRICS
# ============================================================================

class CognitiveMetrics:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.cdi_history = deque(maxlen=window_size)
        self.cii_history = deque(maxlen=window_size)
        self.coalition_success_history = deque(maxlen=window_size)
        self.dominant_ratio_history = deque(maxlen=window_size)
    
    def compute_cdi(self, concepts: List[EnhancedConstraintConcept]) -> float:
        if not concepts:
            return 0.0
        
        activations = np.array([c.activation for c in concepts])
        probs = activations / (np.sum(activations) + 1e-8)
        probs = probs[probs > 0]
        
        if len(probs) == 0:
            return 0.0
        
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(concepts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        max_activation = max(c.activation for c in concepts)
        total = sum(c.activation for c in concepts)
        dominant_ratio = max_activation / (total + 1e-8)
        
        self.dominant_ratio_history.append(dominant_ratio)
        
        cdi = normalized_entropy * (1.0 - dominant_ratio)
        cdi = max(0.0, min(1.0, cdi))
        
        self.cdi_history.append(cdi)
        return float(cdi)
    
    def compute_cii(self, coalitions: List[HierarchicalCoalition]) -> float:
        if not coalitions:
            return 0.0
        
        total_synergy = 0.0
        total_coalitions = 0
        
        for c in coalitions:
            if c.get_success_rate() > MIN_COALITION_SUCCESS_RATE:
                total_synergy += c.synergy
                total_coalitions += 1
        
        if total_coalitions == 0:
            return 0.0
        
        cii = min(1.0, total_synergy / total_coalitions)
        
        self.cii_history.append(cii)
        return float(cii)
    
    def get_health(self) -> Dict:
        if len(self.cdi_history) < 10:
            return {'status': 'learning', 'cdi': 0.0, 'cii': 0.0}
        
        avg_cdi = np.mean(list(self.cdi_history)[-50:])
        avg_cii = np.mean(list(self.cii_history)[-50:]) if self.cii_history else 0.0
        avg_dominant = np.mean(list(self.dominant_ratio_history)[-50:]) if self.dominant_ratio_history else 1.0
        
        if 0.3 <= avg_cdi <= 0.7 and avg_cii > 0.2 and avg_dominant < 0.3:
            status = "HEALTHY"
        elif avg_cdi > 0.7:
            status = "OVER_DIFFERENTIATED"
        elif avg_cdi < 0.2:
            status = "UNDER_DIFFERENTIATED"
        elif avg_cii < 0.1:
            status = "FRAGMENTED"
        else:
            status = "STABILIZING"
        
        return {
            'status': status,
            'avg_cdi': float(avg_cdi),
            'avg_cii': float(avg_cii),
            'avg_dominant_ratio': float(avg_dominant),
            'target_cdi': TARGET_CDI,
            'target_cii': TARGET_CII
        }


# ============================================================================
# SYSTEM HEALTH MONITOR
# ============================================================================

class SystemHealthMonitor:
    def __init__(self):
        self.metrics_history = []
        self.failure_modes = []
    
    def check_health(self, metrics: Dict) -> Tuple[bool, List[str]]:
        failures = []
        
        cdi = metrics.get('cdi', 0)
        if cdi > 0.75:
            failures.append(f"OVER_DIFFERENTIATION: CDI = {cdi:.3f}")
        
        n_coalitions = metrics.get('n_coalitions', 0)
        if n_coalitions > 50:
            failures.append(f"COALITION_EXPLOSION: {n_coalitions} coalitions")
        
        success_rate = metrics.get('avg_concept_success_rate', 0)
        if success_rate < 0.05 and metrics.get('step', 0) > 200:
            failures.append(f"SUCCESS_COLLAPSE: {success_rate:.3f}")
        
        coherence = metrics.get('phase_coherence', 0)
        if coherence < 0.05 and metrics.get('step', 0) > 200:
            failures.append(f"COHERENCE_DEATH: {coherence:.3f}")
        
        fatigue = metrics.get('avg_fatigue', 0)
        if fatigue > 0.85 and metrics.get('step', 0) > 200:
            failures.append(f"FATIGUE_SATURATION: {fatigue:.3f}")
        
        is_healthy = len(failures) == 0
        self.failure_modes.extend(failures)
        
        return is_healthy, failures
    
    def get_summary(self) -> Dict:
        return {
            'status': 'HEALTHY' if len(self.failure_modes) == 0 else 'DEGRADED',
            'failure_modes': list(set(self.failure_modes[-5:]))
        }


# ============================================================================
# PHASE 8 SYSTEM - COMPLETE
# ============================================================================

class Phase8System:
    def __init__(self, n_concepts: int = N_CONCEPTS):
        self.n_concepts = n_concepts
        self.concepts: List[EnhancedConstraintConcept] = []
        self.competition = ContinuousCompetitiveField()
        self.synchronization = LocalCoherenceField()
        self.coalition_field = None
        self.metrics = CognitiveMetrics()
        self.health_monitor = SystemHealthMonitor()
        
        self.state_history = deque(maxlen=10)
        self.step_count = 0
        self.success_threshold = INITIAL_SUCCESS_THRESHOLD
        self.synergy_threshold = INITIAL_SYNERGY_THRESHOLD
        
        # Store coalition history for Phase 9
        self.coalition_history: List[Dict] = []
        
        self._initialize_concepts()
        self.coalition_field = CoalitionEcologyField()
    
    def _initialize_concepts(self):
        all_dims = list(range(LATENT_DIM))
        
        for i in range(self.n_concepts):
            start_idx = (i * SUBSPACE_SIZE) % (LATENT_DIM - SUBSPACE_SIZE + 1)
            subspace_indices = list(range(start_idx, min(start_idx + SUBSPACE_SIZE, LATENT_DIM)))
            
            if len(subspace_indices) < SUBSPACE_SIZE:
                additional = list(set(all_dims) - set(subspace_indices))[:SUBSPACE_SIZE - len(subspace_indices)]
                subspace_indices.extend(additional)
            
            phase = np.random.uniform(0, 2 * np.pi)
            frequency = np.random.uniform(0.8, 1.2)
            
            concept = EnhancedConstraintConcept(
                id=i,
                subspace_indices=sorted(subspace_indices),
                phase=phase,
                frequency=frequency
            )
            self.concepts.append(concept)
    
    def _update_thresholds(self):
        progress = min(1.0, self.step_count / 600)
        self.success_threshold = INITIAL_SUCCESS_THRESHOLD + progress * (TARGET_SUCCESS_THRESHOLD - INITIAL_SUCCESS_THRESHOLD)
        self.synergy_threshold = INITIAL_SYNERGY_THRESHOLD + progress * (TARGET_SYNERGY_THRESHOLD - INITIAL_SYNERGY_THRESHOLD)
        
        if self.coalition_field:
            self.coalition_field.synergy_threshold = self.synergy_threshold
    
    def get_coalition_history(self) -> List[Dict]:
        """Return coalition history for Phase 9."""
        return self.coalition_history
    
    def step(self, current_state: np.ndarray, next_state: np.ndarray = None) -> Dict:
        if np.linalg.norm(current_state) > 0:
            current_state = current_state / np.linalg.norm(current_state)
        
        if next_state is None:
            next_state = current_state
        elif np.linalg.norm(next_state) > 0:
            next_state = next_state / np.linalg.norm(next_state)
        
        self.state_history.append(current_state)
        history = list(self.state_history)
        
        # Constraints
        constraints = []
        for c in self.concepts:
            constraint = c.compute_constraint(history)
            constraints.append(constraint)
        
        # Adaptive migration
        if USE_ADAPTIVE_SUBSPACE:
            for c in self.concepts:
                c.migrate_subspace(self.step_count)
        
        # Local coherence
        coherence = self.synchronization.update(self.concepts)
        
        # Update thresholds
        self._update_thresholds()
        
        # Coalitions
        coalitions = []
        avg_synergy = 0.0
        coalition_success_rate = 0.0
        
        if self.coalition_field:
            coalitions, avg_synergy, coalition_success_rate = self.coalition_field.update(
                self.concepts, next_state, constraints, self.step_count, self.success_threshold
            )
            
            # Store coalition history for Phase 9 (with STABLE IDs)
            for c in coalitions:
                self.coalition_history.append(c.to_dict())
        
        # Competition
        if coalitions and coalitions[0].full_prediction is not None:
            pred_error = 1.0 - cosine_sim(coalitions[0].full_prediction, next_state)
        else:
            pred_error = 0.5
        
        self.concepts = self.competition.update(self.concepts, current_state, pred_error)
        
        # Diversity contributions
        for c in self.concepts:
            overlaps = []
            for other in self.concepts:
                if other.id != c.id:
                    if len(c.subspace_indices) > 0:
                        overlap = len(set(c.subspace_indices) & set(other.subspace_indices)) / max(len(c.subspace_indices), 1)
                        overlaps.append(overlap)
            c.diversity_contribution = 1.0 - (np.mean(overlaps) if overlaps else 0.5)
        
        # Metrics
        cdi = self.metrics.compute_cdi(self.concepts)
        cii = self.metrics.compute_cii(coalitions)
        
        self.step_count += 1
        
        total_success = sum(c.success_count for c in self.concepts)
        total_failures = sum(c.failure_count for c in self.concepts)
        
        dominant_ratio = max(c.activation for c in self.concepts) / (sum(c.activation for c in self.concepts) + 1e-8)
        avg_concept_success = np.mean([c.get_success_rate() for c in self.concepts])
        avg_fatigue = np.mean([c.fatigue for c in self.concepts])
        
        metrics = {
            'step': self.step_count,
            'n_concepts': len(self.concepts),
            'active_concepts': sum(1 for c in self.concepts if c.activation > 0.02),
            'n_coalitions': len(coalitions),
            'cdi': cdi,
            'cii': cii,
            'dominant_ratio': dominant_ratio,
            'avg_synergy': avg_synergy,
            'coalition_success_rate': coalition_success_rate,
            'phase_coherence': coherence,
            'total_successes': total_success,
            'total_failures': total_failures,
            'avg_concept_success_rate': avg_concept_success,
            'avg_subspace_size': np.mean([c.get_current_subspace_size() for c in self.concepts]),
            'success_threshold': self.success_threshold,
            'synergy_threshold': self.synergy_threshold,
            'avg_fatigue': avg_fatigue
        }
        
        is_healthy, failures = self.health_monitor.check_health(metrics)
        metrics['is_healthy'] = is_healthy
        metrics['failure_modes'] = failures
        
        return metrics
    
    def get_health(self) -> Dict:
        return self.metrics.get_health()
    
    def get_system_health(self) -> Dict:
        return self.health_monitor.get_summary()


# ============================================================================
# VALIDATION
# ============================================================================

def generate_test_data(n_steps: int = 500, dim: int = LATENT_DIM):
    data = []
    current = np.random.randn(dim)
    current = current / (np.linalg.norm(current) + 1e-8)
    
    patterns = []
    for _ in range(5):
        pattern = np.random.randn(dim)
        pattern = pattern / (np.linalg.norm(pattern) + 1e-8)
        patterns.append(pattern)
    
    pattern_idx = 0
    pattern_duration = 0
    
    for step in range(n_steps):
        if pattern_duration <= 0:
            pattern_idx = (pattern_idx + 1) % len(patterns)
            pattern_duration = np.random.randint(30, 80)
        
        direction = patterns[pattern_idx] - current
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        current = current + 0.08 * direction + 0.03 * np.random.randn(dim)
        current = current / (np.linalg.norm(current) + 1e-8)
        
        pattern_duration -= 1
        data.append(current.copy())
    
    return data


def run_validation(n_steps: int = 500):
    print("\n" + "█"*80)
    print("PHASE 8: COMPLETE COGNITIVE FIELD ARCHITECTURE")
    print("FIXED: Stable coalition IDs based on member composition")
    print("█"*80)
    
    print(f"\nCONFIGURATION:")
    print(f"  • Coalition energy budget: {COALITION_ENERGY_BUDGET}")
    print(f"  • Synergy threshold: {SYNERGY_THRESHOLD:.3f}")
    print(f"  • Coalition IDs: STABLE (based on member hashing)")
    print(f"  • Local coherence mode: ENABLED")
    print(f"  • Coalition hierarchy: {ENABLE_COALITION_HIERARCHY}")
    
    system = Phase8System(n_concepts=N_CONCEPTS)
    test_data = generate_test_data(n_steps=n_steps + 20)
    
    print("\n" + "="*110)
    print(f"{'Step':<8} {'Active':<8} {'Coalitions':<10} {'CDI':<8} {'CII':<8} {'DomRatio':<10} {'SuccRate':<10} {'Fatigue':<10} {'Unique IDs':<12}")
    print("-"*110)
    
    unique_coalition_ids = set()
    
    for step in range(n_steps):
        current = test_data[step]
        next_state = test_data[step + 1] if step + 1 < len(test_data) else current
        
        metrics = system.step(current, next_state)
        
        # Track unique coalition IDs from history
        for c in system.coalition_history[-metrics['n_coalitions']:]:
            if c:
                unique_coalition_ids.add(c.get('id', -1))
        
        if step % 50 == 0 and step > 0:
            print(f"{step:<8} {metrics['active_concepts']:<8} "
                  f"{metrics['n_coalitions']:<10} {metrics['cdi']:<8.3f} {metrics['cii']:<8.3f} "
                  f"{metrics['dominant_ratio']:<10.3f} {metrics['avg_concept_success_rate']:<10.3f} "
                  f"{metrics['avg_fatigue']:<10.3f} {len(unique_coalition_ids):<12}")
    
    print("\n" + "="*80)
    print("PHASE 8 RESULTS")
    print("="*80)
    print(f"Total unique coalition IDs: {len(unique_coalition_ids)}")
    print(f"Coalition history size: {len(system.coalition_history)}")
    
    # Check if IDs are repeating (good for Phase 9)
    from collections import Counter
    id_counts = Counter([c.get('id', -1) for c in system.coalition_history if c])
    repeating_ids = {cid: count for cid, count in id_counts.items() if count > 1}
    
    print(f"\nCoalition ID repetition:")
    print(f"  • Unique IDs with repetition: {len(repeating_ids)}")
    if repeating_ids:
        print(f"  • Example repeating IDs: {list(repeating_ids.keys())[:5]}")
        print(f"  • Max repetition count: {max(repeating_ids.values()) if repeating_ids else 0}")
    
    if len(repeating_ids) > 0:
        print("\n✅ SUCCESS: Coalition IDs are repeating!")
        print("   Phase 9 can now detect narratives from recurring coalition patterns.")
    else:
        print("\n⚠️ WARNING: No coalition ID repetition detected.")
        print("   Phase 9 needs repeating patterns to form narratives.")
    
    print("\n" + "="*80)
    print("Phase 8 Complete.")
    print("")
    print("'This is not small. This is something that will shape the future.'")
    print("="*80)
    
    return system, unique_coalition_ids, repeating_ids


if __name__ == "__main__":
    system, unique_ids, repeating_ids = run_validation(n_steps=500)