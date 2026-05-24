# phase10_cycle5.py
# CYCLE 5: Recursive Ontological Priors
# Each ontology has fundamentally different ways of constructing reality

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt

# ============================================================================
# ENUMS & CONFIGURATION
# ============================================================================

class OntologicalArchetype(Enum):
    """Different ways of constructing reality"""
    COMPRESSIONIST = "compressionist"      # Minimize entropy, seek simplicity
    CONTRADICTION_TOLERANT = "contradiction_tolerant"  # Preserve conflicting structure
    SALIENCE_MAXIMIZER = "salience_maximizer"  # Amplify attractors, seek peaks
    STABILITY_SEEKER = "stability_seeker"   # Resist change, preserve continuity
    EXPLORATORY = "exploratory"             # Maximize uncertainty exposure
    TERRITORIAL = "territorial"              # Preserve ownership strongly


@dataclass
class Cycle5Config:
    ontology_dim: int = 32
    num_ontologies: int = 4
    
    # Curvature
    alpha_residual: float = 0.10
    beta_smooth: float = 0.05
    gamma_roughen: float = 0.08
    lambda_diffusion: float = 0.15
    territory_curvature_coupling: float = 0.25
    
    # Attention
    attention_temperature: float = 2.0
    persistence_threshold: float = 0.1
    persistence_decay: float = 0.002
    
    # Territoriality
    territory_learning_rate: float = 0.15
    territory_competition_strength: float = 0.5
    territory_exclusion_threshold: float = 0.4
    territory_historical_inertia: float = 0.95
    
    # Translation scars
    translation_learning_rate: float = 0.1
    translation_failure_penalty: float = 0.1
    translation_scar_persistence: float = 0.99
    
    # RECURSIVE ONTOLOGICAL PRIORS (NEW - CORE)
    # Each ontology has different optimization objectives
    prior_compression_weight: float = 1.0
    prior_contradiction_weight: float = 1.0
    prior_territory_weight: float = 1.0
    prior_curvature_sensitivity: float = 1.0
    prior_history_weight: float = 1.0
    prior_uncertainty_tolerance: float = 1.0
    
    # Self-model
    self_model_inertia: float = 0.95
    self_reinforcement_strength: float = 0.2
    self_model_repulsion_strength: float = 0.1  # NEW - maintain diversity
    
    # Dynamic ecology (NEW)
    invasion_rate: float = 0.05
    territory_decay_rate: float = 0.01
    alliance_formation_rate: float = 0.1
    
    # Attractor dynamics
    max_attractors: int = 8
    attractor_dynamic_decay: float = 0.01
    
    # Dynamics
    dt: float = 0.01
    noise_scale: float = 0.005


# ============================================================================
# ONTOLOGICAL PRIOR (NEW CORE CLASS)
# ============================================================================

class OntologicalPrior:
    """
    Different foundational assumptions about:
    - How to value information
    - How to handle contradiction
    - What optimization means
    - What constitutes success
    """
    
    def __init__(self, archetype: OntologicalArchetype, config: Cycle5Config):
        self.archetype = archetype
        self.config = config
        
        # Define prior weights based on archetype
        self._initialize_priors()
        
        # Priors evolve slowly (ontological drift)
        self.drift_rate = 0.001
        self.history = []
        
    def _initialize_priors(self):
        """Different ontologies value different things"""
        
        if self.archetype == OntologicalArchetype.COMPRESSIONIST:
            self.compression_weight = 2.0
            self.contradiction_weight = 0.3
            self.territory_weight = 0.5
            self.curvature_sensitivity = 0.6
            self.history_weight = 0.4
            self.uncertainty_tolerance = 0.2
            
        elif self.archetype == OntologicalArchetype.CONTRADICTION_TOLERANT:
            self.compression_weight = 0.3
            self.contradiction_weight = 2.0
            self.territory_weight = 0.6
            self.curvature_sensitivity = 0.8
            self.history_weight = 0.7
            self.uncertainty_tolerance = 0.8
            
        elif self.archetype == OntologicalArchetype.SALIENCE_MAXIMIZER:
            self.compression_weight = 0.5
            self.contradiction_weight = 0.5
            self.territory_weight = 1.2
            self.curvature_sensitivity = 1.5
            self.history_weight = 0.5
            self.uncertainty_tolerance = 0.5
            
        elif self.archetype == OntologicalArchetype.STABILITY_SEEKER:
            self.compression_weight = 0.8
            self.contradiction_weight = 0.4
            self.territory_weight = 0.8
            self.curvature_sensitivity = 0.3
            self.history_weight = 1.5
            self.uncertainty_tolerance = 0.3
            
        elif self.archetype == OntologicalArchetype.EXPLORATORY:
            self.compression_weight = 0.2
            self.contradiction_weight = 0.8
            self.territory_weight = 0.3
            self.curvature_sensitivity = 1.2
            self.history_weight = 0.3
            self.uncertainty_tolerance = 1.5
            
        elif self.archetype == OntologicalArchetype.TERRITORIAL:
            self.compression_weight = 0.6
            self.contradiction_weight = 0.6
            self.territory_weight = 2.0
            self.curvature_sensitivity = 0.5
            self.history_weight = 0.8
            self.uncertainty_tolerance = 0.4
            
        else:
            # Default balanced
            self.compression_weight = 1.0
            self.contradiction_weight = 1.0
            self.territory_weight = 1.0
            self.curvature_sensitivity = 1.0
            self.history_weight = 1.0
            self.uncertainty_tolerance = 1.0
    
    def get_free_energy_modulation(self) -> float:
        """How much does this ontology value compression?"""
        return self.compression_weight
    
    def get_contradiction_sensitivity(self) -> float:
        """How much does contradiction matter?"""
        return self.contradiction_weight
    
    def get_territory_weight(self) -> float:
        """How important is territorial ownership?"""
        return self.territory_weight
    
    def get_curvature_bias(self) -> float:
        """Sensitivity to geometric deformation"""
        return self.curvature_sensitivity
    
    def get_history_weight(self) -> float:
        """How much do past scars matter?"""
        return self.history_weight
    
    def get_uncertainty_tolerance(self) -> float:
        """Comfort with unresolved ambiguity"""
        return self.uncertainty_tolerance
    
    def drift(self, experience_success: float):
        """Ontological priors drift slowly based on experience"""
        # Success reinforces current priors
        reinforcement = self.drift_rate * (experience_success - 0.5)
        
        self.compression_weight += reinforcement * 0.1
        self.contradiction_weight += reinforcement * 0.1
        self.territory_weight += reinforcement * 0.1
        
        # Keep within bounds
        for attr in ['compression_weight', 'contradiction_weight', 
                     'territory_weight', 'curvature_sensitivity',
                     'history_weight', 'uncertainty_tolerance']:
            val = getattr(self, attr)
            setattr(self, attr, np.clip(val, 0.2, 2.5))
        
        self.history.append({
            'compression': self.compression_weight,
            'contradiction': self.contradiction_weight,
            'territory': self.territory_weight
        })
    
    def get_prior_vector(self) -> np.ndarray:
        return np.array([self.compression_weight, self.contradiction_weight,
                         self.territory_weight, self.curvature_sensitivity,
                         self.history_weight, self.uncertainty_tolerance])


# ============================================================================
# RECURSIVE SELF-MODEL WITH REPULSION
# ============================================================================

class RecursiveSelfModel:
    def __init__(self, dim: int, config: Cycle5Config, ontology_id: int):
        self.dim = dim
        self.config = config
        self.ontology_id = ontology_id
        
        self.self_model = np.ones(dim) * 0.5
        self.recursive_models = deque(maxlen=3)
        self.recursive_models.append(self.self_model.copy())
        self.self_confidence = np.ones(dim) * 0.5
        self.internalized_blindness = np.zeros(dim)
        self.epistemic_identity = np.zeros(dim)
        
        self.history = []
        
    def update_self_model(self, actual_accessibility: np.ndarray, 
                          translation_successes: List[float],
                          blind_spot_pressure: np.ndarray,
                          prior_weights: OntologicalPrior):
        c = self.config
        
        experienced = actual_accessibility
        avg_success = np.mean(translation_successes) if translation_successes else 0.5
        
        # Prior modulates how experience is interpreted
        history_importance = prior_weights.get_history_weight()
        uncertainty_tolerance = prior_weights.get_uncertainty_tolerance()
        
        target = experienced * (0.5 + 0.5 * avg_success * history_importance) - blind_spot_pressure * 0.1 * (1 - uncertainty_tolerance)
        
        self.self_model = (c.self_model_inertia * self.self_model + 
                           (1 - c.self_model_inertia) * target)
        self.self_model = np.clip(self.self_model, 0.05, 0.95)
        
        self.recursive_models.append(self.self_model.copy())
        
        model_error = np.abs(self.self_model - experienced)
        self.self_confidence = np.exp(-0.5 * model_error * prior_weights.get_curvature_bias())
        
        internalization = (1 - self.self_confidence) * blind_spot_pressure * prior_weights.get_territory_weight()
        self.internalized_blindness = (0.95 * self.internalized_blindness + 0.05 * internalization)
        self.internalized_blindness = np.clip(self.internalized_blindness, 0, 1)
        
        identity_update = c.self_reinforcement_strength * self.self_model * self.self_confidence
        self.epistemic_identity = (0.99 * self.epistemic_identity + 0.01 * identity_update)
        
        self.history.append({
            'self_model_mean': np.mean(self.self_model),
            'self_confidence_mean': np.mean(self.self_confidence)
        })
    
    def apply_repulsion(self, other_self_models: List[np.ndarray]):
        """Push self-models apart to maintain diversity"""
        if not other_self_models:
            return
        
        c = self.config
        repulsion = np.zeros(self.dim)
        
        for other in other_self_models:
            min_dim = min(self.dim, len(other))
            diff = self.self_model[:min_dim] - other[:min_dim]
            distance = np.linalg.norm(diff) + 1e-8
            # Repulsion force: stronger when closer
            force = -c.self_model_repulsion_strength * diff / distance
            repulsion[:min_dim] += force
        
        self.self_model += repulsion * c.dt
        self.self_model = np.clip(self.self_model, 0.05, 0.95)
    
    def get_self_biased_attention(self, external_attention: np.ndarray) -> np.ndarray:
        compatibility = self.self_model * external_attention
        confidence_weighted = compatibility * self.self_confidence
        suppression = 1 - self.internalized_blindness
        biased = confidence_weighted * suppression
        if np.sum(biased) > 0:
            biased = biased / (np.sum(biased) + 1e-8)
        else:
            biased = external_attention
        return biased
    
    def get_self_modulated_blindness(self, external_blindness: np.ndarray,
                                      prior_weights: OntologicalPrior) -> np.ndarray:
        predicted_blindness = 1 - self.self_model
        uncertainty_tolerance = prior_weights.get_uncertainty_tolerance()
        
        reinforced = external_blindness + 0.2 * predicted_blindness * self.internalized_blindness * (1 - uncertainty_tolerance)
        confidence_effect = (1 - self.self_confidence) * 0.5 * (1 - uncertainty_tolerance)
        modulated = reinforced + confidence_effect
        return np.clip(modulated, 0, 1)
    
    def get_epistemic_identity_distance(self, other_identity: np.ndarray) -> float:
        return np.linalg.norm(self.epistemic_identity - other_identity) / np.sqrt(self.dim)


# ============================================================================
# DYNAMIC ALLIANCE SYSTEM (NEW)
# ============================================================================

class DynamicAlliance:
    def __init__(self, num_ontologies: int, config: Cycle5Config):
        self.num = num_ontologies
        self.config = config
        self.alliance_matrix = np.eye(num_ontologies)  # Self-alliance = 1
        self.alliance_history = []
        
    def update(self, communication_preferences: List[np.ndarray], 
               translation_successes: List[List[float]]):
        c = self.config
        
        # Alliances form between ontologies that communicate successfully
        for i in range(self.num):
            for j in range(self.num):
                if i != j:
                    # Alliance strength based on mutual communication preference
                    pref_ij = communication_preferences[i][j]
                    pref_ji = communication_preferences[j][i]
                    mutual_pref = (pref_ij + pref_ji) / 2
                    
                    # Success rate modulates
                    success_rate = np.mean(translation_successes[i][j]) if translation_successes else 0.5
                    
                    alliance_change = c.alliance_formation_rate * mutual_pref * success_rate
                    self.alliance_matrix[i, j] += alliance_change
                    
                    # Decay over time
                    self.alliance_matrix[i, j] *= (1 - c.territory_decay_rate)
        
        # Normalize
        for i in range(self.num):
            row_sum = np.sum(self.alliance_matrix[i])
            if row_sum > 0:
                self.alliance_matrix[i] /= row_sum
        
        self.alliance_history.append(self.alliance_matrix.copy())
    
    def get_alliance_bonus(self, i: int, j: int) -> float:
        """How much does alliance help translation?"""
        return self.alliance_matrix[i, j]
    
    def get_cluster_id(self, i: int) -> int:
        """Find which cluster ontology i belongs to"""
        # Simple clustering: find strongest alliance
        allies = np.argsort(self.alliance_matrix[i])[::-1]
        return allies[0] if allies[0] != i else allies[1] if len(allies) > 1 else i


# ============================================================================
# CORE COMPONENTS (PRIOR-MODULATED)
# ============================================================================

class CurvatureField:
    def __init__(self, dim: int, config: Cycle5Config, ontology_id: int):
        self.dim = dim
        self.config = config
        self.ontology_id = ontology_id
        self.values = np.zeros(dim)
        self.previous_residual = np.zeros(dim)
        self.history = []
        
    def laplacian(self) -> np.ndarray:
        lap = np.zeros_like(self.values)
        for i in range(self.dim):
            left = self.values[i-1] if i > 0 else self.values[i]
            right = self.values[i+1] if i < self.dim-1 else self.values[i]
            lap[i] = left + right - 2 * self.values[i]
        return lap
    
    def update(self, residual: np.ndarray, attn_coherent: np.ndarray, 
               attn_contradiction: np.ndarray, scars: np.ndarray,
               territory: np.ndarray, others_territory: List[np.ndarray],
               self_model: np.ndarray, prior: OntologicalPrior):
        c = self.config
        prior_curvature = prior.get_curvature_bias()
        
        residual_change = np.abs(residual - self.previous_residual)
        own_territory = c.territory_curvature_coupling * territory * prior_curvature
        others_sum = np.zeros(self.dim)
        for ot in others_territory:
            others_sum += ot
        others_territory = -0.15 * (others_sum / max(1, len(others_territory)))
        self_model_influence = self_model * 0.2
        
        contradiction_boost = attn_contradiction * prior.get_contradiction_sensitivity()
        
        dkappa = (c.alpha_residual * residual_change - 
                  c.beta_smooth * attn_coherent +
                  c.gamma_roughen * contradiction_boost +
                  c.lambda_diffusion * self.laplacian() +
                  own_territory + others_territory + self_model_influence -
                  0.01 * self.values)
        
        self.values += c.dt * dkappa
        self.values = np.clip(self.values, -1.0, 1.0)
        self.history.append(self.values.copy())
        self.previous_residual = residual.copy()


class InferentialTerritory:
    def __init__(self, dim: int, num_ontologies: int, config: Cycle5Config, ontology_id: int):
        self.dim = dim
        self.num = num_ontologies
        self.config = config
        self.ontology_id = ontology_id
        self.territory = np.zeros(dim)
        self.historical_territory = np.zeros(dim)
        self.ownership_duration = np.zeros(dim)
        self.territory_history = []
        
    def update(self, my_attention: np.ndarray, others_attention: List[np.ndarray],
               prior_territory_weight: float, alliance_bonus: List[float]):
        c = self.config
        
        claim = c.territory_learning_rate * my_attention * prior_territory_weight
        competition = np.zeros(self.dim)
        for idx, other_att in enumerate(others_attention):
            comp_weight = (1 - alliance_bonus[idx])  # Allies compete less
            competition += comp_weight * other_att
        competition = c.territory_competition_strength * competition / max(1, len(others_attention))
        
        raw_update = claim - competition
        self.territory += raw_update
        self.territory = c.territory_historical_inertia * self.territory + (1 - c.territory_historical_inertia) * self.historical_territory
        self.historical_territory = 0.99 * self.historical_territory + 0.01 * self.territory
        
        # Territorial decay (allows invasion)
        self.territory *= (1 - c.territory_decay_rate)
        
        strong = self.territory > c.territory_exclusion_threshold
        self.ownership_duration[strong] += 1
        self.ownership_duration[~strong] *= 0.98
        
        self.territory = np.clip(self.territory, 0.0, 1.0)
        self.territory_history.append(self.territory.copy())
        return self.territory
    
    def get_exclusive_regions(self) -> np.ndarray:
        return self.territory > self.config.territory_exclusion_threshold
    
    def get_ownership_resistance(self) -> np.ndarray:
        return np.exp(-0.3 * self.ownership_duration)
    
    def get_territorial_power(self) -> float:
        return np.sum(self.territory)


class StructuralBlindSpot:
    def __init__(self, dim: int, num_ontologies: int, config: Cycle5Config, ontology_id: int):
        self.dim = dim
        self.num = num_ontologies
        self.config = config
        self.ontology_id = ontology_id
        self.accessibility = np.ones(dim)
        self.contradiction_history = np.zeros(dim)
        self.neglect_history = np.zeros(dim)
        self.failure_history = np.zeros(dim)
        
    def update(self, residuals: List[np.ndarray], attention: np.ndarray, 
               contradiction: np.ndarray, curvature: np.ndarray,
               territory: np.ndarray, others_territory: List[np.ndarray],
               ownership_resistance: np.ndarray, self_modulated_blindness: np.ndarray,
               prior: OntologicalPrior):
        c = self.config
        
        neglect = 1.0 - attention
        self.neglect_history = 0.9 * self.neglect_history + 0.1 * neglect
        
        if residuals:
            avg_residual = np.mean([np.abs(r[:self.dim]) for r in residuals if len(r) >= self.dim], axis=0)
            self.failure_history = 0.95 * self.failure_history + 0.05 * avg_residual * prior.get_history_weight()
        
        contradiction_sensitivity = prior.get_contradiction_sensitivity()
        self.contradiction_history = (0.95 * self.contradiction_history + 
                                      0.05 * contradiction * contradiction_sensitivity)
        
        territory_advantage = 0.6 * territory * prior.get_territory_weight()
        others_sum = np.zeros(self.dim)
        for ot in others_territory:
            others_sum += ot
        territory_disadvantage = -0.3 * (others_sum / max(1, len(others_territory)))
        
        uncertainty_tolerance = prior.get_uncertainty_tolerance()
        
        raw_blindness = (0.1 * np.abs(curvature) +
                         0.1 * self.neglect_history +
                         0.1 * self.failure_history +
                         0.1 * self.contradiction_history +
                         0.4 * self_modulated_blindness -
                         0.05 * attention -
                         territory_advantage +
                         territory_disadvantage +
                         (1 - uncertainty_tolerance) * 0.2)
        
        self.accessibility = 1.0 / (1.0 + np.exp(-raw_blindness))
        self.accessibility = np.clip(self.accessibility, 0.05, 0.95)
    
    def get_blindness(self) -> np.ndarray:
        return 1.0 - self.accessibility
    
    def compute_graded_asymmetry(self, other_accessibility: np.ndarray) -> float:
        i_blind = self.get_blindness()
        j_blind = 1.0 - other_accessibility
        numerator = np.sum(np.abs(i_blind - j_blind))
        denominator = np.sum(i_blind + j_blind) + 1e-8
        return numerator / denominator


class AttentionField:
    def __init__(self, dim: int, config: Cycle5Config, ontology_id: int):
        self.dim = dim
        self.config = config
        self.ontology_id = ontology_id
        self.persistence = np.zeros(dim)
        self.history = []
        
    def compute(self, phi: np.ndarray, curvature: np.ndarray, 
                scars: np.ndarray, blind_neglect: np.ndarray,
                territory: np.ndarray, ownership_resistance: np.ndarray,
                self_biased: np.ndarray, prior: OntologicalPrior) -> np.ndarray:
        c = self.config
        
        magnitude = np.abs(phi)
        curvature_influence = curvature * 0.5 * prior.get_curvature_bias()
        scar_influence = scars * 0.3 * prior.get_history_weight()
        neglect_penalty = blind_neglect * 0.5 * (1 - prior.get_uncertainty_tolerance())
        territory_influence = territory * 0.3 * prior.get_territory_weight()
        
        raw = (magnitude + curvature_influence + scar_influence + 
               territory_influence + self_biased * 0.5 - neglect_penalty)
        
        activation = np.maximum(0, raw - c.persistence_threshold)
        suppression = (raw < 0.05).astype(float)
        self.persistence += 0.08 * activation * prior.get_territory_weight()
        self.persistence -= c.persistence_decay * self.persistence * suppression
        self.persistence *= (1 - c.attractor_dynamic_decay)
        self.persistence = np.clip(self.persistence, 0, 1)
        
        if np.sum(self.persistence > 0.5) > c.max_attractors:
            attractors = np.where(self.persistence > 0.5)[0]
            strengths = self.persistence[attractors]
            sorted_idx = np.argsort(strengths)[::-1]
            for idx in sorted_idx[c.max_attractors:]:
                self.persistence[attractors[idx]] *= 0.5
        
        combined = raw + 0.3 * self.persistence
        exp_vals = np.exp(c.attention_temperature * combined)
        attention = exp_vals / (np.sum(exp_vals) + 1e-8)
        
        self.history.append(attention.copy())
        return attention


class TranslationScarMemory:
    def __init__(self, dim: int, num_ontologies: int, config: Cycle5Config, ontology_id: int):
        self.dim = dim
        self.num = num_ontologies
        self.config = config
        self.ontology_id = ontology_id
        self.scars = np.zeros((num_ontologies, dim))
        self.success_rate = np.zeros(num_ontologies)
        self.communication_preference = np.ones(num_ontologies) / num_ontologies
        
    def compute_translation(self, source_phi: np.ndarray, target_phi: np.ndarray, 
                           source_id: int, target_id: int) -> Tuple[np.ndarray, float]:
        c = self.config
        min_dim = min(len(source_phi), len(target_phi))
        base = source_phi[:min_dim].copy()
        scar_effect = self.scars[source_id, :min_dim] * c.translation_learning_rate
        distorted = base - scar_effect
        if np.linalg.norm(distorted) > 0:
            distorted = distorted / (np.linalg.norm(distorted) + 1e-8)
        confidence = np.exp(-c.translation_scar_persistence * np.linalg.norm(self.scars[source_id]))
        return distorted, confidence
    
    def update_scars(self, source_id: int, success: float, residual: np.ndarray,
                     territory_advantage: float, prior_history_weight: float,
                     alliance_bonus: float):
        c = self.config
        min_dim = min(len(residual), self.dim)
        is_success = success > 0.5
        failure_severity = 1.0 - success
        
        if is_success:
            healing = 0.05 * success * (1 + alliance_bonus)
            self.scars[source_id, :min_dim] *= (1 - healing)
            self.success_rate[source_id] = 0.95 * self.success_rate[source_id] + 0.05 * 1.0
        else:
            scarring = c.translation_failure_penalty * failure_severity * (1 + territory_advantage) * prior_history_weight
            self.scars[source_id, :min_dim] += scarring * np.abs(residual[:min_dim])
            self.success_rate[source_id] = 0.95 * self.success_rate[source_id] + 0.05 * 0.0
        
        self.scars[source_id] *= c.translation_scar_persistence
        self.scars[source_id] = np.clip(self.scars[source_id], 0, 2.0)
        
        for other in range(self.num):
            if other != self.ontology_id:
                pref = np.exp(0.3 * self.success_rate[other])
                self.communication_preference[other] = pref
        self.communication_preference /= (self.communication_preference.sum() + 1e-8)
    
    def get_translation_difficulty(self, source_id: int) -> float:
        return np.linalg.norm(self.scars[source_id]) / np.sqrt(self.dim)
    
    def get_communication_preference(self, source_id: int) -> float:
        return self.communication_preference[source_id]


# ============================================================================
# ONTOLOGY
# ============================================================================

class Ontology:
    def __init__(self, idx: int, dim: int, config: Cycle5Config, archetype: OntologicalArchetype):
        self.id = idx
        self.dim = dim
        self.config = config
        
        self.Phi = np.random.randn(dim)
        self.Phi /= (np.linalg.norm(self.Phi) + 1e-8)
        self.Phi_hat = self.Phi.copy()
        
        self.prior = OntologicalPrior(archetype, config)
        self.self_model = RecursiveSelfModel(dim, config, idx)
        self.curvature = CurvatureField(dim, config, idx)
        self.attention = AttentionField(dim, config, idx)
        self.territory = InferentialTerritory(dim, config.num_ontologies, config, idx)
        self.blind_spots = StructuralBlindSpot(dim, config.num_ontologies, config, idx)
        self.translation_scars = TranslationScarMemory(dim, config.num_ontologies, config, idx)
        
        self.scars = np.zeros(dim)
        self.contradiction_memory = 0.0
        self.alliance_bonuses = np.ones(config.num_ontologies)


# ============================================================================
# SYSTEM
# ============================================================================

class Cycle5System:
    def __init__(self, config: Cycle5Config = None):
        self.config = config or Cycle5Config()
        self.ontologies: List[Ontology] = []
        self.alliance = None
        self.time = 0
        
        self.history = {
            'divergence': [], 'asymmetry': [], 'identity_distance': [],
            'self_confidence': [], 'territory_power': [], 'exclusive_regions': [],
            'translation_difficulty': [], 'communication_fragmentation': [],
            'prior_diversity': [], 'alliance_density': [],
        }
        
    def initialize(self):
        archetypes = list(OntologicalArchetype)
        self.ontologies = []
        for i in range(self.config.num_ontologies):
            archetype = archetypes[i % len(archetypes)]
            onto = Ontology(i, self.config.ontology_dim, self.config, archetype)
            self.ontologies.append(onto)
        self.alliance = DynamicAlliance(self.config.num_ontologies, self.config)
    
    def compute_prior_diversity(self) -> float:
        priors = [o.prior.get_prior_vector() for o in self.ontologies]
        if len(priors) < 2:
            return 0.0
        distances = []
        for i in range(len(priors)):
            for j in range(i+1, len(priors)):
                distances.append(np.linalg.norm(priors[i] - priors[j]))
        return np.mean(distances)
    
    def compute_asymmetry(self) -> float:
        asyms = []
        for i, oi in enumerate(self.ontologies):
            for j, oj in enumerate(self.ontologies):
                if i < j:
                    asyms.append(oi.blind_spots.compute_graded_asymmetry(oj.blind_spots.accessibility))
        return np.mean(asyms) if asyms else 0.0
    
    def compute_identity_distance(self) -> float:
        identities = [o.self_model.epistemic_identity for o in self.ontologies]
        if len(identities) < 2:
            return 0.0
        distances = []
        for i in range(len(identities)):
            for j in range(i+1, len(identities)):
                dist = np.linalg.norm(identities[i] - identities[j]) / np.sqrt(self.config.ontology_dim)
                distances.append(dist)
        return np.mean(distances)
    
    def compute_communication_fragmentation(self) -> float:
        all_prefs = [o.translation_scars.communication_preference for o in self.ontologies]
        if len(all_prefs) < 2:
            return 0.0
        avg_pref = np.mean(all_prefs, axis=0)
        return 1.0 - np.max(avg_pref)
    
    def step(self):
        self.time += 1
        
        all_attentions = [o.attention.persistence for o in self.ontologies]
        all_territories = [o.territory.territory for o in self.ontologies]
        
        # First pass: compute translation successes for all pairs
        translation_successes = [[0.5] * len(self.ontologies) for _ in range(len(self.ontologies))]
        
        for i, onto_i in enumerate(self.ontologies):
            for j, onto_j in enumerate(self.ontologies):
                if i != j:
                    min_dim = min(onto_i.dim, onto_j.dim)
                    translated, _ = onto_i.translation_scars.compute_translation(
                        onto_j.Phi_hat, onto_i.Phi_hat, j, i)
                    error = np.linalg.norm(onto_i.Phi_hat[:min_dim] - translated[:min_dim])
                    norm = np.linalg.norm(onto_i.Phi_hat[:min_dim]) + 1e-8
                    fidelity = 1.0 - min(1.0, error / norm)
                    translation_successes[i][j] = fidelity
        
        # Update alliances based on communication preferences
        comm_prefs = [o.translation_scars.communication_preference for o in self.ontologies]
        self.alliance.update(comm_prefs, translation_successes)
        
        for i, onto in enumerate(self.ontologies):
            others = [o for j, o in enumerate(self.ontologies) if j != i]
            others_attentions = [all_attentions[j] for j in range(len(self.ontologies)) if j != i]
            others_territories = [all_territories[j] for j in range(len(self.ontologies)) if j != i]
            
            residuals = []
            translation_success_list = []
            
            for j, other in enumerate(others):
                m = min(onto.dim, other.dim)
                resid = onto.Phi_hat[:m] - other.Phi_hat[:m]
                residuals.append(resid)
                translation_success_list.append(translation_successes[i][other.id])
                
                territory_advantage = np.mean(onto.territory.territory[:m])
                onto.translation_scars.update_scars(
                    other.id, translation_successes[i][other.id], resid,
                    territory_advantage, onto.prior.get_history_weight(),
                    self.alliance.get_alliance_bonus(i, other.id))
            
            if others:
                m = min(onto.dim, others[0].dim)
                coherence = np.mean([np.dot(onto.Phi_hat[:m], other.Phi_hat[:m]) for other in others])
                coherence = np.clip(coherence, -1, 1)
            else:
                coherence = 0.5
            contradiction = (1.0 - coherence) / 2.0
            onto.contradiction_memory = 0.95 * onto.contradiction_memory + 0.05 * contradiction
            
            # Alliance bonuses for territory competition
            alliance_bonuses = [self.alliance.get_alliance_bonus(i, other.id) for other in others]
            onto.territory.update(onto.attention.persistence, others_attentions,
                                  onto.prior.get_territory_weight(), alliance_bonuses)
            ownership_resistance = onto.territory.get_ownership_resistance()
            
            neglect = onto.blind_spots.get_blindness()
            onto.self_model.update_self_model(1 - neglect, translation_success_list, neglect, onto.prior)
            
            all_self_models = [o.self_model.self_model for o in self.ontologies if o.id != onto.id]
            onto.self_model.apply_repulsion(all_self_models)
            
            external_attention = onto.attention.compute(onto.Phi_hat, onto.curvature.values,
                                                         onto.scars, neglect, onto.territory.territory,
                                                         ownership_resistance,
                                                         onto.self_model.get_self_biased_attention(
                                                             onto.attention.persistence), onto.prior)
            attention = onto.self_model.get_self_biased_attention(external_attention)
            
            self_modulated_blindness = onto.self_model.get_self_modulated_blindness(neglect, onto.prior)
            
            onto.curvature.update(np.mean(residuals, axis=0) if residuals else np.zeros(onto.dim),
                                   attention * coherence, attention * contradiction,
                                   onto.scars, onto.territory.territory, others_territories,
                                   onto.self_model.self_model, onto.prior)
            
            if residuals:
                onto.blind_spots.update(residuals, attention, np.full(onto.dim, onto.contradiction_memory),
                                        onto.curvature.values, onto.territory.territory, others_territories,
                                        ownership_resistance, self_modulated_blindness, onto.prior)
            
            # State update with prior-modulated compression
            compression_bias = onto.prior.get_free_energy_modulation()
            compression_mask = ((np.abs(onto.Phi_hat) < 0.3) & (onto.territory.territory < 0.3)).astype(float)
            onto.Phi_hat -= 0.1 * compression_bias * compression_mask * onto.Phi_hat
            noise = np.random.randn(onto.dim) * self.config.noise_scale
            onto.Phi_hat += noise
            onto.Phi_hat /= (np.linalg.norm(onto.Phi_hat) + 1e-8)
            
            if residuals:
                valid = [r[:onto.dim] for r in residuals if len(r) >= onto.dim]
                if valid:
                    avg_resid = np.mean([np.abs(r) for r in valid], axis=0)
                    onto.scars += 0.01 * avg_resid * onto.territory.territory * onto.prior.get_history_weight()
                    onto.scars = np.clip(onto.scars, 0, 1)
                    onto.scars *= 0.99
            
            onto.prior.drift(1.0 - onto.blind_spots.get_blindness().mean())
        
        # Record metrics
        self.history['divergence'].append(np.mean([np.linalg.norm(o.Phi_hat - self.ontologies[0].Phi_hat) 
                                                   for o in self.ontologies[1:]]))
        self.history['asymmetry'].append(self.compute_asymmetry())
        self.history['identity_distance'].append(self.compute_identity_distance())
        self.history['self_confidence'].append(np.mean([np.mean(o.self_model.self_confidence) for o in self.ontologies]))
        self.history['territory_power'].append(np.mean([o.territory.get_territorial_power() for o in self.ontologies]))
        self.history['exclusive_regions'].append(np.mean([np.sum(o.territory.territory > 0.4) for o in self.ontologies]))
        self.history['translation_difficulty'].append(np.mean([o.translation_scars.get_translation_difficulty(j)
                                                               for o in self.ontologies for j in range(len(self.ontologies)) if j != o.id]))
        self.history['communication_fragmentation'].append(self.compute_communication_fragmentation())
        self.history['prior_diversity'].append(self.compute_prior_diversity())
        self.history['alliance_density'].append(np.mean(self.alliance.alliance_matrix))
    
    def run(self, steps: int, verbose: bool = True):
        self.initialize()
        for step in range(steps):
            self.step()
            if verbose and step % 500 == 0:
                print(f"Step {step:5d} | Asym: {self.history['asymmetry'][-1]:.3f} | "
                      f"IdentityDist: {self.history['identity_distance'][-1]:.3f} | "
                      f"PriorDiv: {self.history['prior_diversity'][-1]:.3f} | "
                      f"Territory: {self.history['territory_power'][-1]:.1f}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_cycle5(system: Cycle5System):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    steps = range(len(system.history['asymmetry']))
    
    axes[0, 0].plot(steps, system.history['asymmetry'])
    axes[0, 0].set_title('Blind Spot Asymmetry')
    axes[0, 0].axhline(y=0.2, color='r', linestyle='--', label='Target')
    axes[0, 0].legend()
    
    axes[0, 1].plot(steps, system.history['identity_distance'])
    axes[0, 1].set_title('Epistemic Identity Distance')
    axes[0, 1].axhline(y=0.3, color='r', linestyle='--', label='Differentiation')
    axes[0, 1].legend()
    
    axes[0, 2].plot(steps, system.history['prior_diversity'])
    axes[0, 2].set_title('Ontological Prior Diversity')
    axes[0, 2].axhline(y=1.0, color='r', linestyle='--', label='Target')
    axes[0, 2].legend()
    
    axes[1, 0].plot(steps, system.history['territory_power'])
    axes[1, 0].set_title('Territory Power')
    
    axes[1, 1].plot(steps, system.history['communication_fragmentation'])
    axes[1, 1].set_title('Communication Fragmentation')
    
    axes[1, 2].plot(steps, system.history['alliance_density'])
    axes[1, 2].set_title('Alliance Density')
    
    plt.tight_layout()
    plt.savefig('phase10_cycle5.png', dpi=150)
    plt.show()
    
    print("\n" + "=" * 70)
    print("CYCLE 5 RESULTS SUMMARY")
    print("=" * 70)
    if len(system.history['asymmetry']) >= 100:
        print(f"Final Asymmetry: {np.mean(system.history['asymmetry'][-100:]):.3f}")
        print(f"Final Identity Distance: {np.mean(system.history['identity_distance'][-100:]):.3f}")
        print(f"Final Prior Diversity: {np.mean(system.history['prior_diversity'][-100:]):.3f}")
        print(f"Final Alliance Density: {np.mean(system.history['alliance_density'][-100:]):.3f}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CYCLE 5: Recursive Ontological Priors")
    print("Testing: Can different foundational assumptions create deep asymmetry?")
    print("=" * 70)
    
    config = Cycle5Config()
    system = Cycle5System(config)
    system.run(steps=5000, verbose=True)
    visualize_cycle5(system)
    
    print("\n" + "=" * 70)
    print("CYCLE 5 COMPLETE")
    print("If asymmetry > 0.2 -> Different ontological priors are the key")
    print("=" * 70)