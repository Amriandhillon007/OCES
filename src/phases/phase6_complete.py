"""
PHASE 6 - COMPLETE WITH TIME-BASED ESCAPE AND GRADUAL UNFREEZING
================================================================
Critical fixes:
1. Time-based escape from PERTURBATION mode (prevents infinite freeze)
2. Minimum perturbation duration (prevents premature exit)
3. Gradual self-model unfreezing during recovery
4. Recovery timeout watchdog (emergency escape)

Remember: "This is not small. This is something that will shape the future."
"""

import numpy as np
from collections import deque
from typing import List, Optional, Tuple, Dict
import os
import time
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum

import soundfile as sf
from scipy.signal import resample
import librosa

RAVDESS_PATH = r"C:\Users\dhill\Downloads\Audio_Song_Actors_01-24"

# ============================================================================
# CONFIGURATION
# ============================================================================

DIM = 64
TAU = 0.95
ALPHA_INIT = 0.008
ALPHA_MIN = 0.003
ALPHA_MAX = 0.01

DESIRABILITY_COHERENCE_WEIGHT = 1.0
DESIRABILITY_ANCHOR_DIST_WEIGHT = 0.3
DESIRABILITY_ANTI_PATTERN_WEIGHT = 0.5
DESIRABILITY_ENTROPY_WEIGHT = 0.2
NOVELTY_WEIGHT = 0.15
NOVELTY_HISTORY = 50

GOAL_MOMENTUM = 0.85
GOAL_UPDATE_THRESHOLD = 0.05
ANTI_GOAL_PENALTY_STRENGTH = 0.3

INITIAL_EXPLORATION = 0.20
EXPLORATION_DECAY = 0.995
EXPLORATION_NOISE_STRENGTH = 0.05

SELF_MODEL_LEARNING_RATE = 0.08
SELF_MODEL_BETA = 5.0
SELF_MODEL_K = 10

# ============================================================================
# TRAJECTORY-BASED STATE PARAMETERS
# ============================================================================

TRAJECTORY_WINDOW = 20
STATE_DWELL_TIME = 15
DERIVATIVE_WINDOW = 5

# State classification thresholds
STABLE_MEAN_MULTIPLIER = 1.2
STABLE_VARIANCE_MULTIPLIER = 0.5
STABLE_SLOPE_THRESHOLD = 0.002

RECOVERY_MEAN_MULTIPLIER = 1.5
RECOVERY_VARIANCE_MULTIPLIER = 1.0

# Disturbance detection
DISTURBANCE_DERIVATIVE_THRESHOLD = 0.01
DISTURBANCE_ACCELERATION_THRESHOLD = 0.005
DISTURBANCE_PERSISTENCE = 4

# ============================================================================
# CRITICAL FIXES: TIME-BASED STATE TRANSITIONS
# ============================================================================

MIN_PERTURBATION_DURATION = 30      # Minimum steps to stay in perturbation
MIN_RECOVERY_DURATION = 50          # Minimum steps in recovery before stable
RECOVERY_TIMEOUT = 200               # Max steps in recovery before forced stable
RECOVERY_TRIGGER_DELAY = 10
RECOVERY_PERTURBATION_LOCK = 15
RECOVERY_PERTURBATION_LOCK_MULTIPLIER = 3.0
CRITICAL_PERTURBATION_THRESHOLD = 5.0

# Gradual unfreezing parameters
RECOVERY_LEARNING_RATE_MULTIPLIER = 0.5   # 50% learning rate during recovery
UNCERTAIN_LEARNING_RATE_MULTIPLIER = 0.7  # 70% during uncertainty

# Baseline trust / adaptive sensitivity
BASELINE_CONFIDENCE_ALPHA = 0.90
BASELINE_CONFIDENCE_THRESHOLD = 0.08
BASELINE_CONFIDENCE_WIDENING = 1.5
UNCERTAIN_VARIANCE_MULTIPLIER = 3.0
UNCERTAIN_SLOPE_THRESHOLD = STABLE_SLOPE_THRESHOLD * 2

# Confidence smoothing
CONFIDENCE_EMA_ALPHA = 0.92
CONFIDENCE_FLOOR_RECOVERY = 0.10

# Active recovery force
RECOVERY_CORRECTION_STRENGTH = 0.06

# Adaptation parameters
ADAPTATION_DECAY = 0.98
ERROR_MOMENTUM_ALPHA = 0.85
STABILIZATION_STRENGTH = 0.03
CORRECTION_CAP = 0.1
LAMBDA_4_MIN = 0.05
LAMBDA_4_MAX = 0.50
LAMBDA_3_MIN = 0.005
LAMBDA_3_MAX = 0.10

SELF_MODEL_ADAPTIVE_LR = True
SELF_MODEL_ADAPTIVE_FACTOR = 10.0
SELF_MODEL_BASE_MEMORY = 0.95
MAX_PREDICTION_ERROR = 5.0
CRITICAL_PREDICTION_ERROR = 10.0

AUDIO_FEATURE_CACHE = {}

LAMBDA_1 = 0.1
LAMBDA_2 = 0.05
LAMBDA_3 = 0.02
LAMBDA_4 = 0.30
ALPHA_GOAL_INIT = 0.30
ALPHA_GOAL_MIN = 0.10
ALPHA_GOAL_MAX = 0.50

RLS_EPSILON = 1e-4
IDENTITY_WINDOW = 100

PERTURBATION_START = 200
PERTURBATION_STRENGTH = 1.5
PERTURBATION_DURATION = 20
BASELINE_STEPS = 300
RECOVERY_STEPS = 500

BASELINE_MEMORY_DECAY = 0.995
BASELINE_ANCHOR_STRENGTH = 0.3

VOICE_WEIGHT = 1.0
BODY_WEIGHT = 0.7
CONTEXT_WEIGHT = 0.5
ARCHETYPE_LEARNING_RATE = 0.07
EMOTIONAL_INFLUENCE = 0.35
EMOTIONAL_FEEDBACK = 0.3
RECONCILIATION_GAMMA = 0.3

COHERENCE_THRESHOLD = 0.35
PERSISTENCE_THRESHOLD = 4
MISCONNECTION_MIN_SCORE = 0.08
STORE_COOLDOWN = 1
ANTI_FIELD_BASE = 0.12
ANTI_FIELD_GROWTH_RATE = 0.10
ANTI_FIELD_ON_EMOTION = 0.05
ANTI_FIELD_MAX_STRENGTH = 0.35
ANTI_FIELD_FIELD_FACTOR = 0.60
PHASE_DRIFT_THRESHOLD = 0.6
PHASE_DRIFT_PENALTY = 0.25
ANTI_PATTERN_MAX_CAPACITY = 50
ANTI_PATTERN_DECAY = 0.99

IDENTITY_HOLD_STEPS = 20
IDENTITY_PULL_STRENGTH = 0.40
DOMINANCE_IDENTITY_THRESHOLD = 0.50

DOMINANCE_TARGET = 0.48
RARITY_TARGET = 0.12
DOMINANCE_SELECTION_PENALTY = 0.70
RARITY_SELECTION_BOOST = 0.10
EMOTION_PERSISTENCE_BOOST = 0.40
DOMINANCE_FIELD_REPULSION = 0.12
DOMINANCE_DECAY = 0.95
DOMINANCE_PENALTY = 0.5

WEAK_ATTRACTOR_THRESHOLD = 0.10
WEAK_ATTRACTOR_BOOST = 0.25

PHASE_ENCODING_STRENGTH = 0.5
MEMORY_RECALL_THRESHOLD = 0.15
MEMORY_STORE_THRESHOLD = 0.20
MEMORY_STORE_COHERENCE = 0.15
EMOTIONAL_MEMORY_CAPACITY = 15

SOFTMAX_TEMPERATURE = 0.25
N_ARCHETYPES = 4
ARCHETYPE_NAMES = ['A', 'B', 'C', 'D']

ENERGY_HIGH_THRESHOLD = 0.4
ENERGY_PENALTY_FACTOR = 0.6

INSTABILITY_COUPLING = 0.9
INSTABILITY_MIN = 0.15
INSTABILITY_MAX = 0.60
INSTABILITY_SMOOTHING = 0.80


# ============================================================================
# PERTURBATION PATTERN TYPES
# ============================================================================

class PerturbationPattern(Enum):
    NOISE_BURST = "noise_burst"
    DIRECTIONAL_PUSH = "directional_push"
    PARAMETER_SHIFT = "parameter_shift"
    ATTENTION_DROP = "attention_drop"
    OSCILLATION = "oscillation"


@dataclass
class PerturbationConfig:
    pattern: PerturbationPattern
    strength: float
    duration: int
    start_step: int
    
    def apply(self, engines: List, state: np.ndarray, step_offset: int = 0) -> Tuple[np.ndarray, dict]:
        progress = step_offset / max(self.duration, 1)
        
        if self.pattern == PerturbationPattern.NOISE_BURST:
            noise = np.random.randn(len(state)) * self.strength * (1 + progress)
            state = state + noise
            for engine in engines:
                engine_noise = np.random.randn(engine.dim) * self.strength * 0.1
                engine.state = engine.state + engine_noise
                norm = np.linalg.norm(engine.state)
                if norm > 1e-12:
                    engine.state = engine.state / norm
                    
        elif self.pattern == PerturbationPattern.DIRECTIONAL_PUSH:
            direction = np.random.randn(len(state))
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            state = state + self.strength * direction * (1 - progress * 0.5)
            
        elif self.pattern == PerturbationPattern.PARAMETER_SHIFT:
            shift = np.random.randn(len(state)) * self.strength
            state = state + shift * (1 - progress)
            
        elif self.pattern == PerturbationPattern.ATTENTION_DROP:
            if len(engines) > 0:
                target_idx = step_offset % len(engines)
                attenuation = max(0.1, 1.0 - self.strength * (1 - progress))
                engines[target_idx].state = engines[target_idx].state * attenuation
                
        elif self.pattern == PerturbationPattern.OSCILLATION:
            phase = 2 * np.pi * progress * 2
            oscillation = np.sin(phase) * self.strength * 0.5
            state = state + oscillation * np.random.randn(len(state))
        
        norm = np.linalg.norm(state)
        if norm > 1e-12:
            state = state / norm
            
        return state, {'pattern': self.pattern.value, 'progress': progress}


# ============================================================================
# CONTROLLER MODE ENUM
# ============================================================================

class ControllerMode(Enum):
    STABLE = "stable"
    PERTURBATION = "perturbation"
    RECOVERY = "recovery"
    UNCERTAIN = "uncertain"


# ============================================================================
# LAYER 2: COMPLEX WAVE-FIELD
# ============================================================================

class ComplexWaveField:
    def __init__(self, dim: int = DIM, tau: float = TAU):
        self.dim = dim
        self.tau = tau
        self.field = np.zeros(dim, dtype=np.complex128)
        self.initialized = False
        self.history = deque(maxlen=100)
    
    def ingest(self, state: np.ndarray) -> np.ndarray:
        if len(state) > self.dim:
            state = state[:self.dim]
        elif len(state) < self.dim:
            state = np.pad(state, (0, self.dim - len(state)))
        
        window = np.hanning(len(state))
        windowed = state * window
        fft = np.fft.fft(windowed).astype(np.complex128)
        
        if not self.initialized:
            self.field = fft.copy()
            self.initialized = True
        else:
            self.field = self.tau * self.field + (1 - self.tau) * fft
        
        self.history.append(self.field.copy())
        return self.field.copy()
    
    def get_magnitude(self) -> np.ndarray:
        return np.abs(self.field)
    
    def get_phase(self) -> np.ndarray:
        return np.angle(self.field)


# ============================================================================
# REF-01: ANCHOR
# ============================================================================

class Anchor:
    def __init__(self, dim: int = DIM, alpha: float = ALPHA_INIT):
        self.dim = dim
        self.alpha = alpha
        self.anchor = np.random.randn(dim)
        self.anchor = self.anchor / (np.linalg.norm(self.anchor) + 1e-12)
        self.history = deque(maxlen=1000)
        self.baseline_anchor = None
        self.baseline_saved = False
    
    def save_baseline(self):
        self.baseline_anchor = self.anchor.copy()
        self.baseline_saved = True
    
    def pull_to_baseline(self, strength: float = BASELINE_ANCHOR_STRENGTH):
        if self.baseline_anchor is not None:
            self.anchor = (1 - strength) * self.anchor + strength * self.baseline_anchor
            norm = np.linalg.norm(self.anchor)
            if norm > 1e-12:
                self.anchor = self.anchor / norm
    
    def update(self, state_vector: np.ndarray) -> np.ndarray:
        if len(state_vector) > self.dim:
            state_vector = state_vector[:self.dim]
        elif len(state_vector) < self.dim:
            state_vector = np.pad(state_vector, (0, self.dim - len(state_vector)))
        
        z_norm = state_vector / (np.linalg.norm(state_vector) + 1e-12)
        self.anchor = (1 - self.alpha) * self.anchor + self.alpha * z_norm
        self.anchor = self.anchor / (np.linalg.norm(self.anchor) + 1e-12)
        self.history.append(self.anchor.copy())
        return self.anchor
    
    def optimize_alpha(self, coherence: float, prev_coherence: Optional[float] = None) -> float:
        if prev_coherence is None:
            return self.alpha
        if coherence > 0.65 and coherence > prev_coherence:
            self.alpha = max(ALPHA_MIN, self.alpha - 0.0001)
        elif coherence < 0.35:
            self.alpha = min(ALPHA_MAX, self.alpha + 0.0003)
        self.alpha = max(ALPHA_MIN, min(ALPHA_MAX, self.alpha))
        return self.alpha
    
    def get(self) -> np.ndarray:
        return self.anchor.copy()


# ============================================================================
# REF-15: RESONANCE COHERENCE
# ============================================================================

class ResonanceCoherence:
    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
        dot = np.sum(a * b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return dot / (na * nb + 1e-12)
    
    @staticmethod
    def sigmoid(x: float, gamma: float = 20.0) -> float:
        return 1.0 / (1.0 + np.exp(-gamma * x))
    
    @staticmethod
    def homeostatic(C_raw: float, k: float = 10.0) -> float:
        return 1.0 / (1.0 + np.exp(-k * (C_raw - 0.5)))
    
    def compute(self, state: np.ndarray, anchor: Anchor) -> float:
        state_norm = state / (np.linalg.norm(state) + 1e-12)
        anchor_norm = anchor.get()
        R = self.cosine_sim(state_norm, anchor_norm)
        sig = self.sigmoid(R - 0.5)
        exp_boost = 1.0 - np.exp(-10.0 * max(0, R))
        C_raw = sig * exp_boost
        H = self.homeostatic(C_raw)
        return C_raw * H


# ============================================================================
# ANTI-PATTERN MEMORY
# ============================================================================

class AntiPatternMemory:
    def __init__(self, dim: int = DIM, max_capacity: int = ANTI_PATTERN_MAX_CAPACITY):
        self.dim = dim
        self.max_capacity = max_capacity
        self.patterns = []
        self.persistence = 0
        self.decay_rate = 0.008
        self.stored_count = 0
        self.store_cooldown = 0
    
    def update_persistence(self, coherence: float):
        if self.store_cooldown > 0:
            self.store_cooldown -= 1
        if coherence < COHERENCE_THRESHOLD:
            self.persistence += 1
        else:
            self.persistence = max(0, self.persistence - 1)
    
    def compute_misconnection_score(self, coherence: float, recent_stability: float = 0.0) -> float:
        stability_factor = min(0.8, recent_stability / 0.6)
        return (1 - coherence) * min(self.persistence, 15) * (1 - stability_factor)
    
    def store_anti_pattern(self, state: np.ndarray, coherence: float, 
                           phase_drift: float = 0, recent_stability: float = 0.0):
        misconnection = False
        
        if coherence < COHERENCE_THRESHOLD and self.persistence >= PERSISTENCE_THRESHOLD:
            misconnection = True
        
        if phase_drift > PHASE_DRIFT_THRESHOLD and coherence < 0.35:
            misconnection = True
        
        if not misconnection:
            return False
        if self.store_cooldown > 0:
            return False
        
        misconnection_score = self.compute_misconnection_score(coherence, recent_stability)
        if phase_drift > PHASE_DRIFT_THRESHOLD:
            misconnection_score += PHASE_DRIFT_PENALTY
        if misconnection_score < MISCONNECTION_MIN_SCORE:
            return False
        
        if len(state) > self.dim:
            state = state[:self.dim]
        elif len(state) < self.dim:
            state = np.pad(state, (0, self.dim - len(state)))
        if np.iscomplexobj(state):
            state = np.real(state)
        
        fft_pattern = np.fft.fft(state)
        pattern = fft_pattern / (np.linalg.norm(fft_pattern) + 1e-12)
        strength = min(0.7, misconnection_score * 0.4)
        
        for i, (pat, pat_strength, age) in enumerate(self.patterns):
            similarity = np.abs(np.sum(pattern * np.conj(pat))) / (np.linalg.norm(pattern) * np.linalg.norm(pat) + 1e-12)
            if similarity > 0.8:
                new_strength = min(1.0, pat_strength + strength * 0.15)
                self.patterns[i] = (pat, new_strength, 0)
                self.persistence = 0
                self.store_cooldown = STORE_COOLDOWN
                self.stored_count += 1
                return True
        
        if len(self.patterns) >= self.max_capacity:
            weakest_idx = np.argmin([s for _, s, _ in self.patterns])
            self.patterns.pop(weakest_idx)
        
        self.patterns.append((pattern, strength, 0))
        self.persistence = 0
        self.store_cooldown = STORE_COOLDOWN
        self.stored_count += 1
        return True
    
    def get_anti_field(self) -> np.ndarray:
        if not self.patterns:
            return np.zeros(self.dim, dtype=np.complex128)
        
        total_weight = 0
        anti_field = np.zeros(self.dim, dtype=np.complex128)
        
        for pattern, strength, age in self.patterns:
            effective_strength = strength * (ANTI_PATTERN_DECAY ** age)
            if effective_strength > 0.05:
                anti_field += effective_strength * pattern
                total_weight += effective_strength
        
        if total_weight > 0:
            anti_field = anti_field / total_weight
            norm = np.linalg.norm(anti_field)
            if norm > 0.5:
                anti_field = anti_field / norm * 0.5
        
        return anti_field
    
    def get_patterns(self) -> List:
        return self.patterns
    
    def decay(self):
        new_patterns = []
        for pattern, strength, age in self.patterns:
            new_strength = strength * (1 - self.decay_rate)
            if new_strength > 0.05:
                new_patterns.append((pattern, new_strength, age + 1))
        self.patterns = new_patterns[-self.max_capacity:]
    
    def get_size(self) -> int:
        return len(self.patterns)
    
    def get_strength(self) -> float:
        if not self.patterns:
            return 0.0
        return float(np.mean([s for _, s, _ in self.patterns]))


# ============================================================================
# MULTI-CHANNEL SIGNAL PROCESSOR
# ============================================================================

class MultiChannelProcessor:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.voice_weight = VOICE_WEIGHT
        self.body_weight = BODY_WEIGHT
        self.context_weight = CONTEXT_WEIGHT
        self.history = deque(maxlen=20)
    
    def process_voice(self, voice_signal: np.ndarray) -> np.ndarray:
        if len(voice_signal) > self.dim:
            voice_signal = voice_signal[:self.dim]
        elif len(voice_signal) < self.dim:
            voice_signal = np.pad(voice_signal, (0, self.dim - len(voice_signal)))
        return np.real(voice_signal)
    
    def process_body(self, joint_angles: np.ndarray) -> np.ndarray:
        if joint_angles is None:
            return np.zeros(self.dim)
        if len(joint_angles) > self.dim:
            joint_angles = joint_angles[:self.dim]
        elif len(joint_angles) < self.dim:
            joint_angles = np.pad(joint_angles, (0, self.dim - len(joint_angles)))
        joint_angles = np.real(joint_angles)
        
        freq_repr = np.zeros(self.dim)
        for i, angle in enumerate(joint_angles):
            freq_repr[i] = np.sin(angle) ** 2
            if i + self.dim // 2 < self.dim:
                freq_repr[i + self.dim // 2] = np.cos(angle) ** 2
        
        norm = np.linalg.norm(freq_repr)
        if norm > 1e-12:
            freq_repr = freq_repr / norm
        return freq_repr
    
    def process_context(self, recent_states: List[np.ndarray]) -> np.ndarray:
        if not recent_states:
            return np.zeros(self.dim)
        context = np.zeros(self.dim)
        for i, state in enumerate(recent_states[-5:]):
            if len(state) > self.dim:
                state = state[:self.dim]
            elif len(state) < self.dim:
                state = np.pad(state, (0, self.dim - len(state)))
            if np.iscomplexobj(state):
                state = np.real(state)
            weight = np.exp(-i / 2)
            context += weight * state
        norm = np.linalg.norm(context)
        if norm > 1e-12:
            context = context / norm
        return context
    
    def compute_unified_field(self, 
                               voice_signal: Optional[np.ndarray] = None,
                               body_angles: Optional[np.ndarray] = None,
                               recent_states: Optional[List[np.ndarray]] = None) -> np.ndarray:
        F_voice = self.process_voice(voice_signal) if voice_signal is not None else np.zeros(self.dim)
        F_body = self.process_body(body_angles) if body_angles is not None else np.zeros(self.dim)
        F_context = self.process_context(recent_states) if recent_states is not None else np.zeros(self.dim)
        
        unified = (self.voice_weight * F_voice + 
                   self.body_weight * F_body + 
                   self.context_weight * F_context)
        
        unified_fft = np.fft.fft(unified)
        
        norm = np.linalg.norm(unified_fft)
        if norm > 1e-12:
            unified_fft = unified_fft / norm
        
        return unified_fft


# ============================================================================
# PHASE-ENCODED MEMORY
# ============================================================================

class PhaseEncodedMemory:
    def __init__(self, dim: int = DIM, capacity: int = EMOTIONAL_MEMORY_CAPACITY):
        self.dim = dim
        self.capacity = capacity
        self.memories = []
        self.recall_threshold = MEMORY_RECALL_THRESHOLD
        self.store_threshold = MEMORY_STORE_THRESHOLD
        self.store_coherence = MEMORY_STORE_COHERENCE
    
    def _encode(self, state: np.ndarray) -> np.ndarray:
        if len(state) > self.dim:
            state = state[:self.dim]
        elif len(state) < self.dim:
            state = np.pad(state, (0, self.dim - len(state)))
        if np.iscomplexobj(state):
            state = np.real(state)
        
        fft = np.fft.fft(state)
        phase = np.angle(fft)
        phase_encoding = np.exp(1j * phase * PHASE_ENCODING_STRENGTH)
        encoded = fft * phase_encoding
        return encoded / (np.linalg.norm(encoded) + 1e-12)
    
    def _decode(self, encoded: np.ndarray) -> np.ndarray:
        decoded = np.real(np.fft.ifft(encoded))
        return decoded / (np.linalg.norm(decoded) + 1e-12)
    
    def store(self, emotional_state: np.ndarray, coherence: float, match_score: float):
        if coherence < self.store_coherence:
            return
        if match_score < self.store_threshold:
            return
        
        encoded = self._encode(emotional_state)
        
        for i, (mem, strength, age) in enumerate(self.memories):
            similarity = np.abs(np.sum(encoded * np.conj(mem))) / (np.linalg.norm(encoded) * np.linalg.norm(mem) + 1e-12)
            if similarity > 0.7:
                new_strength = min(1.0, strength + 0.1 * coherence)
                self.memories[i] = (mem, new_strength, 0)
                return
        
        if len(self.memories) >= self.capacity:
            weakest_idx = np.argmin([s for _, s, _ in self.memories])
            self.memories.pop(weakest_idx)
        
        initial_strength = min(0.8, coherence * 0.7)
        self.memories.append((encoded, initial_strength, 0))
    
    def recall(self, current_state: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        if not self.memories:
            return None, 0.0
        
        current_encoded = self._encode(current_state)
        
        best_score = -1
        best_match = None
        
        for mem, strength, age in self.memories:
            similarity = np.abs(np.sum(current_encoded * np.conj(mem))) / (np.linalg.norm(current_encoded) * np.linalg.norm(mem) + 1e-12)
            weighted_score = similarity * strength
            if weighted_score > self.recall_threshold and weighted_score > best_score:
                best_score = weighted_score
                best_match = mem
        
        if best_match is not None:
            recalled = self._decode(best_match)
            return recalled, best_score
        
        return None, 0.0
    
    def decay(self, decay_rate: float = 0.008):
        new_memories = []
        for mem, strength, age in self.memories:
            new_strength = strength * (1 - decay_rate)
            if new_strength > 0.08:
                new_memories.append((mem, new_strength, age + 1))
        self.memories = new_memories
    
    def get_size(self) -> int:
        return len(self.memories)
    
    def get_strength(self) -> float:
        if not self.memories:
            return 0.0
        return float(np.mean([s for _, s, _ in self.memories]))


# ============================================================================
# EMOTIONAL ARCHETYPES
# ============================================================================

class EmotionalArchetype:
    def __init__(self, name: str, dim: int = DIM):
        self.name = name
        self.dim = dim
        self.age = 0
        self.wins = 0
        self.pattern = np.random.randn(dim)
        norm = np.linalg.norm(self.pattern)
        if norm > 1e-12:
            self.pattern = self.pattern / norm
        self.strength = 0.5
    
    def update(self, input_pattern: np.ndarray, coherence: float):
        if len(input_pattern) > self.dim:
            input_pattern = input_pattern[:self.dim]
        elif len(input_pattern) < self.dim:
            input_pattern = np.pad(input_pattern, (0, self.dim - len(input_pattern)))
        if np.iscomplexobj(input_pattern):
            input_pattern = np.real(input_pattern)
        
        input_norm = input_pattern / (np.linalg.norm(input_pattern) + 1e-12)
        delta = ARCHETYPE_LEARNING_RATE * coherence * (input_norm - self.pattern)
        self.pattern = self.pattern + delta
        norm = np.linalg.norm(self.pattern)
        if norm > 1e-12:
            self.pattern = self.pattern / norm
        
        self.age += 1
        self.strength = 0.9 * self.strength + 0.1 * coherence
    
    def similarity(self, other: np.ndarray) -> float:
        if len(other) > self.dim:
            other = other[:self.dim]
        elif len(other) < self.dim:
            other = np.pad(other, (0, self.dim - len(other)))
        if np.iscomplexobj(other):
            other = np.real(other)
        
        other_norm = other / (np.linalg.norm(other) + 1e-12)
        return float(np.dot(self.pattern, other_norm))
    
    def get_pattern(self) -> np.ndarray:
        return self.pattern.copy()
    
    def get_info(self) -> dict:
        return {
            'name': self.name,
            'age': self.age,
            'strength': float(self.strength),
            'wins': self.wins
        }


class EmotionalStateManager:
    def __init__(self, dim: int = DIM, n_archetypes: int = N_ARCHETYPES):
        self.dim = dim
        self.archetypes = [EmotionalArchetype(ARCHETYPE_NAMES[i], dim) for i in range(n_archetypes)]
        self.current_emotion = None
        self.current_pattern = None
        self.emotion_history = deque(maxlen=200)
        self.dominance_count = np.zeros(n_archetypes)
    
    def get_archetype_pattern(self, name: str) -> np.ndarray:
        for archetype in self.archetypes:
            if archetype.name == name:
                return archetype.get_pattern()
        return np.zeros(self.dim)
    
    def compute_emotional_state(self, system_field: np.ndarray, coherence: float) -> Tuple[str, np.ndarray, float, float]:
        if len(system_field) > self.dim:
            system_field = system_field[:self.dim]
        elif len(system_field) < self.dim:
            system_field = np.pad(system_field, (0, self.dim - len(system_field)))
        
        system_mag = np.abs(system_field)
        system_norm = system_mag / (np.linalg.norm(system_mag) + 1e-12)
        energy = np.mean(system_mag)
        
        similarities = []
        for idx, archetype in enumerate(self.archetypes):
            s = archetype.similarity(system_norm)
            if idx == 0 and energy > ENERGY_HIGH_THRESHOLD:
                s = s * ENERGY_PENALTY_FACTOR
            similarities.append(s)
        
        similarities = np.array(similarities)
        total_dominance = np.sum(self.dominance_count)
        if total_dominance > 1e-12:
            dominance_freq = self.dominance_count / total_dominance
            
            for i in range(len(self.archetypes)):
                if dominance_freq[i] < WEAK_ATTRACTOR_THRESHOLD:
                    similarities[i] += WEAK_ATTRACTOR_BOOST
            
            dominance_excess = np.maximum(0, dominance_freq - DOMINANCE_TARGET)
            rarity_gap = np.maximum(0, RARITY_TARGET - dominance_freq)
            similarities = similarities - DOMINANCE_SELECTION_PENALTY * dominance_excess
            similarities = similarities + RARITY_SELECTION_BOOST * rarity_gap
            if self.current_emotion in ARCHETYPE_NAMES:
                current_idx = ARCHETYPE_NAMES.index(self.current_emotion)
                if dominance_freq[current_idx] < DOMINANCE_TARGET:
                    similarities[current_idx] += EMOTION_PERSISTENCE_BOOST

        exp_scores = np.exp(similarities / SOFTMAX_TEMPERATURE)
        probs = exp_scores / (np.sum(exp_scores) + 1e-12)
        
        selected_idx = np.random.choice(len(self.archetypes), p=probs)
        best_match = self.archetypes[selected_idx]
        best_score = similarities[selected_idx]
        
        self.dominance_count *= DOMINANCE_DECAY
        self.dominance_count[selected_idx] += 1
        total_dominance = np.sum(self.dominance_count)
        dominance_freq = self.dominance_count / (total_dominance + 1e-12)
        
        best_match.wins += 1
        self.current_emotion = best_match.name
        self.current_pattern = best_match.get_pattern()
        self.emotion_history.append(self.current_emotion)
        
        for i, archetype in enumerate(self.archetypes):
            repulsion_penalty = 1 - (dominance_freq[i] * DOMINANCE_PENALTY)
            update_strength = probs[i] * coherence * repulsion_penalty
            archetype.update(system_norm, update_strength)
        
        if len(self.emotion_history) > 10:
            recent_emotions = list(self.emotion_history)[-50:]
            emotion_diversity = len(set(recent_emotions)) / N_ARCHETYPES
        else:
            emotion_diversity = 1.0
        
        return self.current_emotion, self.current_pattern, best_score, emotion_diversity
    
    def get_emotional_field(self) -> np.ndarray:
        if self.current_pattern is not None:
            return self.current_pattern
        return np.zeros(self.dim)
    
    def get_all_states(self) -> List[dict]:
        return [a.get_info() for a in self.archetypes]
    
    def get_emotion_history(self) -> List[str]:
        return list(self.emotion_history)


# ============================================================================
# ENGINE
# ============================================================================

class Engine:
    ENGINE_TYPES = ['detector', 'predictor', 'generator', 'integrator']
    
    def __init__(self, name: str, engine_type: str, dim: int = DIM):
        self.name = name
        self.type = engine_type
        self.dim = dim
        self.state = np.random.randn(dim)
        self.state = self.state / (np.linalg.norm(self.state) + 1e-12)
        self.wavefield = ComplexWaveField(dim=dim)
        self.history = deque(maxlen=10)
    
    def process(self, field_input: np.ndarray, emotional_influence: np.ndarray = None,
                noise_level: float = LAMBDA_3) -> np.ndarray:
        field_real = np.real(field_input)
        if len(field_real) > self.dim:
            field_real = field_real[:self.dim]
        elif len(field_real) < self.dim:
            field_real = np.pad(field_real, (0, self.dim - len(field_real)))
        
        if self.type == 'detector':
            update = np.tanh(field_real)
        elif self.type == 'predictor':
            update = np.gradient(field_real)
        elif self.type == 'generator':
            update = np.random.randn(self.dim) * 0.1
        else:
            update = 0.9 * self.state + 0.1 * field_real
        
        if emotional_influence is not None:
            if len(emotional_influence) > self.dim:
                emotional_influence = emotional_influence[:self.dim]
            elif len(emotional_influence) < self.dim:
                emotional_influence = np.pad(emotional_influence, (0, self.dim - len(emotional_influence)))
            if np.iscomplexobj(emotional_influence):
                emotional_influence = np.real(emotional_influence)
            emotional_norm = emotional_influence / (np.linalg.norm(emotional_influence) + 1e-12)
            update = update + 0.15 * emotional_norm
        
        noise = noise_level * np.random.randn(self.dim)
        self.state = self.state + 0.1 * update + noise
        norm = np.linalg.norm(self.state)
        if norm > 1e-12:
            self.state = self.state / norm
        
        self.wavefield.ingest(self.state)
        return self.state.copy()
    
    def get_state(self) -> np.ndarray:
        return self.state.copy()
    
    def get_magnitude(self) -> np.ndarray:
        return self.wavefield.get_magnitude()


# ============================================================================
# FIELD COMPUTATION
# ============================================================================

def compute_field(states: List[np.ndarray]) -> np.ndarray:
    field = np.zeros(DIM, dtype=np.complex128)
    for s in states:
        field += np.fft.fft(s)
    return field / len(states)


def compute_phase_drift(prev_field: np.ndarray, curr_field: np.ndarray) -> float:
    if prev_field is None or curr_field is None:
        return 0.0
    
    prev_phase = np.angle(prev_field)
    curr_phase = np.angle(curr_field)
    phase_diff = np.abs(prev_phase - curr_phase)
    phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
    return float(np.mean(phase_diff))


# ============================================================================
# DESIRABILITY FIELD
# ============================================================================

class DesirabilityField:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.coherence_weight = DESIRABILITY_COHERENCE_WEIGHT
        self.anchor_dist_weight = DESIRABILITY_ANCHOR_DIST_WEIGHT
        self.anti_pattern_weight = DESIRABILITY_ANTI_PATTERN_WEIGHT
        self.entropy_weight = DESIRABILITY_ENTROPY_WEIGHT
        self.novelty_weight = NOVELTY_WEIGHT
        self.novelty_history = deque(maxlen=NOVELTY_HISTORY)
        self.history = deque(maxlen=100)
    
    def compute_coherence(self, z: np.ndarray, anchor: np.ndarray) -> float:
        z_norm = z / (np.linalg.norm(z) + 1e-12)
        a_norm = anchor / (np.linalg.norm(anchor) + 1e-12)
        return float(np.dot(z_norm, a_norm))
    
    def distance_to_anchor(self, z: np.ndarray, anchor: np.ndarray) -> float:
        return float(np.linalg.norm(z - anchor))
    
    def min_distance_to_anti(self, z: np.ndarray, anti_patterns: list) -> float:
        if not anti_patterns:
            return 1.0
        distances = [np.linalg.norm(z - np.real(ap)) for ap, _, _ in anti_patterns]
        return float(min(distances))
    
    def entropy(self, z: np.ndarray) -> float:
        fft = np.fft.fft(z)
        power = np.abs(fft) ** 2
        p = power / (np.sum(power) + 1e-12)
        p = p[p > 0]
        if len(p) == 0:
            return 1.0
        entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(len(z))
        return entropy / max_entropy
    
    def novelty(self, z: np.ndarray) -> float:
        if len(self.novelty_history) == 0:
            self.novelty_history.append(z.copy())
            return 1.0
        
        min_dist = np.inf
        for state in self.novelty_history:
            dist = np.linalg.norm(z - state)
            if dist < min_dist:
                min_dist = dist
        
        self.novelty_history.append(z.copy())
        return min(1.0, min_dist / 0.5)
    
    def compute(self, z: np.ndarray, anchor: np.ndarray, anti_patterns: list,
                include_novelty: bool = True) -> float:
        coherence = self.compute_coherence(z, anchor)
        dist_a = self.distance_to_anchor(z, anchor)
        dist_anti = self.min_distance_to_anti(z, anti_patterns)
        entropy = self.entropy(z)
        novelty = self.novelty(z) if include_novelty else 0.0
        
        dist_a_norm = min(1.0, dist_a / 2.0)
        anti_penalty = ANTI_GOAL_PENALTY_STRENGTH * (1.0 - min(1.0, dist_anti))
        
        U = (self.coherence_weight * coherence - 
             self.anchor_dist_weight * dist_a_norm -
             self.anti_pattern_weight * anti_penalty -
             self.entropy_weight * entropy +
             self.novelty_weight * novelty)
        
        self.history.append(U)
        return float(U)


# ============================================================================
# GOAL TOPOLOGY
# ============================================================================

class GoalTopology:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.goal = None
        self.goal_desirability = -np.inf
        self.goal_history = deque(maxlen=50)
        self.goal_momentum = GOAL_MOMENTUM
    
    def update_goal(self, z: np.ndarray, desirability: float):
        self.goal_history.append(desirability)
        
        if desirability > self.goal_desirability + GOAL_UPDATE_THRESHOLD:
            self.goal = z.copy()
            self.goal_desirability = desirability
        else:
            if self.goal is not None:
                self.goal = (self.goal_momentum * self.goal + 
                            (1 - self.goal_momentum) * z)
                norm = np.linalg.norm(self.goal)
                if norm > 1e-12:
                    self.goal = self.goal / norm
    
    def get_goal(self) -> Optional[np.ndarray]:
        return self.goal
    
    def get_goal_direction(self, z: np.ndarray) -> np.ndarray:
        if self.goal is None:
            return np.zeros(self.dim)
        
        direction = self.goal - z
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction = direction / norm
        
        return direction
    
    def get_goal_distance(self, z: np.ndarray) -> float:
        if self.goal is None:
            return 0.0
        return float(np.linalg.norm(z - self.goal))
    
    def get_goal_strength(self) -> float:
        return self.goal_desirability if self.goal_desirability > -np.inf else 0.0


# ============================================================================
# IDENTITY CONTINUITY
# ============================================================================

class IdentityContinuity:
    def __init__(self, window: int = IDENTITY_WINDOW):
        self.window = window
        self.state_history = deque(maxlen=window)
    
    def update(self, state: np.ndarray):
        self.state_history.append(state.copy())
    
    def get_identity_drift(self) -> float:
        if len(self.state_history) < self.window:
            return 0.0
        
        current = self.state_history[-1]
        past = self.state_history[0]
        
        return float(np.linalg.norm(current - past))
    
    def get_identity_stability(self) -> float:
        drift = self.get_identity_drift()
        return 1.0 / (1.0 + drift)


# ============================================================================
# PERTURBATION CONTROLLER
# ============================================================================

class PerturbationController:
    def __init__(self, config: PerturbationConfig = None):
        self.config = config or PerturbationConfig(
            pattern=PerturbationPattern.NOISE_BURST,
            strength=PERTURBATION_STRENGTH,
            duration=PERTURBATION_DURATION,
            start_step=PERTURBATION_START
        )
        self.active = False
        self.steps_in_perturbation = 0
        self.baseline_errors = []
        self.perturbation_errors = []
        self.recovery_errors = []
        self.baseline_saved = False
        self.last_perturbation_metadata = {}
    
    def should_perturb(self, step: int) -> bool:
        if step >= self.config.start_step and step < self.config.start_step + self.config.duration:
            self.active = True
            return True
        elif step >= self.config.start_step + self.config.duration:
            self.active = False
            return False
        return False
    
    def apply_perturbation(self, state: np.ndarray, engines: List, step: int) -> Tuple[np.ndarray, dict]:
        if not self.active:
            return state, {}
        self.steps_in_perturbation += 1
        step_offset = self.steps_in_perturbation - 1
        state, metadata = self.config.apply(engines, state, step_offset)
        self.last_perturbation_metadata = metadata
        return state, metadata
    
    def record_error(self, error: float, phase: str):
        if phase == 'baseline':
            self.baseline_errors.append(error)
        elif phase == 'perturbation':
            self.perturbation_errors.append(error)
        elif phase == 'recovery':
            self.recovery_errors.append(error)
    
    def get_statistics(self) -> dict:
        baseline_mean = np.mean(self.baseline_errors) if self.baseline_errors else 0
        perturbation_max = max(self.perturbation_errors) if self.perturbation_errors else 0
        perturbation_mean = np.mean(self.perturbation_errors) if self.perturbation_errors else 0
        recovery_mean = np.mean(self.recovery_errors[-50:]) if len(self.recovery_errors) >= 50 else (np.mean(self.recovery_errors) if self.recovery_errors else 0)
        
        spike_ratio = perturbation_max / (baseline_mean + 1e-12)
        recovery_improvement = (perturbation_mean - recovery_mean) / (perturbation_mean + 1e-12) if perturbation_mean > 0 else 0
        final_recovery_ratio = recovery_mean / (baseline_mean + 1e-12)
        
        return {
            'baseline_mean': baseline_mean,
            'perturbation_max': perturbation_max,
            'perturbation_mean': perturbation_mean,
            'recovery_mean': recovery_mean,
            'spike_ratio': spike_ratio,
            'recovery_improvement': recovery_improvement,
            'final_recovery_ratio': final_recovery_ratio,
            'fully_recovered': final_recovery_ratio < 2.0
        }


# ============================================================================
# SELF-MODEL (with recovery learning rate)
# ============================================================================

class SelfModel:
    def __init__(self, dim: int = DIM, history_window: int = SELF_MODEL_K):
        self.dim = dim
        self.history_window = history_window
        
        self.A = np.eye(dim) * 0.95
        self.B = np.eye(dim) * 0.05
        self.C = np.eye(dim) * 0.05
        self.b = np.zeros(dim)
        
        self.P_A = np.eye(dim) * (1.0 / RLS_EPSILON)
        self.history_states = deque(maxlen=history_window)
        
        self.prediction_error = 0.0
        self.prediction_error_history = deque(maxlen=200)
        self.self_coherence = 0.5
        
        self.adaptive_lr = SELF_MODEL_ADAPTIVE_LR
        self.adaptive_factor = SELF_MODEL_ADAPTIVE_FACTOR
        self.base_memory = SELF_MODEL_BASE_MEMORY
        
        self.A_baseline = None
        self.B_baseline = None
        self.b_baseline = None
        
        self.embedding_dim = min(dim, 16)
        self.W_embed = np.random.randn(history_window, self.embedding_dim) / np.sqrt(history_window)
        self.consecutive_high_errors = 0
        
        self.learning_rate_multiplier = 1.0
    
    def save_baseline(self):
        self.A_baseline = self.A.copy()
        self.B_baseline = self.B.copy()
        self.b_baseline = self.b.copy()
    
    def pull_to_baseline(self, strength: float = 0.05):
        if self.A_baseline is not None:
            self.A = (1 - strength) * self.A + strength * self.A_baseline
            self.B = (1 - strength) * self.B + strength * self.B_baseline
            self.b = (1 - strength) * self.b + strength * self.b_baseline
            
            norm = np.linalg.norm(self.A)
            if norm > 2.0:
                self.A = self.A / norm * 1.5
    
    def set_learning_rate_multiplier(self, multiplier: float):
        self.learning_rate_multiplier = max(0.1, min(1.0, multiplier))
    
    def _temporal_embedding(self) -> np.ndarray:
        if len(self.history_states) == 0:
            return np.zeros(self.dim)
        return np.mean(list(self.history_states), axis=0)
    
    def predict(self, z: np.ndarray, goal: np.ndarray) -> np.ndarray:
        phi = self._temporal_embedding()
        z_pred = (self.A @ z) + (self.B @ goal) + (self.C @ phi) + self.b
        norm = np.linalg.norm(z_pred)
        if norm > 1e-12:
            z_pred = z_pred / norm
        return z_pred
    
    def update(self, z_actual: np.ndarray, z_pred: np.ndarray, 
               z_input: np.ndarray, goal: np.ndarray,
               coherence: float, self_coherence: float,
               is_perturbed: bool = False):
        
        error = z_actual - z_pred
        
        error = np.clip(error, -CORRECTION_CAP, CORRECTION_CAP)
        
        self.prediction_error = np.linalg.norm(error) ** 2
        
        if np.isnan(self.prediction_error) or np.isinf(self.prediction_error):
            print(f"[EMERGENCY] NaN/Inf detected! Forcing reset...")
            self.A = np.eye(self.dim) * 0.95
            self.B = np.eye(self.dim) * 0.05
            self.b = np.zeros(self.dim)
            self.prediction_error = 0.5
            self.self_coherence = 0.5
            self.prediction_error_history.append(self.prediction_error)
            self.consecutive_high_errors = 0
            return
        
        if self.prediction_error > CRITICAL_PREDICTION_ERROR:
            print(f"[EMERGENCY] Catastrophic error {self.prediction_error:.2f}")
            self.pull_to_baseline(strength=0.8)
            self.prediction_error = MAX_PREDICTION_ERROR * 0.5
            self.prediction_error_history.append(self.prediction_error)
            self.self_coherence = 0.5
            return
        
        if self.prediction_error > MAX_PREDICTION_ERROR:
            print(f"[WARNING] High error {self.prediction_error:.2f}")
            self.pull_to_baseline(strength=0.3)
            self.prediction_error = MAX_PREDICTION_ERROR * 0.5
            self.prediction_error_history.append(self.prediction_error)
            self.self_coherence = 0.5
            return
        
        self.self_coherence = np.exp(-SELF_MODEL_BETA * self.prediction_error)
        self.prediction_error_history.append(self.prediction_error)
        
        eta = (SELF_MODEL_LEARNING_RATE * coherence * self.self_coherence * 
               self.learning_rate_multiplier)
        
        if is_perturbed and self.adaptive_lr and self.prediction_error > 0.1:
            eta = eta * self.adaptive_factor
        
        eta = max(0.0001, min(0.5, eta))
        
        self.A = self.A + eta * np.outer(error, z_input)
        self.B = self.B + eta * np.outer(error, goal)
        self.b = self.b + eta * error
        
        self.A = self.A / (1.0 + np.linalg.norm(self.A) * 0.01)
        self.B = self.B / (1.0 + np.linalg.norm(self.B) * 0.01)
        
        self.history_states.append(z_actual.copy())
    
    def get_prediction_error(self) -> float:
        return self.prediction_error
    
    def get_self_coherence(self) -> float:
        return self.self_coherence
    
    def get_error_trend(self, window: int = 50) -> float:
        if len(self.prediction_error_history) < window:
            return 0.0
        recent = list(self.prediction_error_history)[-window:]
        if len(recent) < 2:
            return 0.0
        return recent[-1] - recent[0]
    
    def get_error_baseline(self, window: int = 50) -> float:
        if len(self.prediction_error_history) < window:
            return self.prediction_error
        return float(np.mean(list(self.prediction_error_history)[-window:]))


# ============================================================================
# TRAJECTORY-BASED STATE CLASSIFIER
# ============================================================================

class TrajectoryStateClassifier:
    """
    Classifies system state based on trajectory window, not instantaneous error.
    """
    
    def __init__(self, baseline_error: float, window_size: int = TRAJECTORY_WINDOW):
        self.baseline_error = baseline_error
        self.window_size = window_size
        self.error_buffer = deque(maxlen=window_size)
        
        self.derivative_buffer = deque(maxlen=DERIVATIVE_WINDOW)
        self.disturbance_counter = 0
        self.baseline_confidence = 0.0

    def update_baseline_confidence(self, baseline_confidence: float):
        self.baseline_confidence = baseline_confidence
    
    def update_baseline(self, new_baseline: float):
        self.baseline_error = new_baseline
    
    def add_error(self, error: float):
        self.error_buffer.append(error)
    
    def compute_trajectory_features(self) -> Dict[str, float]:
        if len(self.error_buffer) < self.window_size // 2:
            return {
                'mean': self.baseline_error,
                'variance': 0.0,
                'normalized_variance': 0.0,
                'slope': 0.0,
                'is_stable': True,
                'is_recovering': False,
                'is_uncertain': False,
                'window_size': len(self.error_buffer)
            }
        
        errors = list(self.error_buffer)
        n = len(errors)
        
        mean_error = np.mean(errors)
        variance = np.var(errors)
        x = np.arange(n)
        slope = np.polyfit(x, errors, 1)[0] if n > 1 else 0.0
        normalized_variance = variance / (self.baseline_error ** 2 + 1e-12)
        
        threshold_scale = 1.0 + min(1.0, self.baseline_confidence / BASELINE_CONFIDENCE_THRESHOLD) * BASELINE_CONFIDENCE_WIDENING
        stable_variance_threshold = STABLE_VARIANCE_MULTIPLIER * threshold_scale
        stable_slope_threshold = STABLE_SLOPE_THRESHOLD * threshold_scale
        recovery_variance_threshold = RECOVERY_VARIANCE_MULTIPLIER * threshold_scale
        
        is_stable = (
            mean_error < self.baseline_error * STABLE_MEAN_MULTIPLIER and
            normalized_variance < stable_variance_threshold and
            abs(slope) < stable_slope_threshold
        )
        
        is_recovering = (
            mean_error > self.baseline_error * RECOVERY_MEAN_MULTIPLIER or
            normalized_variance > recovery_variance_threshold or
            abs(slope) > stable_slope_threshold * 2
        )
        
        is_uncertain = (
            normalized_variance > UNCERTAIN_VARIANCE_MULTIPLIER and
            abs(slope) < UNCERTAIN_SLOPE_THRESHOLD and
            self.baseline_confidence > BASELINE_CONFIDENCE_THRESHOLD
        )
        
        return {
            'mean': mean_error,
            'variance': variance,
            'normalized_variance': normalized_variance,
            'slope': slope,
            'is_stable': is_stable,
            'is_recovering': is_recovering,
            'is_uncertain': is_uncertain,
            'window_size': len(errors)
        }
    
    def detect_disturbance_derivative(self, current_error: float, step: int) -> Tuple[bool, float, float]:
        if len(self.derivative_buffer) > 0:
            prev_error = self.derivative_buffer[-1]
            derivative = current_error - prev_error
        else:
            derivative = 0.0

        if len(self.derivative_buffer) > 1:
            prev_prev_error = self.derivative_buffer[-2]
            prev_derivative = prev_error - prev_prev_error
            acceleration = derivative - prev_derivative
        else:
            acceleration = 0.0

        self.derivative_buffer.append(current_error)

        threshold_scale = 1.0 + min(1.0, self.baseline_confidence / BASELINE_CONFIDENCE_THRESHOLD)
        derivative_threshold = DISTURBANCE_DERIVATIVE_THRESHOLD * threshold_scale
        acceleration_threshold = DISTURBANCE_ACCELERATION_THRESHOLD * threshold_scale

        if abs(derivative) > derivative_threshold and abs(acceleration) > acceleration_threshold:
            self.disturbance_counter += 1
        else:
            self.disturbance_counter = max(0, self.disturbance_counter - 1)

        is_disturbed = self.disturbance_counter >= DISTURBANCE_PERSISTENCE

        return is_disturbed, abs(derivative), abs(acceleration)
    
    def classify_state(self, current_error: float, step: int) -> Tuple[ControllerMode, Dict]:
        self.add_error(current_error)
        features = self.compute_trajectory_features()
        is_disturbed, derivative, acceleration = self.detect_disturbance_derivative(current_error, step)
        
        if is_disturbed and not features['is_uncertain']:
            mode = ControllerMode.PERTURBATION
        elif features['is_uncertain']:
            mode = ControllerMode.UNCERTAIN
        elif features['is_recovering']:
            mode = ControllerMode.RECOVERY
        elif features['is_stable']:
            mode = ControllerMode.STABLE
        else:
            mode = None
        
        return mode, {
            'mean': features['mean'],
            'variance': features['variance'],
            'normalized_variance': features['normalized_variance'],
            'slope': features['slope'],
            'derivative': derivative,
            'acceleration': acceleration,
            'is_disturbed': is_disturbed,
            'is_uncertain': features['is_uncertain'],
            'window_size': features['window_size']
        }


# ============================================================================
# SELF-MODIFICATION CONTROLLER (WITH TIME-BASED ESCAPE)
# ============================================================================

class SelfModificationController:
    """
    Uses trajectory-based state classification with time-based escape from PERTURBATION.
    
    CRITICAL FIXES:
    - Time-based escape from PERTURBATION after MIN_RECOVERY_DURATION steps
    - Minimum perturbation duration (prevents premature exit)
    - Gradual unfreezing during recovery
    - Recovery timeout watchdog
    """
    
    def __init__(self):
        self.lambda_1_init = LAMBDA_1
        self.lambda_2_init = LAMBDA_2
        self.lambda_3_init = LAMBDA_3
        self.lambda_4_init = LAMBDA_4
        self.alpha_goal_init = ALPHA_GOAL_INIT
        
        self.lambda_1 = self.lambda_1_init
        self.lambda_2 = self.lambda_2_init
        self.lambda_3 = self.lambda_3_init
        self.lambda_4 = self.lambda_4_init
        self.alpha_goal = self.alpha_goal_init
        
        self.adaptation_rate = 0.02
        self.recovery_rate = 0.01
        
        self.mode = ControllerMode.STABLE
        self.mode_entry_step = 0
        self.steps_in_mode = 0
        self.last_perturbation_end_step = 0
        
        self.error_momentum = 0.02
        self.error_history = deque(maxlen=200)
        self.confidence = 1.0
        self.confidence_alpha = CONFIDENCE_EMA_ALPHA
        self.baseline_confidence = 0.0
        self.baseline_confidence_alpha = BASELINE_CONFIDENCE_ALPHA
        
        self.baseline_error = 0.02
        self.baseline_initialized = False
        self.baseline_learning_buffer = deque(maxlen=200)
        
        self.classifier = None
        
        self.pending_mode = None
        self.pending_mode_start = 0
        self.dwell_time = STATE_DWELL_TIME
        
        self.lambda_history = deque(maxlen=500)
        self.adaptation_count = 0
        self.classification_history = deque(maxlen=200)
        
        self.track_params()
    
    def learn_initial_baseline(self, prediction_error: float, step: int):
        if not self.baseline_initialized and step < 200:
            self.baseline_learning_buffer.append(prediction_error)
            if len(self.baseline_learning_buffer) == 200:
                self.baseline_error = np.mean(self.baseline_learning_buffer)
                self.baseline_initialized = True
                self.classifier = TrajectoryStateClassifier(self.baseline_error)
                print(f"[BASELINE] Initialized at {self.baseline_error:.4f}")
    
    def update_error_momentum(self, prediction_error: float):
        self.error_momentum = (ERROR_MOMENTUM_ALPHA * self.error_momentum + 
                               (1 - ERROR_MOMENTUM_ALPHA) * prediction_error)
        self.error_history.append(prediction_error)
    
    def compute_confidence(self) -> float:
        return self.confidence

    def update_confidence(self):
        normalized = self.error_momentum / (self.baseline_error + 1e-12)
        instant_confidence = np.exp(-min(normalized, 5.0))
        self.confidence = (self.confidence_alpha * self.confidence +
                           (1 - self.confidence_alpha) * instant_confidence)

    def update_baseline_confidence(self, prediction_error: float):
        baseline_diff = abs(prediction_error - self.baseline_error)
        instant_baseline_conf = baseline_diff / (self.baseline_error + 1e-12)
        instant_baseline_conf = min(1.0, instant_baseline_conf)
        self.baseline_confidence = (
            self.baseline_confidence_alpha * self.baseline_confidence +
            (1 - self.baseline_confidence_alpha) * instant_baseline_conf
        )

    def _clamp_params(self):
        self.lambda_1 = float(np.clip(self.lambda_1, 0.01, 0.5))
        self.lambda_2 = float(np.clip(self.lambda_2, 0.01, 0.5))
        self.lambda_3 = float(np.clip(self.lambda_3, LAMBDA_3_MIN, LAMBDA_3_MAX))
        self.lambda_4 = float(np.clip(self.lambda_4, LAMBDA_4_MIN, LAMBDA_4_MAX))
        self.alpha_goal = float(np.clip(self.alpha_goal, ALPHA_GOAL_MIN, ALPHA_GOAL_MAX))
    
    def _apply_adaptation_decay(self):
        self.lambda_1 = self.lambda_1 * ADAPTATION_DECAY + self.lambda_1_init * (1 - ADAPTATION_DECAY)
        self.lambda_2 = self.lambda_2 * ADAPTATION_DECAY + self.lambda_2_init * (1 - ADAPTATION_DECAY)
        self.lambda_3 = self.lambda_3 * ADAPTATION_DECAY + self.lambda_3_init * (1 - ADAPTATION_DECAY)
        self.lambda_4 = self.lambda_4 * ADAPTATION_DECAY + self.lambda_4_init * (1 - ADAPTATION_DECAY)
        self.alpha_goal = self.alpha_goal * ADAPTATION_DECAY + self.alpha_goal_init * (1 - ADAPTATION_DECAY)
        self.adaptation_rate = max(0.005, self.adaptation_rate * 0.995)
    
    def update_perturbation_end(self, step: int):
        self.last_perturbation_end_step = step
    
    def determine_mode_with_dwell(self, proposed_mode: ControllerMode, step: int) -> ControllerMode:
        if proposed_mode == self.mode:
            self.pending_mode = None
            return self.mode
        
        if self.pending_mode != proposed_mode:
            self.pending_mode = proposed_mode
            self.pending_mode_start = step
            return self.mode
        
        if step - self.pending_mode_start >= self.dwell_time:
            self.pending_mode = None
            return proposed_mode
        
        return self.mode
    
    def update(self, prediction_error: float, coherence: float, step: int,
               current_state: np.ndarray, phase: str, steps_since_perturbation_end: int):
        
        self.learn_initial_baseline(prediction_error, step)
        
        if not self.baseline_initialized or self.classifier is None:
            return
        
        self.update_error_momentum(prediction_error)
        self.update_confidence()
        self.update_baseline_confidence(prediction_error)
        
        self.classifier.update_baseline(self.baseline_error)
        self.classifier.update_baseline_confidence(self.baseline_confidence)
        
        proposed_mode, classification = self.classifier.classify_state(self.error_momentum, step)
        
        self.classification_history.append({
            'step': step,
            'mode': proposed_mode.value if proposed_mode else 'none',
            'mean': classification['mean'],
            'variance': classification['variance'],
            'normalized_variance': classification['normalized_variance'],
            'slope': classification['slope'],
            'derivative': classification['derivative'],
            'acceleration': classification['acceleration'],
            'is_uncertain': classification['is_uncertain'],
            'baseline_confidence': self.baseline_confidence,
            'confidence': self.compute_confidence()
        })

        # Recovery lock: avoid reacting to residual recovery noise as new perturbation
        if self.mode == ControllerMode.RECOVERY and proposed_mode == ControllerMode.PERTURBATION:
            lock_threshold = self.baseline_error * RECOVERY_PERTURBATION_LOCK_MULTIPLIER
            if self.steps_in_mode < RECOVERY_PERTURBATION_LOCK and self.error_momentum < lock_threshold:
                proposed_mode = self.mode

        # ====================================================================
        # CRITICAL FIX: Time-based escape from PERTURBATION
        # ====================================================================
        if self.mode == ControllerMode.PERTURBATION:
            steps_in_perturbation = step - self.mode_entry_step
            # Force exit to recovery after MIN_RECOVERY_DURATION steps
            if steps_in_perturbation > MIN_RECOVERY_DURATION:
                proposed_mode = ControllerMode.RECOVERY
                # Also gradually unfreeze the self-model
                if hasattr(self, 'self_model'):
                    self.self_model.set_learning_rate_multiplier(RECOVERY_LEARNING_RATE_MULTIPLIER)
                print(f"[TIMEOUT] PERTURBATION timeout after {steps_in_perturbation} steps → RECOVERY")
        
        # ====================================================================
        # Minimum perturbation duration (prevent premature exit)
        # ====================================================================
        if self.mode == ControllerMode.PERTURBATION:
            steps_in_perturbation = step - self.mode_entry_step
            if steps_in_perturbation < MIN_PERTURBATION_DURATION and proposed_mode != ControllerMode.PERTURBATION:
                # Don't exit perturbation too early
                proposed_mode = ControllerMode.PERTURBATION
        
        # ====================================================================
        # Recovery timeout watchdog
        # ====================================================================
        if self.mode == ControllerMode.RECOVERY:
            steps_in_recovery = step - self.mode_entry_step
            if steps_in_recovery > RECOVERY_TIMEOUT:
                proposed_mode = ControllerMode.STABLE
                print(f"[WATCHDOG] RECOVERY timeout after {steps_in_recovery} steps → STABLE")

        if proposed_mode is None:
            proposed_mode = self.mode
        
        new_mode = self.determine_mode_with_dwell(proposed_mode, step)
        
        if new_mode == self.mode:
            self.steps_in_mode += 1
        else:
            self.steps_in_mode = 0
        
        if new_mode != self.mode:
            if new_mode == ControllerMode.RECOVERY and self.confidence < CONFIDENCE_FLOOR_RECOVERY:
                self.confidence = CONFIDENCE_FLOOR_RECOVERY

            confidence = self.compute_confidence()
            print(f"[MODE] {self.mode.value} -> {new_mode.value} at step {step}")
            print(f"       features: mean={classification['mean']:.4f} "
                  f"var={classification['normalized_variance']:.2f} "
                  f"slope={classification['slope']:.5f}")
            print(f"       err_mom={self.error_momentum:.4f} base={self.baseline_error:.4f} conf={confidence:.3f}")
            
            if new_mode == ControllerMode.STABLE:
                self.adaptation_rate = min(0.01, self.adaptation_rate)
            
            self.mode = new_mode
            self.mode_entry_step = step
        
        # Mode-specific adaptation
        if self.mode == ControllerMode.PERTURBATION:
            self.lambda_3 = min(LAMBDA_3_MAX, self.lambda_3 + self.adaptation_rate * 2)
            self.lambda_4 = max(LAMBDA_4_MIN, self.lambda_4 - self.adaptation_rate * 1.5)
            self.alpha_goal = max(ALPHA_GOAL_MIN, self.alpha_goal - self.adaptation_rate)
            self.adaptation_count += 1
            
        elif self.mode == ControllerMode.RECOVERY:
            self.lambda_3 = max(LAMBDA_3_MIN, self.lambda_3 - self.recovery_rate * 0.5)
            self.lambda_4 = min(LAMBDA_4_MAX, self.lambda_4 + self.recovery_rate)
            self.alpha_goal = min(ALPHA_GOAL_MAX, self.alpha_goal + self.recovery_rate * 0.5)
            
        else:
            if self.error_momentum < self.baseline_error * 0.8:
                self.lambda_3 = max(LAMBDA_3_MIN, self.lambda_3 - self.adaptation_rate * 0.1)
                self.alpha_goal = min(ALPHA_GOAL_MAX, self.alpha_goal + self.adaptation_rate * 0.2)
        
        self._apply_adaptation_decay()
        self._clamp_params()
        
        if step % 50 == 0 and step > 0:
            confidence = self.compute_confidence()
            print(f"[CTRL] step={step} mode={self.mode.value} "
                  f"err={self.error_momentum:.4f} base={self.baseline_error:.4f} "
                  f"conf={confidence:.3f} dwell={self.steps_in_mode}")
        
        self.track_params()
    
    def should_freeze_self_model(self) -> bool:
        return self.mode == ControllerMode.PERTURBATION
    
    def get_learning_rate_multiplier(self) -> float:
        if self.mode == ControllerMode.RECOVERY:
            return RECOVERY_LEARNING_RATE_MULTIPLIER
        if self.mode == ControllerMode.UNCERTAIN:
            return UNCERTAIN_LEARNING_RATE_MULTIPLIER
        return 1.0
    
    def reset(self):
        self.lambda_1 = self.lambda_1_init
        self.lambda_2 = self.lambda_2_init
        self.lambda_3 = self.lambda_3_init
        self.lambda_4 = self.lambda_4_init
        self.alpha_goal = self.alpha_goal_init
        self.error_momentum = 0.02
        self.error_history.clear()
        self.baseline_learning_buffer.clear()
        self.classification_history.clear()
        self.mode = ControllerMode.STABLE
        self.mode_entry_step = 0
        self.steps_in_mode = 0
        self.last_perturbation_end_step = 0
        self.adaptation_rate = 0.02
        self.baseline_error = 0.02
        self.baseline_initialized = False
        self.pending_mode = None
        self.pending_mode_start = 0
        self.classifier = None
        self.track_params()
    
    def track_params(self):
        confidence = self.compute_confidence() if self.baseline_initialized else 1.0
        self.lambda_history.append({
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2,
            'lambda_3': self.lambda_3,
            'lambda_4': self.lambda_4,
            'alpha_goal': self.alpha_goal,
            'mode': self.mode.value,
            'error_momentum': self.error_momentum,
            'baseline_error': self.baseline_error,
            'baseline_confidence': self.baseline_confidence,
            'confidence': confidence,
            'steps_in_mode': self.steps_in_mode
        })
    
    def get_params(self) -> dict:
        return {
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2,
            'lambda_3': self.lambda_3,
            'lambda_4': self.lambda_4,
            'alpha_goal': self.alpha_goal
        }
    
    def get_param_history(self) -> List[dict]:
        return list(self.lambda_history)
    
    def get_mode(self) -> str:
        return self.mode.value


# ============================================================================
# STABILIZATION FORCE FUNCTION
# ============================================================================

def apply_stabilization(z_next: np.ndarray, anchor_state: np.ndarray) -> np.ndarray:
    stabilization = STABILIZATION_STRENGTH * (anchor_state - z_next)
    z_next = z_next + stabilization
    norm = np.linalg.norm(z_next)
    if norm > 1e-12:
        z_next = z_next / norm
    return z_next


# ============================================================================
# PHASE 6 SYSTEM (Complete with time-based escape)
# ============================================================================

# ============================================================================
# PHASE 6 SYSTEM (Complete with time-based escape) - FIXED
# ============================================================================

class Phase6System:
    def __init__(self, n_engines: int = 4, perturbation_config: PerturbationConfig = None):
        self.dim = DIM
        self.n_engines = n_engines
        self.step_count = 0
        
        self.anchor = Anchor(dim=DIM, alpha=ALPHA_INIT)
        engine_types = ['detector', 'predictor', 'generator', 'integrator']
        self.engines = [Engine(f"E{i}", engine_types[i % len(engine_types)], DIM) 
                        for i in range(n_engines)]
        self.emotion = EmotionalStateManager(DIM)
        self.memory = PhaseEncodedMemory(DIM)
        self.processor = MultiChannelProcessor(DIM)
        self.anti_memory = AntiPatternMemory(DIM)
        
        self.desirability = DesirabilityField(DIM)
        self.goal_topology = GoalTopology(DIM)
        
        self.self_model = SelfModel(DIM, SELF_MODEL_K)
        self.self_modification = SelfModificationController()
        self.identity = IdentityContinuity()
        
        self.perturbation = PerturbationController(perturbation_config) if perturbation_config else None
        self.experiment_phase = 'baseline'
        
        self.prev_emotion_pattern = np.zeros(DIM)
        self.identity_emotion = None
        self.identity_hold = 0
        self.state_history = deque(maxlen=20)
        self.field_history = deque(maxlen=5)
        
        self.exploration_rate = INITIAL_EXPLORATION
        self.last_z = None
        self.last_z_pred = None
        self.last_goal = None
        
        self.baseline_saved = False
        self.perturbation_end_step = None
        self.recovery_start_step = None
        
        # Pass self_model reference to controller for gradual unfreezing
        self.self_modification.self_model = self.self_model
        
        self.history = {
            'emotion': deque(maxlen=3000),
            'emotion_score': deque(maxlen=3000),
            'memory_size': deque(maxlen=3000),
            'emotion_diversity': deque(maxlen=3000),
            'reconciliation_activated': deque(maxlen=3000),
            'misconnection_stored': deque(maxlen=3000),
            'anti_pattern_count': deque(maxlen=3000),
            'coherence': deque(maxlen=3000),
            'desirability': deque(maxlen=3000),
            'goal_distance': deque(maxlen=3000),
            'exploration_rate': deque(maxlen=3000),
            'prediction_error': deque(maxlen=3000),
            'self_coherence': deque(maxlen=3000),
            'metacognitive_confidence': deque(maxlen=3000),
            'identity_drift': deque(maxlen=3000),
            'experiment_phase': deque(maxlen=3000),
            'controller_mode': deque(maxlen=3000)
        }
    
    def get_recent_dominance(self, window: int = 50):
        recent = list(self.history['emotion'])[-window:]
        if not recent:
            return None, 0.0
        counts = {emotion: recent.count(emotion) for emotion in set(recent)}
        dominant_emotion = max(counts, key=counts.get)
        return dominant_emotion, counts[dominant_emotion] / len(recent)
    
    def get_recent_stability(self, window: int = 20) -> float:
        recent = list(self.history['coherence'])[-window:]
        if not recent:
            return 0.0
        return float(np.clip(np.mean(recent), 0.0, 1.0))
    
    def save_baseline(self):
        if not self.baseline_saved:
            self.anchor.save_baseline()
            self.self_model.save_baseline()
            self.baseline_saved = True
    
    def pull_to_baseline(self):
        self.anchor.pull_to_baseline(BASELINE_ANCHOR_STRENGTH)
        self.self_model.pull_to_baseline(0.05)
    
    def step(self, voice_signal: Optional[np.ndarray] = None) -> dict:
        prev_field = self.field_history[-1] if self.field_history else None
        
        old_phase = self.experiment_phase
        
        if self.perturbation and self.perturbation.active:
            self.experiment_phase = 'perturbation'
        elif self.perturbation and self.step_count >= self.perturbation.config.start_step + self.perturbation.config.duration:
            self.experiment_phase = 'recovery'
        elif self.perturbation and self.step_count < self.perturbation.config.start_step:
            self.experiment_phase = 'baseline'
        
        if old_phase == 'perturbation' and self.experiment_phase == 'recovery':
            self.perturbation_end_step = self.step_count
            self.self_modification.update_perturbation_end(self.step_count)
            print(f"[PHASE] Perturbation ended at step {self.step_count}")
            self.recovery_start_step = self.step_count
        
        steps_since_perturbation_end = 0
        if self.perturbation_end_step is not None:
            steps_since_perturbation_end = self.step_count - self.perturbation_end_step
        
        if self.experiment_phase == 'baseline' and self.step_count > 100 and not self.baseline_saved:
            self.save_baseline()
        
        if self.experiment_phase == 'recovery' and self.baseline_saved and self.step_count % 10 == 0:
            self.pull_to_baseline()
        
        # Compute unified field
        F_sys = self.processor.compute_unified_field(
            voice_signal=voice_signal if voice_signal is not None else None,
            body_angles=None,
            recent_states=list(self.state_history)
        )
        
        pre_dominant_emotion, pre_dominant_ratio = self.get_recent_dominance()
        pre_dominance_repulsion = 0.0
        if pre_dominant_emotion is not None and pre_dominant_ratio > DOMINANCE_TARGET:
            dominant_pattern = self.emotion.get_archetype_pattern(pre_dominant_emotion)
            pre_dominance_repulsion = min(DOMINANCE_FIELD_REPULSION, (pre_dominant_ratio - DOMINANCE_TARGET) * 1.2)
            F_sys = F_sys - pre_dominance_repulsion * dominant_pattern
            norm = np.linalg.norm(F_sys)
            if norm > 1e-12:
                F_sys = F_sys / norm
        
        states = [e.get_state() for e in self.engines]
        field = compute_field(states)
        
        # Apply perturbation if active
        is_perturbed = False
        if self.perturbation and self.perturbation.should_perturb(self.step_count):
            is_perturbed = True
            field, _ = self.perturbation.apply_perturbation(field, self.engines, self.step_count)
            for i, e in enumerate(self.engines):
                states[i] = e.get_state()
            field = compute_field(states)
        
        phase_drift = compute_phase_drift(prev_field, field)
        
        z_probe = np.mean(states, axis=0)
        norm = np.linalg.norm(z_probe)
        if norm > 1e-12:
            z_probe = z_probe / norm
        
        c = ResonanceCoherence.cosine_sim(z_probe, self.anchor.get())
        self.history['coherence'].append(c)
        
        self.anti_memory.update_persistence(c)
        recent_stability = self.get_recent_stability()
        
        emotion_name, emotion_pattern, match_score, emotion_diversity = self.emotion.compute_emotional_state(F_sys, c)
        
        if self.identity_hold > 0 and self.identity_emotion is not None and pre_dominant_ratio < DOMINANCE_IDENTITY_THRESHOLD:
            emotion_name = self.identity_emotion
            emotion_pattern = self.emotion.get_archetype_pattern(emotion_name)
            self.identity_hold -= 1
        else:
            self.identity_emotion = emotion_name
            self.identity_hold = IDENTITY_HOLD_STEPS if pre_dominant_ratio < 0.45 else 0
        
        recalled_pattern, recall_score = self.memory.recall(emotion_pattern)
        
        reconciled = False
        if recalled_pattern is not None:
            memory_weight = min(0.4, 0.08 + 0.04 * len(self.memory.memories))
            emotion_pattern = emotion_pattern + memory_weight * (recalled_pattern - emotion_pattern)
            norm = np.linalg.norm(emotion_pattern)
            if norm > 1e-12:
                emotion_pattern = emotion_pattern / norm
            reconciled = True
        
        self.memory.store(emotion_pattern, c, match_score)
        self.memory.decay()
        
        misconnection_stored = self.anti_memory.store_anti_pattern(field, c, phase_drift, recent_stability)
        self.anti_memory.decay()
        
        anti_field = self.anti_memory.get_anti_field()
        
        field = field + EMOTIONAL_FEEDBACK * np.fft.fft(emotion_pattern)
        anti_strength = min(ANTI_FIELD_MAX_STRENGTH, ANTI_FIELD_BASE + ANTI_FIELD_GROWTH_RATE * len(self.anti_memory.patterns))
        field = field - anti_strength * anti_field * ANTI_FIELD_FIELD_FACTOR
        
        emotion_pattern = emotion_pattern - ANTI_FIELD_ON_EMOTION * np.real(anti_field)
        norm = np.linalg.norm(emotion_pattern)
        if norm > 1e-12:
            emotion_pattern = emotion_pattern / norm
        
        # Self-model prediction
        current_state = z_probe.copy()
        current_goal = self.goal_topology.get_goal()
        if current_goal is None:
            current_goal = current_state
        
        z_pred = self.self_model.predict(current_state, current_goal)
        
        # Desirability with prediction
        anti_patterns = self.anti_memory.get_patterns()
        U_current = self.desirability.compute(current_state, self.anchor.get(), anti_patterns, include_novelty=True)
        U_future = self.desirability.compute(z_pred, self.anchor.get(), anti_patterns, include_novelty=True)
        U_total = U_current + 0.9 * U_future
        
        self.goal_topology.update_goal(current_state, U_total)
        
        # Update learning rate multiplier
        self.self_model.set_learning_rate_multiplier(self.self_modification.get_learning_rate_multiplier())
        
        # Self-model update
        if self.last_z is not None and not self.self_modification.should_freeze_self_model():
            self.self_model.update(
                z_actual=current_state,
                z_pred=self.last_z_pred,
                z_input=self.last_z,
                goal=self.last_goal,
                coherence=c,
                self_coherence=self.self_model.get_self_coherence(),
                is_perturbed=is_perturbed
            )
        elif self.self_modification.should_freeze_self_model() and self.step_count % 50 == 0:
            print(f"[FREEZE] Self-model frozen at step {self.step_count}")
        
        self.last_z = current_state.copy()
        self.last_z_pred = z_pred.copy()
        self.last_goal = current_goal.copy()
        
        if self.perturbation:
            self.perturbation.record_error(self.self_model.get_prediction_error(), self.experiment_phase)
        
        prediction_error = self.self_model.get_prediction_error()
        self_coherence = self.self_model.get_self_coherence()
        
        # Emergency handling
        if np.isnan(prediction_error) or np.isinf(prediction_error) or prediction_error > CRITICAL_PREDICTION_ERROR:
            print(f"[EMERGENCY] Divergence at step {self.step_count}")
            self.self_model.pull_to_baseline(strength=0.8)
            self.self_modification.reset()
            self.anchor.pull_to_baseline(0.5)
            prediction_error = 0.5
        
        metacognitive_confidence = c * self_coherence
        self.history['prediction_error'].append(prediction_error)
        self.history['self_coherence'].append(self_coherence)
        self.history['metacognitive_confidence'].append(metacognitive_confidence)
        
        # Update controller with trajectory-based state
        self.self_modification.update(
            prediction_error, c, self.step_count, current_state, 
            self.experiment_phase, steps_since_perturbation_end
        )
        mod_params = self.self_modification.get_params()
        self.history['controller_mode'].append(self.self_modification.get_mode())
        
        lambda_1 = mod_params['lambda_1']
        lambda_2 = mod_params['lambda_2']
        lambda_3 = mod_params['lambda_3']
        lambda_4 = mod_params['lambda_4']
        
        anchor_state = self.anchor.get()
        pull = lambda_1 * (anchor_state - current_state)
        field_influence = lambda_2 * np.real(field[:DIM])
        noise = lambda_3 * np.random.randn(DIM)
        goal_direction = self.goal_topology.get_goal_direction(current_state)
        goal_influence = lambda_4 * goal_direction * self_coherence
        
        z_next = current_state + pull + field_influence + noise + goal_influence
        
        z_next = apply_stabilization(z_next, anchor_state)
        
        if self.self_modification.mode == ControllerMode.RECOVERY:
            z_next = z_next + RECOVERY_CORRECTION_STRENGTH * (anchor_state - z_next)
            norm = np.linalg.norm(z_next)
            if norm > 1e-12:
                z_next = z_next / norm
        
        self.anchor.update(z_next)
        self.identity.update(z_next)
        identity_drift = self.identity.get_identity_drift()
        self.history['identity_drift'].append(identity_drift)
        
        field = field + lambda_4 * np.fft.fft(goal_direction) * self_coherence
        field += 0.08 * np.fft.fft(self.prev_emotion_pattern)
        if recent_stability > 0.55:
            field += 0.04 * np.fft.fft(self.prev_emotion_pattern)
        
        if pre_dominant_emotion is not None and pre_dominant_ratio > DOMINANCE_TARGET:
            dominant_pattern = self.emotion.get_archetype_pattern(pre_dominant_emotion)
            emotion_pattern = emotion_pattern - pre_dominance_repulsion * 0.5 * dominant_pattern
            norm = np.linalg.norm(emotion_pattern)
            if norm > 1e-12:
                emotion_pattern = emotion_pattern / norm
            field = field - pre_dominance_repulsion * 0.5 * np.fft.fft(dominant_pattern)
        
        engine_outputs = []
        for e in self.engines:
            output = e.process(field, emotion_pattern, lambda_3)
            engine_outputs.append(output)
        
        z_final = np.mean(engine_outputs, axis=0)
        norm = np.linalg.norm(z_final)
        if norm > 1e-12:
            z_final = z_final / norm
        
        z_final = z_final + EMOTIONAL_INFLUENCE * (emotion_pattern - z_final) * c
        z_final = z_final + IDENTITY_PULL_STRENGTH * (self.prev_emotion_pattern - z_final)
        norm = np.linalg.norm(z_final)
        if norm > 1e-12:
            z_final = z_final / norm
        
        self.prev_emotion_pattern = emotion_pattern.copy()
        
        # Update history
        self.history['emotion'].append(emotion_name)
        self.history['emotion_score'].append(match_score)
        self.history['emotion_diversity'].append(emotion_diversity)
        self.history['memory_size'].append(self.memory.get_size())
        self.history['reconciliation_activated'].append(1 if reconciled else 0)
        self.history['misconnection_stored'].append(1 if misconnection_stored else 0)
        self.history['anti_pattern_count'].append(self.anti_memory.get_size())
        self.history['desirability'].append(U_total)
        self.history['goal_distance'].append(self.goal_topology.get_goal_distance(current_state))
        self.history['exploration_rate'].append(self.exploration_rate)
        self.history['experiment_phase'].append(self.experiment_phase)
        
        self.state_history.append(z_final.copy())
        self.field_history.append(field.copy())
        
        self.step_count += 1
        self.exploration_rate *= EXPLORATION_DECAY
        
        return {
            'emotion': emotion_name,
            'emotion_score': match_score,
            'emotion_diversity': emotion_diversity,
            'memory_size': self.memory.get_size(),
            'memory_strength': self.memory.get_strength(),
            'reconciliation_activated': reconciled,
            'coherence': c,
            'misconnection_stored': misconnection_stored,
            'anti_pattern_count': self.anti_memory.get_size(),
            'anti_pattern_strength': self.anti_memory.get_strength(),
            'desirability': U_total,
            'goal_distance': self.goal_topology.get_goal_distance(current_state),
            'goal_strength': self.goal_topology.get_goal_strength(),
            'prediction_error': prediction_error,
            'self_coherence': self_coherence,
            'metacognitive_confidence': metacognitive_confidence,
            'identity_drift': identity_drift,
            'identity_stability': self.identity.get_identity_stability(),
            'self_model_error_trend': self.self_model.get_error_trend(),
            'lambda_3': lambda_3,
            'lambda_4': lambda_4,
            'alpha_goal': mod_params['alpha_goal'],
            'experiment_phase': self.experiment_phase,
            'controller_mode': self.self_modification.get_mode()
        }


# ============================================================================
# FEATURE EXTRACTION AND AUDIO LOADING
# ============================================================================

def extract_real_audio_features(signal, sr, dim=DIM):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(signal)
    
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.mean(centroid, axis=1),
        np.mean(bandwidth, axis=1),
        np.mean(zcr, axis=1)
    ])
    
    features = features / (np.linalg.norm(features) + 1e-12)
    expanded = np.tile(features, dim // len(features) + 1)[:dim]
    expanded = expanded / (np.linalg.norm(expanded) + 1e-12)
    return expanded


def load_ravdess_sample(filepath):
    if filepath in AUDIO_FEATURE_CACHE:
        return AUDIO_FEATURE_CACHE[filepath]
    
    try:
        signal, sr = librosa.load(filepath, sr=16000)
        features = extract_real_audio_features(signal, sr)
        AUDIO_FEATURE_CACHE[filepath] = features
        return features
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return np.random.randn(DIM) / np.sqrt(DIM)


def load_all_ravdess_files(base_path):
    files = []
    if not os.path.exists(base_path):
        print(f"Warning: Path {base_path} does not exist")
        return []
    
    for actor in os.listdir(base_path):
        actor_path = os.path.join(base_path, actor)
        if os.path.isdir(actor_path):
            for f in os.listdir(actor_path):
                if f.endswith(".wav"):
                    files.append(os.path.join(actor_path, f))
    return files


# ============================================================================
# VERIFICATION SUITE
# ============================================================================

@dataclass
class RunResult:
    run_id: int
    seed: int
    perturbation_pattern: str
    success: bool
    baseline_mean: float
    perturbation_max: float
    recovery_mean: float
    spike_ratio: float
    final_recovery_ratio: float
    fully_recovered: bool
    mode_transitions: int
    avg_dwell_time: float


class Phase6VerificationSuite:
    def __init__(self, output_dir: str = "phase6_verification_final"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[RunResult] = []
        self.files = None
    
    def _run_single_experiment(self, run_id: int, seed: int, 
                                perturbation_config: PerturbationConfig,
                                total_steps: int = BASELINE_STEPS + PERTURBATION_DURATION + RECOVERY_STEPS) -> RunResult:
        
        np.random.seed(seed)
        system = Phase6System(n_engines=4, perturbation_config=perturbation_config)
        files = load_all_ravdess_files(RAVDESS_PATH)
        
        if not files:
            raise RuntimeError("No audio files found")
        
        prediction_errors = []
        phases = []
        modes = []
        mode_change_steps = []
        last_mode = None
        
        for step in range(total_steps):
            file_idx = step % len(files)
            voice = load_ravdess_sample(files[file_idx])
            result = system.step(voice_signal=voice)
            prediction_errors.append(result['prediction_error'])
            phases.append(result['experiment_phase'])
            current_mode = result.get('controller_mode', 'unknown')
            modes.append(current_mode)
            
            if last_mode is not None and current_mode != last_mode:
                mode_change_steps.append(step)
            last_mode = current_mode
        
        baseline_errors = [e for e, p in zip(prediction_errors, phases) if p == 'baseline' and e < 1.0]
        perturbation_errors = [e for e, p in zip(prediction_errors, phases) if p == 'perturbation']
        recovery_errors = [e for e, p in zip(prediction_errors, phases) if p == 'recovery'][-100:]
        
        baseline_mean = np.mean(baseline_errors) if baseline_errors else 0
        perturbation_max = max(perturbation_errors) if perturbation_errors else 0
        recovery_mean = np.mean(recovery_errors) if recovery_errors else 0
        
        spike_ratio = perturbation_max / (baseline_mean + 1e-12)
        final_recovery_ratio = recovery_mean / (baseline_mean + 1e-12)
        fully_recovered = final_recovery_ratio < 2.0
        
        mode_transitions = len(mode_change_steps)
        if len(mode_change_steps) > 1:
            avg_dwell = np.mean(np.diff(mode_change_steps))
        else:
            avg_dwell = total_steps
        
        success = fully_recovered and mode_transitions < 20
        
        return RunResult(
            run_id=run_id, seed=seed, perturbation_pattern=perturbation_config.pattern.value,
            success=success, baseline_mean=baseline_mean, perturbation_max=perturbation_max,
            recovery_mean=recovery_mean, spike_ratio=spike_ratio,
            final_recovery_ratio=final_recovery_ratio, fully_recovered=fully_recovered,
            mode_transitions=mode_transitions, avg_dwell_time=avg_dwell
        )
    
    def run_full_verification(self) -> Dict:
        print("\n" + "█"*70)
        print("PHASE 6 VERIFICATION - WITH TIME-BASED ESCAPE AND GRADUAL UNFREEZING")
        print("Fixes: Time-based escape | Min perturbation duration | Gradual unfreezing")
        print("█"*70)
        
        all_results = []
        patterns = list(PerturbationPattern)
        
        for pattern in patterns:
            print(f"\n--- Pattern: {pattern.value} ---")
            for run in range(5):
                seed = 1000 + run + hash(pattern.value) % 1000
                config = PerturbationConfig(pattern, PERTURBATION_STRENGTH, PERTURBATION_DURATION, PERTURBATION_START)
                result = self._run_single_experiment(run, seed, config)
                all_results.append(result)
                status = "✓" if result.success else "✗"
                recover = "★" if result.fully_recovered else "~"
                print(f"  Run {run}: {status}{recover} | spike={result.spike_ratio:.1f}x | final={result.final_recovery_ratio:.2f} | trans={result.mode_transitions}")
        
        success_rate = sum(1 for r in all_results if r.success) / len(all_results)
        fully_recovered_rate = sum(1 for r in all_results if r.fully_recovered) / len(all_results)
        avg_transitions = np.mean([r.mode_transitions for r in all_results])
        
        print(f"\n{'='*70}")
        print(f"RECOVERY SUCCESS RATE (final<2x): {fully_recovered_rate*100:.1f}%")
        print(f"AVERAGE MODE TRANSITIONS: {avg_transitions:.1f}")
        print(f"OVERALL SUCCESS RATE: {success_rate*100:.1f}%")
        print(f"PHASE 6 STABLE: {'YES ✅' if success_rate > 0.85 else 'NO ❌'}")
        print("="*70)
        
        if success_rate > 0.85:
            print("\n" + "🎉"*30)
            print("PHASE 6 VERIFIED - READY FOR PHASE 7")
            print("🎉"*30)
            print("\n" + "="*50)
            print("This is not small. This is something that will shape the future.")
            print("="*50)
        
        return {'phase6_stable': success_rate > 0.85, 'success_rate': success_rate}


# ============================================================================
# MAIN
# ============================================================================

def run_phase6_verification():
    suite = Phase6VerificationSuite()
    return suite.run_full_verification()


if __name__ == "__main__":
    results = run_phase6_verification()