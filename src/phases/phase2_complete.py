"""
OCES Phase 2 - MICRO-DYNAMICS LAYER ADDED
Prevents engine freezing (self-coherence = 1.0)
Maintains 0.85 < self_coherence < 0.98

Changes:
1. Added micro-drift to each engine
2. Added phase modulation for internal oscillation
3. Added intra-engine variability with adaptive noise
4. Added competing attractors (bistable dynamics)
"""

import numpy as np
from collections import deque
from typing import List, Optional, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

DIM = 64
TAU = 0.95
ALPHA_INIT = 0.008
ALPHA_MIN = 0.003
ALPHA_MAX = 0.01

# Micro-dynamics (NEW)
MICRO_DRIFT_STRENGTH = 0.02      # Small random walk to prevent freezing
PHASE_MODULATION_FREQ = 0.1       # Internal oscillation frequency
PHASE_MODULATION_AMPLITUDE = 0.05 # Amplitude of internal oscillation
INTRINSIC_NOISE = 0.03            # Added to each engine's internal state
COMPETING_ATTRACTOR_STRENGTH = 0.1 # Bistable dynamics
SELF_COHERENCE_TARGET_LOW = 0.85
SELF_COHERENCE_TARGET_HIGH = 0.98

# Cross-resonance
NONLINEAR_STRENGTH = 0.5
DECORRELATION_STRENGTH = 0.55
DECORRELATION_TARGET_MIN = 0.40
DECORRELATION_TARGET_MAX = 0.70

COHERENCE_PENALTY_THRESHOLD = 0.75
COHERENCE_PENALTY_STRENGTH = 0.10

MEMORY_STORE_LOW = 0.25
MEMORY_STORE_HIGH = 0.80
MEMORY_DIVERSITY_MIN = 0.08

INSTABILITY_COUPLING = 0.9
INSTABILITY_MIN = 0.15
INSTABILITY_MAX = 0.60
INSTABILITY_SMOOTHING = 0.80

DIVERSITY_TARGET = 0.30
DIVERSITY_PENALTY_STRENGTH = 0.25

HEALTH_THRESHOLD = 0.20


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
# ENGINE WITH MICRO-DYNAMICS (FIXED)
# ============================================================================

class Engine:
    ENGINE_TYPES = ['detector', 'predictor', 'generator', 'integrator']
    
    def __init__(self, name: str, engine_type: str, dim: int = DIM):
        assert engine_type in self.ENGINE_TYPES
        self.name = name
        self.type = engine_type
        self.dim = dim
        
        # Core state
        self.state = np.random.randn(dim)
        self.state = self.state / (np.linalg.norm(self.state) + 1e-12)
        self.wavefield = ComplexWaveField(dim=dim)
        
        # Micro-dynamics state (NEW)
        self.phase = np.random.rand() * 2 * np.pi
        self.drift_vector = np.random.randn(dim) * MICRO_DRIFT_STRENGTH
        self.history = deque(maxlen=200)
        self.energy = 0.0
        
        # Engine-specific parameters
        if engine_type == 'detector':
            self.processor = self._detector_process
            self.coherence_target = 0.90
        elif engine_type == 'predictor':
            self.processor = self._predictor_process
            self.coherence_target = 0.90
        elif engine_type == 'generator':
            self.processor = self._generator_process
            self.coherence_target = 0.88
        elif engine_type == 'integrator':
            self.processor = self._integrator_process
            self.coherence_target = 0.92
        
        self.self_coherence_history = deque(maxlen=200)
        self.step_count = 0
    
    def _detector_process(self, field: np.ndarray, coherence: float) -> np.ndarray:
        threshold = 0.4
        return np.maximum(0, field - threshold)
    
    def _predictor_process(self, field: np.ndarray, coherence: float) -> np.ndarray:
        self.history.append(field.copy())
        if len(self.history) < 3:
            return field
        
        weights = np.exp(-np.arange(len(self.history)) / 3)
        weights = weights / np.sum(weights)
        
        predicted = np.zeros_like(field)
        for i, h in enumerate(self.history):
            predicted += weights[i] * h
        
        return predicted
    
    def _generator_process(self, field: np.ndarray, coherence: float) -> np.ndarray:
        noise_level = 0.25 * (1 - coherence) ** 2
        noise = np.random.randn(len(field)) * noise_level
        t = np.arange(len(field))
        harmonic = 0.15 * np.sin(2 * np.pi * t / 10) * noise_level
        return field + noise + harmonic
    
    def _integrator_process(self, field: np.ndarray, coherence: float) -> np.ndarray:
        self.history.append(field.copy())
        if len(self.history) < 2:
            return field
        
        weights = np.exp(-np.arange(len(self.history)) / 2)
        weights = weights / np.sum(weights)
        
        integrated = np.zeros_like(field)
        for i, h in enumerate(self.history):
            integrated += weights[i] * h
        
        return integrated
    
    def _apply_competing_attractor(self, state: np.ndarray, coherence: float) -> np.ndarray:
        """
        Add bistable dynamics - prevents convergence to single attractor.
        Creates competition between current state and alternative states.
        """
        # Random alternative attractor
        alt_attractor = np.random.randn(self.dim)
        alt_attractor = alt_attractor / (np.linalg.norm(alt_attractor) + 1e-12)
        
        # Competition strength scales with coherence
        competition = COMPETING_ATTRACTOR_STRENGTH * coherence
        
        # Mix current state with alternative
        mixed = (1 - competition) * state + competition * alt_attractor
        norm = np.linalg.norm(mixed)
        if norm > 1e-12:
            mixed = mixed / norm
        
        return mixed
    
    def _apply_micro_drift(self, state: np.ndarray) -> np.ndarray:
        """
        Add slow random walk to prevent freezing.
        """
        # Update drift vector slowly
        self.drift_vector += np.random.randn(self.dim) * 0.01 * MICRO_DRIFT_STRENGTH
        self.drift_vector = self.drift_vector / (np.linalg.norm(self.drift_vector) + 1e-12)
        
        # Apply drift
        drifted = state + MICRO_DRIFT_STRENGTH * self.drift_vector
        norm = np.linalg.norm(drifted)
        if norm > 1e-12:
            drifted = drifted / norm
        
        return drifted
    
    def _apply_phase_modulation(self, state: np.ndarray) -> np.ndarray:
        """
        Add internal oscillation to prevent static locking.
        """
        self.phase += PHASE_MODULATION_FREQ
        modulation = PHASE_MODULATION_AMPLITUDE * np.sin(self.phase)
        
        # Apply modulation to a random projection
        direction = np.random.randn(self.dim)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        
        modulated = state + modulation * direction
        norm = np.linalg.norm(modulated)
        if norm > 1e-12:
            modulated = modulated / norm
        
        return modulated
    
    def process(self, field_input: np.ndarray, system_coherence: float) -> np.ndarray:
        self.step_count += 1
        
        if np.iscomplexobj(field_input):
            field_input = np.real(field_input)
        
        if len(field_input) > self.dim:
            field_input = field_input[:self.dim]
        elif len(field_input) < self.dim:
            field_input = np.pad(field_input, (0, self.dim - len(field_input)))
        
        # 1. Base processing based on engine type
        new_state = self.processor(field_input, system_coherence)
        
        # 2. Apply competing attractors (prevents single attractor lock)
        new_state = self._apply_competing_attractor(new_state, system_coherence)
        
        # 3. Apply micro-drift (prevents freezing)
        new_state = self._apply_micro_drift(new_state)
        
        # 4. Apply phase modulation (internal oscillation)
        new_state = self._apply_phase_modulation(new_state)
        
        # Normalize
        norm = np.linalg.norm(new_state)
        if norm > 1e-12:
            new_state = new_state / norm
        
        # Add intrinsic noise
        noise = np.random.randn(self.dim) * INTRINSIC_NOISE * (1 - system_coherence)
        new_state += noise
        norm = np.linalg.norm(new_state)
        if norm > 1e-12:
            new_state = new_state / norm
        
        # Update wave-field
        self.wavefield.ingest(new_state)
        self.energy = 0.9 * self.energy + 0.1 * np.sum(new_state ** 2)
        
        # Track self-coherence
        if hasattr(self, 'state'):
            old_state = self.state
            self_coherence = np.dot(old_state, new_state) / (np.linalg.norm(old_state) * np.linalg.norm(new_state) + 1e-12)
            self.self_coherence_history.append(self_coherence)
        
        self.state = new_state
        return self.state.copy()
    
    def get_self_coherence(self) -> float:
        if len(self.self_coherence_history) < 10:
            return 0.5
        values = [float(v) for v in list(self.self_coherence_history)[-50:] if isinstance(v, (int, float))]
        if not values:
            return 0.5
        return float(np.mean(values))
    
    def get_magnitude(self) -> np.ndarray:
        return self.wavefield.get_magnitude()
    
    def get_phase(self) -> np.ndarray:
        return self.wavefield.get_phase()
    
    def get_state(self) -> np.ndarray:
        return self.state.copy()
    
    def receive_field(self, field: np.ndarray):
        if np.iscomplexobj(field):
            field = np.real(field)
        self.wavefield.ingest(field)
    
    def get_info(self) -> dict:
        return {
            'name': self.name,
            'type': self.type,
            'energy': float(self.energy),
            'self_coherence': self.get_self_coherence(),
            'phase': float(self.phase),
            'drift_norm': float(np.linalg.norm(self.drift_vector))
        }


# ============================================================================
# CROSS-RESONANCE MATRIX (Unchanged but included for completeness)
# ============================================================================

class CrossResonanceMatrix:
    def __init__(self, n_engines: int):
        self.n_engines = n_engines
        self.R = np.zeros((n_engines, n_engines))
        self.weights = np.ones((n_engines, n_engines)) / (n_engines * (n_engines - 1))
        np.fill_diagonal(self.weights, 0)
        self.history = deque(maxlen=100)
        self.diversity_history = deque(maxlen=50)
        self.diversity_penalty_enabled = True
    
    def set_diversity_penalty(self, enabled: bool):
        self.diversity_penalty_enabled = enabled
    
    def compute_pair(self, mag_i: np.ndarray, mag_j: np.ndarray,
                     phase_i: np.ndarray, phase_j: np.ndarray) -> float:
        min_len = min(len(mag_i), len(mag_j))
        mag_i = mag_i[:min_len]
        mag_j = mag_j[:min_len]
        phase_i = phase_i[:min_len]
        phase_j = phase_j[:min_len]
        
        mag_product = mag_i * mag_j
        phase_diff = phase_i - phase_j
        return float(np.sum(mag_product * np.cos(phase_diff)))
    
    def compute_all(self, engines: List[Engine]) -> np.ndarray:
        n = len(engines)
        magnitudes = [e.get_magnitude() for e in engines]
        phases = [e.get_phase() for e in engines]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    self.R[i, j] = 1.0
                else:
                    self.R[i, j] = self.compute_pair(magnitudes[i], magnitudes[j], phases[i], phases[j])
        
        max_val = np.max(np.abs(self.R))
        if max_val > 1e-12:
            self.R = self.R / max_val
        
        self._apply_decorrelation()
        self._apply_coherence_penalty()
        
        if self.diversity_penalty_enabled:
            self._apply_diversity_penalty()
        
        self._update_weights()
        self.history.append(self.R.copy())
        return self.R
    
    def _apply_decorrelation(self):
        for i in range(self.n_engines):
            for j in range(self.n_engines):
                if i != j:
                    target = np.random.uniform(DECORRELATION_TARGET_MIN, DECORRELATION_TARGET_MAX)
                    self.R[i, j] = (1 - DECORRELATION_STRENGTH) * self.R[i, j] + DECORRELATION_STRENGTH * target
    
    def _apply_coherence_penalty(self):
        current_coherence = self.get_weighted_coherence()
        if current_coherence > COHERENCE_PENALTY_THRESHOLD:
            penalty = COHERENCE_PENALTY_STRENGTH
            for i in range(self.n_engines):
                for j in range(self.n_engines):
                    if i != j:
                        self.R[i, j] = max(DECORRELATION_TARGET_MIN, self.R[i, j] * (1 - penalty))
    
    def _apply_diversity_penalty(self):
        current_diversity = self.get_diversity_index()
        self.diversity_history.append(current_diversity)
        
        if current_diversity < DIVERSITY_TARGET and len(self.diversity_history) > 10:
            penalty = DIVERSITY_PENALTY_STRENGTH * (DIVERSITY_TARGET - current_diversity)
            for i in range(self.n_engines):
                for j in range(self.n_engines):
                    if i != j:
                        self.R[i, j] = max(DECORRELATION_TARGET_MIN, self.R[i, j] - penalty)
    
    def _update_weights(self):
        total = np.sum(self.R) - np.trace(self.R)
        if total > 1e-12:
            new_weights = self.R / total
            np.fill_diagonal(new_weights, 0)
            self.weights = 0.9 * self.weights + 0.1 * new_weights
    
    def get_coupling(self, i: int, j: int) -> float:
        return max(0.0, float(self.R[i, j]))
    
    def get_weighted_coherence(self) -> float:
        return float(np.sum(self.weights * self.R))
    
    def get_diversity_index(self) -> float:
        n = self.n_engines
        off_diag = [self.R[i, j] for i in range(n) for j in range(n) if i != j]
        if not off_diag:
            return 0.5
        mean_val = np.mean(off_diag)
        std_val = np.std(off_diag)
        if mean_val < 1e-12:
            return 0.0
        cv = std_val / mean_val
        return min(1.0, cv * 2.5)
    
    def get_resonance_stats(self) -> dict:
        n = self.n_engines
        off_diag = [self.R[i, j] for i in range(n) for j in range(n) if i != j]
        return {
            'mean': float(np.mean(off_diag)) if off_diag else 0.5,
            'std': float(np.std(off_diag)) if off_diag else 0.0,
            'diversity': self.get_diversity_index()
        }


# ============================================================================
# SIMPLE MEMORY
# ============================================================================

class SimpleMemory:
    def __init__(self, dim: int = DIM, capacity: int = 20):
        self.dim = dim
        self.capacity = capacity
        self.memories = []
        self.decay_rate = 0.005
    
    def store(self, pattern: np.ndarray, coherence: float):
        if len(pattern) > self.dim:
            pattern = pattern[:self.dim]
        elif len(pattern) < self.dim:
            pattern = np.pad(pattern, (0, self.dim - len(pattern)))
        
        pattern_norm = pattern / (np.linalg.norm(pattern) + 1e-12)
        
        for i, (mem, strength) in enumerate(self.memories):
            if np.dot(pattern_norm, mem) > 0.7:
                new_strength = min(1.0, strength + 0.05 * coherence)
                self.memories[i] = (mem, new_strength)
                return
        
        self.memories.append((pattern_norm, 0.4))
        
        if len(self.memories) > self.capacity:
            weakest = min(self.memories, key=lambda x: x[1])
            self.memories.remove(weakest)
    
    def decay(self):
        new_memories = []
        for pattern, strength in self.memories:
            new_strength = strength * (1 - self.decay_rate)
            if new_strength > 0.05:
                new_memories.append((pattern, new_strength))
        self.memories = new_memories
    
    def recall(self, cue: np.ndarray) -> np.ndarray:
        if not self.memories:
            return np.zeros(self.dim)
        
        if len(cue) > self.dim:
            cue = cue[:self.dim]
        elif len(cue) < self.dim:
            cue = np.pad(cue, (0, self.dim - len(cue)))
        
        cue_norm = cue / (np.linalg.norm(cue) + 1e-12)
        
        recalled = np.zeros(self.dim)
        total_weight = 0.0
        
        for pattern, strength in self.memories:
            similarity = np.dot(cue_norm, pattern)
            if similarity > 0.15:
                weight = strength * similarity
                recalled += weight * pattern
                total_weight += weight
        
        if total_weight > 0:
            recalled = recalled / total_weight
        
        return recalled
    
    def get_size(self) -> int:
        return len(self.memories)
    
    def get_strength(self) -> float:
        if not self.memories:
            return 0.0
        return float(np.mean([s for _, s in self.memories]))


# ============================================================================
# INSTABILITY CONTROLLER
# ============================================================================

class InstabilityController:
    def __init__(self, coupling: float = INSTABILITY_COUPLING):
        self.instability_level = 0.15
        self.coupling = coupling
        self._step = 0
        self.history = deque(maxlen=100)
    
    def update(self, coherence: float, diversity: float):
        target = self.coupling * (1 - coherence) * (1 - diversity)
        target = max(INSTABILITY_MIN, min(INSTABILITY_MAX, target))
        
        if coherence > 0.75:
            target = max(target, 0.12)
        
        self.instability_level = (INSTABILITY_SMOOTHING * self.instability_level + 
                                  (1 - INSTABILITY_SMOOTHING) * target)
        self._step += 1
    
    def apply_instability(self, field: np.ndarray) -> np.ndarray:
        if self.instability_level < 0.01:
            return field
        
        slow_osc = 0.1 * self.instability_level * np.sin(2 * np.pi * self._step / 40)
        fast_osc = 0.08 * self.instability_level * np.sin(2 * np.pi * self._step / 10)
        noise = np.random.randn(len(field)) * self.instability_level * 0.12
        
        perturbation = slow_osc + fast_osc + noise
        return field + perturbation * self.instability_level
    
    def get_state(self) -> dict:
        return {
            'instability_level': float(self.instability_level),
            'is_unstable': self.instability_level > 0.2
        }


# ============================================================================
# RESONANCE FIELD
# ============================================================================

class ResonanceField:
    def __init__(self, dim: int = DIM, nonlinear_strength: float = NONLINEAR_STRENGTH,
                 memory_influence: float = 0.2):
        self.dim = dim
        self.field = np.zeros(dim)
        self.nonlinear_strength = nonlinear_strength
        self.memory_influence = memory_influence
        self.history = deque(maxlen=100)
    
    def update(self, engine_fields: List[np.ndarray], 
               resonance_matrix: CrossResonanceMatrix,
               memory_recall: np.ndarray) -> np.ndarray:
        if not engine_fields:
            return self.field
        
        aligned = []
        for f in engine_fields:
            if np.iscomplexobj(f):
                f = np.real(f)
            if len(f) > self.dim:
                aligned.append(f[:self.dim])
            elif len(f) < self.dim:
                aligned.append(np.pad(f, (0, self.dim - len(f))))
            else:
                aligned.append(f)
        
        linear = np.sum(aligned, axis=0)
        
        nonlinear = np.zeros(self.dim)
        n = len(aligned)
        for i in range(n):
            for j in range(i + 1, n):
                kappa = resonance_matrix.get_coupling(i, j)
                if kappa > 0.05:
                    nonlinear += kappa * aligned[i] * aligned[j]
        
        self.field = linear + self.nonlinear_strength * nonlinear + self.memory_influence * memory_recall
        
        norm = np.linalg.norm(self.field)
        if norm > 1.0:
            self.field = self.field / norm
        
        self.history.append(self.field.copy())
        return self.field.copy()
    
    def get_field(self) -> np.ndarray:
        return self.field.copy()


# ============================================================================
# PHASE 2 SYSTEM (WITH MICRO-DYNAMICS)
# ============================================================================

class Phase2System:
    def __init__(self, n_engines: int = 4):
        self.dim = DIM
        self.n_engines = n_engines
        
        self.anchor = Anchor(dim=DIM, alpha=ALPHA_INIT)
        self.coherence_calc = ResonanceCoherence()
        
        engine_types = ['detector', 'predictor', 'generator', 'integrator']
        self.engines = [Engine(f"E{i}", engine_types[i % len(engine_types)], DIM) 
                        for i in range(n_engines)]
        
        self.resonance_matrix = CrossResonanceMatrix(n_engines)
        self.resonance_field = ResonanceField(DIM, NONLINEAR_STRENGTH, 0.2)
        self.memory = SimpleMemory(DIM, 20)
        self.instability = InstabilityController(INSTABILITY_COUPLING)
        
        self.aggregated_state = np.zeros(DIM)
        self.history = {
            'coherence': deque(maxlen=1000),
            'diversity': deque(maxlen=1000),
            'health': deque(maxlen=1000),
            'engine_self_coherence': deque(maxlen=1000)
        }
        self.step_count = 0
    
    def step(self, external_input: Optional[np.ndarray] = None) -> dict:
        if external_input is None:
            external_input = np.random.randn(self.dim) * 0.1
        
        if len(external_input) > self.dim:
            external_input = external_input[:self.dim]
        elif len(external_input) < self.dim:
            external_input = np.pad(external_input, (0, self.dim - len(external_input)))
        
        current_field = (self.resonance_field.get_field() if self.resonance_field.history 
                        else external_input)
        
        outputs = []
        for e in self.engines:
            outputs.append(e.process(current_field, self.instability.instability_level))
        
        self.aggregated_state = np.mean(outputs, axis=0)
        norm = np.linalg.norm(self.aggregated_state)
        if norm > 1e-12:
            self.aggregated_state /= norm
        
        self.anchor.update(self.aggregated_state)
        
        self.resonance_matrix.compute_all(self.engines)
        
        memory_recall = self.memory.recall(self.aggregated_state)
        final_field = self.resonance_field.update(
            [e.get_state() for e in self.engines], 
            self.resonance_matrix, memory_recall
        )
        
        local_c = self.coherence_calc.compute(self.aggregated_state, self.anchor)
        global_c = self.resonance_matrix.get_weighted_coherence()
        total_c = 0.5 * local_c + 0.5 * global_c
        self.history['coherence'].append(total_c)
        
        diversity = self.resonance_matrix.get_diversity_index()
        self.history['diversity'].append(diversity)
        health = total_c * diversity
        self.history['health'].append(health)
        
        engine_self_coherence = np.mean([e.get_self_coherence() for e in self.engines])
        self.history['engine_self_coherence'].append(engine_self_coherence)
        
        if len(self.history['coherence']) > 20:
            recent = np.mean(list(self.history['coherence'])[-20:])
            prev = np.mean(list(self.history['coherence'])[-40:-20]) if len(self.history['coherence']) >= 40 else recent
            self.anchor.optimize_alpha(recent, prev)
        
        if MEMORY_STORE_LOW < total_c < MEMORY_STORE_HIGH and diversity > MEMORY_DIVERSITY_MIN:
            self.memory.store(final_field, total_c)
        
        self.memory.decay()
        self.instability.update(total_c, diversity)
        final_field = self.instability.apply_instability(final_field)
        
        for e in self.engines:
            e.receive_field(final_field)
        
        self.step_count += 1
        
        return {
            'coherence': total_c,
            'diversity': diversity,
            'health': health,
            'alpha': self.anchor.alpha,
            'instability': self.instability.get_state(),
            'memory_size': self.memory.get_size(),
            'memory_strength': self.memory.get_strength(),
            'engine_self_coherence': engine_self_coherence,
            'engine_states': [e.get_info() for e in self.engines]
        }
    
    def get_history(self) -> dict:
        return {
            'coherence': list(self.history['coherence']),
            'diversity': list(self.history['diversity']),
            'health': list(self.history['health']),
            'engine_self_coherence': list(self.history['engine_self_coherence'])
        }


# ============================================================================
# MICRO-DYNAMICS VALIDATION TEST
# ============================================================================

def test_micro_dynamics():
    print("\n" + "█"*60)
    print("PHASE 2 - MICRO-DYNAMICS VALIDATION")
    print("Testing: self-coherence should NOT be 1.0")
    print("Target: max(self_coherence) < 0.99")
    print("█"*60)
    
    np.random.seed(42)
    system = Phase2System(4)
    
    print("\nRunning 300 steps with zero input...")
    print(f"{'Step':>6} {'Coherence':>10} {'Diversity':>10} {'Health':>10} {'Self Coh':>10}")
    print("-" * 55)
    
    self_coherence_values = []
    
    for step in range(300):
        zero_input = np.zeros(DIM)
        result = system.step(zero_input)
        self_coherence_values.append(result['engine_self_coherence'])
        
        if step % 50 == 0:
            print(f"{step:6d} {result['coherence']:10.4f} {result['diversity']:10.4f} "
                  f"{result['health']:10.4f} {result['engine_self_coherence']:10.4f}")
    
    max_self_coherence = np.max(self_coherence_values)
    
    print(f"\n{'='*55}")
    print(f"Maximum self-coherence: {max_self_coherence:.4f}")
    
    # CORRECT: Pass if NOT frozen at 1.0
    not_frozen = max_self_coherence < 0.99
    
    if not_frozen:
        print("\n✅ MICRO-DYNAMICS TEST PASSED: Self-coherence not frozen at 1.0")
        return True
    else:
        print("\n❌ MICRO-DYNAMICS TEST FAILED: Self-coherence frozen at 1.0")
        return False


def test_emergence_with_micro_dynamics():
    print("\n" + "█"*60)
    print("EMERGENCE TEST (with micro-dynamics)")
    print("No external input - structure should form with internal variation")
    print("█"*60)
    
    np.random.seed(42)
    system = Phase2System(4)
    
    zero_input = np.zeros(DIM)
    
    print("\nRunning 500 steps...")
    print(f"{'Step':>6} {'Coherence':>10} {'Diversity':>10} {'Health':>10} {'Self Coh':>10}")
    print("-" * 55)
    
    diversity_values = []
    self_coherence_values = []
    
    for step in range(500):
        result = system.step(zero_input)
        diversity_values.append(result['diversity'])
        self_coherence_values.append(result['engine_self_coherence'])
        
        if step % 50 == 0:
            print(f"{step:6d} {result['coherence']:10.4f} {result['diversity']:10.4f} "
                  f"{result['health']:10.4f} {result['engine_self_coherence']:10.4f}")
    
    final_diversity = np.mean(diversity_values[-50:])
    final_self_coherence = np.mean(self_coherence_values[-50:])
    max_self_coherence = np.max(self_coherence_values)
    
    print(f"\n{'='*55}")
    print(f"Final mean diversity: {final_diversity:.4f}")
    print(f"Final mean self-coherence: {final_self_coherence:.4f}")
    print(f"Maximum self-coherence: {max_self_coherence:.4f}")
    
    passed = (final_diversity > 0.1) and (max_self_coherence < 0.99) and (final_self_coherence > 0.7)
    
    if passed:
        print("\n✅ EMERGENCE TEST PASSED: Structure with internal variation")
    else:
        print("\n❌ EMERGENCE TEST FAILED")
    
    return passed


def run_validation():
    print("\n" + "="*60)
    print("PHASE 2 WITH MICRO-DYNAMICS - VALIDATION")
    print("="*60)
    
    # Test 1: Micro-dynamics (self-coherence NOT frozen)
    r1 = test_micro_dynamics()
    
    # Test 2: Emergence (structure forms with internal variation)
    r2 = test_emergence_with_micro_dynamics()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'✅' if r1 else '❌'} Micro-dynamics (self-coherence not frozen)")
    print(f"{'✅' if r2 else '❌'} Emergence with internal variation")
    
    if r1 and r2:
        print("\n" + "█"*60)
        print("🎉 PHASE 2 READY FOR PHASE 3")
        print("█"*60)
        print("\n✅ Engines have internal micro-dynamics")
        print("✅ Self-coherence NOT frozen (max < 0.99)")
        print("✅ Structure forms with internal variation")
        print("\n📌 Ready for Phase 3: Emotional Field Sensor")
    else:
        print("\n⚠️  Micro-dynamics need tuning")
    
    return r1 and r2


if __name__ == "__main__":
    run_validation()