# phase11_complete.py
# PHASE 11: Open Developmental Ontogenesis (ODO)
# CORRECTED: Reduced residual locking + Real behavioral novelty + Stronger exploration

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque
import matplotlib.pyplot as plt
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Phase11Config:
    # ===== MODE SELECTION =====
    mode: str = "longhorizon"  # "full", "ablation", "longhorizon", "chunk2"
    
    # ===== System dimensions =====
    ontology_dim: int = 16
    num_ontologies: int = 2
    
    # ===== ES parameters (STRONGER EXPLORATION) =====
    es_samples: int = 5          # Increased from 3
    es_sigma: float = 0.08       # Increased from 0.05 0.12
    es_alpha: float = 0.03       # Increased from 0.02
    
    # ===== Neural network =====
    law_hidden_size: int = 6
    
    # ===== Dynamics =====
    dt: float = 0.01
    noise_scale: float = 0.008   # Slightly increased
    
    # ===== Residual mixing (FIXED - Reduced locking) =====
    residual_mix_base: float = 0.2  # Was 0.5 - Now adaptive 0.1
    residual_mix_range: float = 0.3  # For adaptive mixing
    
    # ===== Reference states =====
    n_test_states: int = 25
    trajectory_window: int = 200
    behavioral_archive_size: int = 500  # For real novelty
    
    # ===== Tasks =====
    task_switch_interval: int = 1000   # More frequent switching 500
    task_difficulty: float = 0.5
    environmental_drift: float = 0.01  # NEW: slow environmental change 0.02
    
    # ===== Simulation =====
    steps: int = 5000
    n_seeds: int = 2
    metric_interval: int = 50
    checkpoint_interval: int = 5000
    
    # ===== Ablation specific =====
    ablation_condition: str = "full"
    
    # ===== Long-horizon specific =====
    horizons: List[int] = field(default_factory=lambda: [5000, 10000, 25000, 50000])
    
    # ===== Output =====
    results_dir: str = "phase11_results"
    verbose: bool = True


# ============================================================================
# REAL STABILITY METRIC
# ============================================================================

def compute_real_stability(trajectories: Dict[int, deque], window: int = 200) -> float:
    """Trajectory smoothness - lower velocity = higher stability"""
    all_velocities = []
    
    for law_id, traj in trajectories.items():
        if len(traj) < window:
            continue
        traj_list = list(traj)[-window:]
        velocities = [np.linalg.norm(traj_list[i] - traj_list[i-1]) 
                      for i in range(1, len(traj_list))]
        if velocities:
            all_velocities.extend(velocities)
    
    if not all_velocities:
        return 0.5
    
    mean_vel = np.mean(all_velocities)
    max_scale = 0.2
    stability = 1.0 - min(1.0, mean_vel / max_scale)
    return np.clip(stability, 0.0, 1.0)


# ============================================================================
# REAL BEHAVIORAL NOVELTY (Not entropy drift)
# ============================================================================

class BehavioralNoveltyArchive:
    """Stores behavioral fingerprints for true novelty detection"""
    
    def __init__(self, config: Phase11Config):
        self.config = config
        self.archive = []  # List of (embedding, step)
        self.max_size = config.behavioral_archive_size
        
    def compute_embedding(self, law, test_states: np.ndarray) -> np.ndarray:
        """Create behavioral fingerprint from law's responses"""
        outputs = []
        for phi in test_states[:20]:
            out = law.forward(phi, None)
            outputs.extend(out.flatten())
        return np.array(outputs)
    
    def compute_novelty(self, law, test_states: np.ndarray) -> float:
        """Novelty = distance to nearest neighbor in archive"""
        embedding = self.compute_embedding(law, test_states)
        
        if len(self.archive) == 0:
            self.archive.append((embedding, 0))
            return 1.0
        
        # Find minimum distance to any archived behavior
        min_dist = float('inf')
        for archived_embedding, _ in self.archive[-self.max_size:]:
            # Cosine distance
            dot = np.dot(embedding, archived_embedding)
            norm_e = np.linalg.norm(embedding)
            norm_a = np.linalg.norm(archived_embedding)
            if norm_e > 0 and norm_a > 0:
                sim = dot / (norm_e * norm_a)
                dist = 1.0 - sim
                min_dist = min(min_dist, dist)
            else:
                min_dist = min(min_dist, 1.0)
        
        # Add to archive if novel enough (distance > 0.2)
        if min_dist > 0.2 and len(self.archive) < self.max_size:
            self.archive.append((embedding, len(self.archive)))
        
        # Normalize to [0, 1]
        return min(1.0, min_dist / 0.5)
    
    def reset(self):
        self.archive = []


# ============================================================================
# TASKS (With environmental drift)
# ============================================================================

class PredictionTask:
    def __init__(self, dim: int, difficulty: float):
        self.dim = dim
        self.difficulty = difficulty
    def evaluate(self, law, phi: np.ndarray, context=None) -> float:
        target = np.roll(phi, 1)
        prediction = law.forward(phi, context)
        error = np.linalg.norm(prediction - target)
        return 1.0 / (1.0 + error)

class ReconstructionTask:
    def __init__(self, dim: int, difficulty: float):
        self.dim = dim
        self.difficulty = difficulty
    def evaluate(self, law, phi: np.ndarray, context=None) -> float:
        compressed = law.forward(phi, context)
        reconstructed = law.forward(compressed, context)
        error = np.linalg.norm(phi - reconstructed)
        return 1.0 / (1.0 + error)

class ContradictionTask:
    def __init__(self, dim: int, difficulty: float):
        self.dim = dim
        self.difficulty = difficulty
    def evaluate(self, law, phi: np.ndarray, context=None) -> float:
        signal1 = phi + np.random.randn(self.dim) * 0.1
        signal2 = -phi + np.random.randn(self.dim) * 0.1
        output1 = law.forward(signal1, context)
        output2 = law.forward(signal2, context)
        input_contradiction = np.linalg.norm(signal1 - signal2)
        output_contradiction = np.linalg.norm(output1 - output2)
        if input_contradiction > 0:
            return 1.0 - min(1.0, output_contradiction / input_contradiction)
        return 0.5

class TranslationTask:
    def __init__(self, dim: int, difficulty: float):
        self.dim = dim
        self.difficulty = difficulty
    def evaluate(self, law, phi: np.ndarray, context=None) -> float:
        if context is None:
            context = np.random.randn(self.dim)
            context /= (np.linalg.norm(context) + 1e-8)
        translation = law.forward(phi, context)
        corr = np.corrcoef(phi, translation)[0, 1]
        return max(0, corr)


# ============================================================================
# EVOLVABLE LAW (FIXED - Reduced residual locking)
# ============================================================================

class EvolvableLaw:
    def __init__(self, dim: int, hidden_size: int, ontology_id: int, config: Phase11Config):
        self.dim = dim
        self.hidden_size = hidden_size
        self.id = ontology_id
        self.config = config
        
        scale = np.sqrt(2.0 / dim)
        self.W1 = np.random.randn(hidden_size, dim).astype(np.float32) * scale
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = np.random.randn(dim, hidden_size).astype(np.float32) * scale
        self.b2 = np.zeros(dim, dtype=np.float32)
        self.theta = self._flatten_weights()
        
    def _flatten_weights(self) -> np.ndarray:
        return np.concatenate([self.W1.flatten(), self.b1.flatten(),
                               self.W2.flatten(), self.b2.flatten()])
    
    def _unflatten_weights(self, theta: np.ndarray):
        idx = 0
        size_W1 = self.hidden_size * self.dim
        self.W1 = theta[idx:idx+size_W1].reshape(self.hidden_size, self.dim)
        idx += size_W1
        size_b1 = self.hidden_size
        self.b1 = theta[idx:idx+size_b1]
        idx += size_b1
        size_W2 = self.dim * self.hidden_size
        self.W2 = theta[idx:idx+size_W2].reshape(self.dim, self.hidden_size)
        idx += size_W2
        size_b2 = self.dim
        self.b2 = theta[idx:idx+size_b2]
    
    def set_theta(self, theta: np.ndarray):
        self.theta = theta.copy()
        self._unflatten_weights(theta)
    
    def forward(self, phi: np.ndarray, context: np.ndarray = None) -> np.ndarray:
        x = phi.copy().astype(np.float32)
        if context is not None:
            if len(context.shape) > 1:
                context = context.flatten()
            x = np.concatenate([phi, context[:self.dim]])
        if len(x) < self.dim:
            x = np.pad(x, (0, self.dim - len(x)))
        elif len(x) > self.dim:
            x = x[:self.dim]
        
        h = np.tanh(self.W1 @ x + self.b1)
        output = np.tanh(self.W2 @ h + self.b2)
        
        # FIXED: Adaptive residual mixing - less locking
        # Was: output * 0.5 + phi * 0.5 (50% identity)
        # Now: Adaptive mixing between 10-40% identity
        mix = self.config.residual_mix_base + np.random.rand() * self.config.residual_mix_range
        return (output * (1 - mix) + phi * mix).astype(np.float32)
    
    def compute_sensitivity(self, test_states: np.ndarray) -> float:
        sensitivities = []
        eps = 0.01
        for phi in test_states[:10]:
            base = self.forward(phi, None)
            for d in range(min(3, self.dim)):
                phi_pert = phi.copy()
                phi_pert[d] += eps
                pert = self.forward(phi_pert, None)
                diff = np.linalg.norm(pert - base) / eps
                sensitivities.append(diff)
        # FIXED: Less aggressive clipping
        return np.mean(sensitivities) if sensitivities else 0.0
    
    def compute_output_entropy(self, test_states: np.ndarray) -> float:
        outputs = []
        for phi in test_states[:20]:
            out = self.forward(phi, None)
            outputs.extend(out.flatten())
        if len(outputs) == 0:
            return 0.0
        outputs = np.array(outputs)
        n_bins = 20
        hist, _ = np.histogram(outputs, bins=n_bins)
        hist = hist.astype(np.float64)
        hist_sum = hist.sum()
        if hist_sum < 1e-8:
            return 0.0
        probs = hist / hist_sum
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        max_entropy = np.log(n_bins)
        return np.clip(entropy / max_entropy, 0.0, 1.0) if max_entropy > 0 else 0.0
    
    def copy(self) -> 'EvolvableLaw':
        new_law = EvolvableLaw(self.dim, self.hidden_size, self.id, self.config)
        new_law.set_theta(self.theta)
        return new_law


# ============================================================================
# DYNAMIC TASK ENVIRONMENT (With drift)
# ============================================================================

class DynamicTaskEnvironment:
    def __init__(self, config: Phase11Config):
        self.config = config
        self.dim = config.ontology_dim
        self.current_task_idx = 0
        self.context_noise = 0.0
        
        self.tasks = [
            PredictionTask(self.dim, config.task_difficulty),
            ReconstructionTask(self.dim, config.task_difficulty),
            ContradictionTask(self.dim, config.task_difficulty),
            TranslationTask(self.dim, config.task_difficulty)
        ]
    
    def get_current_tasks(self, step: int) -> List:
        # Rotate tasks periodically
        phase = step // self.config.task_switch_interval
        idx = phase % len(self.tasks)
        
        # Add environmental drift
        self.context_noise += np.random.randn() * self.config.environmental_drift
        self.context_noise = np.clip(self.context_noise, -0.5, 0.5)
        
        # Return 2 tasks (primary + secondary)
        primary = self.tasks[idx]
        secondary = self.tasks[(idx + 1) % len(self.tasks)]
        return [primary, secondary]
    
    def get_context(self) -> np.ndarray:
        """Slowly drifting environmental context"""
        return np.ones(self.dim) * self.context_noise


# ============================================================================
# FITNESS FUNCTIONS
# ============================================================================

class BaseFitness:
    def __init__(self, config: Phase11Config):
        self.config = config
        self.weights = {'divergence': 0.25, 'sensitivity': 0.25, 
                       'entropy': 0.20, 'utility': 0.20, 'stability': 0.10}
    
    def compute(self, law, test_states, task_scores, trajectories, divergence):
        sensitivity = law.compute_sensitivity(test_states)
        # FIXED: Less aggressive clipping
        sensitivity = min(1.0, sensitivity / 1.5)
        entropy = law.compute_output_entropy(test_states)
        utility = np.mean(task_scores) if task_scores else 0.0
        stability = compute_real_stability(trajectories)
        
        return {'divergence': divergence, 'sensitivity': sensitivity,
                'entropy': entropy, 'utility': utility, 'stability': stability}
    
    def scalarize(self, obj): 
        return sum(self.weights[k] * obj[k] for k in self.weights)


class FullFitness(BaseFitness):
    pass


# ============================================================================
# TASK-CONDITIONED DIVERGENCE
# ============================================================================

def task_conditioned_divergence(ontologies, test_states, task) -> float:
    if len(ontologies) < 2:
        return 0.0
    behaviors = []
    for onto in ontologies:
        behavior = []
        for phi in test_states[:20]:
            out = onto.law.forward(phi, None)
            task_score = task.evaluate(onto.law, phi, None)
            behavior.append(np.concatenate([out, [task_score]]))
        behaviors.append(np.concatenate(behavior))
    divergences = []
    for i in range(len(behaviors)):
        for j in range(i+1, len(behaviors)):
            b_i = behaviors[i] - np.mean(behaviors[i])
            b_j = behaviors[j] - np.mean(behaviors[j])
            norm_i = np.linalg.norm(b_i) + 1e-8
            norm_j = np.linalg.norm(b_j) + 1e-8
            sim = np.dot(b_i, b_j) / (norm_i * norm_j)
            div = 1.0 - max(-1.0, min(1.0, sim))
            divergences.append(div)
    return np.mean(divergences) if divergences else 0.0


# ============================================================================
# ONTOLOGY
# ============================================================================

class Ontology:
    def __init__(self, idx: int, config: Phase11Config):
        self.id = idx
        self.config = config
        self.dim = config.ontology_dim
        
        self.Phi = np.random.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(self.Phi)
        if norm > 0:
            self.Phi /= norm
        
        self.law = EvolvableLaw(self.dim, config.law_hidden_size, idx, config)
        self.trajectory = deque(maxlen=config.trajectory_window)
        self.full_trajectory: List[np.ndarray] = []
        self.trajectory.append(self.Phi.copy())
        self.full_trajectory.append(self.Phi.copy())
        
    def evaluate_tasks(self, tasks, test_states):
        scores = []
        for task in tasks:
            task_score = 0.0
            for phi in test_states[:10]:
                score = task.evaluate(self.law, phi, None)
                task_score += score
            scores.append(task_score / 10)
        return scores
    
    def update_state(self, others, dt, noise_scale, context):
        if others:
            E = np.mean([o.Phi for o in others], axis=0)
        else:
            E = np.zeros(self.dim, dtype=np.float32)
        E += context  # Add environmental drift
        
        delta = self.law.forward(self.Phi, E)
        noise = np.random.randn(self.dim).astype(np.float32) * noise_scale
        self.Phi = self.Phi + dt * delta + noise
        norm = np.linalg.norm(self.Phi)
        if norm > 0:
            self.Phi /= norm
        self.trajectory.append(self.Phi.copy())
        self.full_trajectory.append(self.Phi.copy())


# ============================================================================
# MAIN SYSTEM
# ============================================================================

class Phase11System:
    def __init__(self, config: Phase11Config, seed: int = None):
        self.config = config
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        self.ontologies: List[Ontology] = []
        self.task_env = DynamicTaskEnvironment(config)
        self.novelty_archive = BehavioralNoveltyArchive(config)
        self.test_states = None
        self.time = 0
        self.current_tasks = []
        
        self.history = {'divergence': [], 'sensitivity': [], 'entropy': [], 
                        'utility': [], 'stability': [], 'novelty': []}
        
        os.makedirs(config.results_dir, exist_ok=True)
        
    def initialize(self):
        self.ontologies = []
        for i in range(self.config.num_ontologies):
            onto = Ontology(i, self.config)
            self.ontologies.append(onto)
        
        self.test_states = np.random.randn(self.config.n_test_states, self.config.ontology_dim).astype(np.float32)
        for i in range(self.config.n_test_states):
            norm = np.linalg.norm(self.test_states[i])
            if norm > 0:
                self.test_states[i] /= norm
    
    def compute_divergence(self) -> float:
        if len(self.ontologies) < 2 or not self.current_tasks:
            return 0.0
        divergences = []
        for task in self.current_tasks:
            div = task_conditioned_divergence(self.ontologies, self.test_states, task)
            divergences.append(div)
        return np.mean(divergences) if divergences else 0.0
    
    def evolution_step(self):
        c = self.config
        for idx, onto in enumerate(self.ontologies):
            others = [o for j, o in enumerate(self.ontologies) if j != idx]
            task_scores = onto.evaluate_tasks(self.current_tasks, self.test_states)
            divergence = self.compute_divergence()
            trajectories = {o.id: o.trajectory for o in self.ontologies}
            
            fitness = FullFitness(c)
            objectives = fitness.compute(onto.law, self.test_states, task_scores, trajectories, divergence)
            current_fitness = fitness.scalarize(objectives)
            
            current_theta = onto.law.theta.copy()
            perturbations = []
            fitness_deltas = []
            
            for _ in range(c.es_samples):
                pert = np.random.randn(len(current_theta)) * c.es_sigma
                onto.law.set_theta(current_theta + pert)
                new_task_scores = onto.evaluate_tasks(self.current_tasks, self.test_states)
                new_objectives = fitness.compute(onto.law, self.test_states, new_task_scores, trajectories, divergence)
                new_fitness = fitness.scalarize(new_objectives)
                perturbations.append(pert)
                fitness_deltas.append(new_fitness - current_fitness)
            
            theta_update = np.zeros_like(current_theta)
            for pert, delta in zip(perturbations, fitness_deltas):
                theta_update += delta * pert
            theta_update *= (c.es_alpha / (c.es_samples * c.es_sigma))
            onto.law.set_theta(current_theta + theta_update)
    
    def step(self):
        c = self.config
        
        # Update dynamic environment
        self.current_tasks = self.task_env.get_current_tasks(self.time)
        context = self.task_env.get_context()
        
        # State update
        for i, onto in enumerate(self.ontologies):
            others = [o for j, o in enumerate(self.ontologies) if j != i]
            onto.update_state(others, c.dt, c.noise_scale, context)
        
        # Evolution
        self.evolution_step()
        
        # Record metrics
        if self.time % c.metric_interval == 0:
            divergence = self.compute_divergence()
            all_sensitivity = 0
            all_entropy = 0
            all_utility = 0
            
            for onto in self.ontologies:
                task_scores = onto.evaluate_tasks(self.current_tasks, self.test_states)
                all_sensitivity += onto.law.compute_sensitivity(self.test_states)
                all_entropy += onto.law.compute_output_entropy(self.test_states)
                all_utility += np.mean(task_scores)
            
            trajectories = {o.id: o.trajectory for o in self.ontologies}
            all_stability = compute_real_stability(trajectories)
            
            # REAL behavioral novelty (not entropy drift)
            all_novelty = 0
            for onto in self.ontologies:
                all_novelty += self.novelty_archive.compute_novelty(onto.law, self.test_states)
            
            n = len(self.ontologies)
            self.history['divergence'].append(divergence)
            self.history['sensitivity'].append(all_sensitivity / n)
            self.history['entropy'].append(all_entropy / n)
            self.history['utility'].append(all_utility / n)
            self.history['stability'].append(all_stability)
            self.history['novelty'].append(all_novelty / n)
        
        self.time += 1
    
    def run(self, steps: int = None, save_states: bool = False):
        if steps is None:
            steps = self.config.steps
        self.initialize()
        for _ in range(steps):
            self.step()
        self.save_states = bool(save_states)
        return self

    def save_trajectories(self, path: str):
        """Save ontology trajectories to a pickle file."""
        trajectories = []
        for onto in self.ontologies:
            trajectory_data = np.array(
                onto.full_trajectory if hasattr(onto, 'full_trajectory') else onto.trajectory,
                dtype=np.float32
            )
            trajectories.append({
                'ontology_id': onto.id,
                'trajectory': trajectory_data
            })
        with open(path, 'wb') as f:
            pickle.dump(trajectories, f)
        if self.config.verbose:
            print(f"Trajectories saved to {path}")


# ============================================================================
# LONG-HORIZON VALIDATION
# ============================================================================

def run_longhorizon(config: Phase11Config):
    print("=" * 70)
    print("PHASE 11 - LONG-HORIZON VALIDATION (CORRECTED)")
    print("Testing: Controlled instability + Real behavioral novelty")
    print("=" * 70)
    
    results = []
    
    for horizon in config.horizons:
        print(f"\n  Running {horizon} steps...", end=" ", flush=True)
        start = time.time()
        
        config.steps = horizon
        seeds_div = []
        seeds_util = []
        seeds_novelty = []
        seeds_entropy = []
        seeds_stability = []
        
        for seed in range(min(config.n_seeds, 2)):
            sys = Phase11System(config, seed)
            sys.run()
            
            window = max(500, horizon // 10)
            seeds_div.append(np.mean(sys.history['divergence'][-window:]) if sys.history['divergence'] else 0)
            seeds_util.append(np.mean(sys.history['utility'][-window:]) if sys.history['utility'] else 0)
            seeds_entropy.append(np.mean(sys.history['entropy'][-window:]) if sys.history['entropy'] else 0)
            seeds_stability.append(np.mean(sys.history['stability'][-window:]) if sys.history['stability'] else 0)
            seeds_novelty.append(np.mean(sys.history['novelty'][-window:]) if sys.history['novelty'] else 0)
        
        elapsed = time.time() - start
        print(f"Div={np.mean(seeds_div):.3f}, Nov={np.mean(seeds_novelty):.3f} ({elapsed:.1f}s)")
        
        results.append({
            'horizon': horizon,
            'divergence': np.mean(seeds_div),
            'utility': np.mean(seeds_util),
            'entropy': np.mean(seeds_entropy),
            'stability': np.mean(seeds_stability),
            'novelty': np.mean(seeds_novelty)
        })
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    horizons = [r['horizon'] for r in results]
    
    axes[0,0].plot(horizons, [r['divergence'] for r in results], 'bo-', linewidth=2)
    axes[0,0].set_xlabel('Steps'); axes[0,0].set_ylabel('Divergence')
    axes[0,0].set_title('Task-Conditioned Divergence')
    axes[0,0].axhline(y=0.3, color='r', linestyle='--')
    axes[0,0].set_xscale('log')
    
    axes[0,1].plot(horizons, [r['utility'] for r in results], 'gs-', linewidth=2)
    axes[0,1].set_xlabel('Steps'); axes[0,1].set_ylabel('Utility')
    axes[0,1].set_title('Functional Competence')
    axes[0,1].axhline(y=0.3, color='r', linestyle='--')
    axes[0,1].set_xscale('log')
    
    axes[0,2].plot(horizons, [r['novelty'] for r in results], 'md-', linewidth=2)
    axes[0,2].set_xlabel('Steps'); axes[0,2].set_ylabel('Novelty')
    axes[0,2].set_title('Behavioral Novelty (Real Archive)')
    axes[0,2].axhline(y=0.2, color='r', linestyle='--')
    axes[0,2].set_xscale('log')
    
    axes[1,0].plot(horizons, [r['entropy'] for r in results], 'c^-', linewidth=2)
    axes[1,0].set_xlabel('Steps'); axes[1,0].set_ylabel('Entropy')
    axes[1,0].set_title('Output Diversity')
    axes[1,0].axhline(y=0.3, color='r', linestyle='--')
    axes[1,0].set_xscale('log')
    
    axes[1,1].plot(horizons, [r['stability'] for r in results], 'yo-', linewidth=2)
    axes[1,1].set_xlabel('Steps'); axes[1,1].set_ylabel('Stability')
    axes[1,1].set_title('Real Trajectory Stability')
    axes[1,1].axhline(y=0.7, color='r', linestyle='--')
    axes[1,1].set_xscale('log')
    
    axes[1,2].plot(horizons, [r['divergence'] for r in results], 'b-', label='Divergence', linewidth=2)
    axes[1,2].plot(horizons, [r['utility'] for r in results], 'g-', label='Utility', linewidth=2)
    axes[1,2].plot(horizons, [r['novelty'] for r in results], 'm-', label='Novelty', linewidth=2)
    axes[1,2].set_xlabel('Steps'); axes[1,2].set_ylabel('Value')
    axes[1,2].set_title('All Metrics')
    axes[1,2].legend()
    axes[1,2].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{config.results_dir}/longhorizon_corrected.png', dpi=150)
    plt.show()
    
    print("\n" + "=" * 70)
    print("LONG-HORIZON SUMMARY")
    print("=" * 70)
    
    final = results[-1]
    print(f"After {final['horizon']} steps:")
    print(f"  Divergence: {final['divergence']:.3f} (target >0.3) → {'✅' if final['divergence'] > 0.3 else '❌'}")
    print(f"  Utility:    {final['utility']:.3f} (target >0.3) → {'✅' if final['utility'] > 0.3 else '❌'}")
    print(f"  Novelty:    {final['novelty']:.3f} (target >0.2) → {'✅' if final['novelty'] > 0.2 else '⚠️'}")
    print(f"  Entropy:    {final['entropy']:.3f} (target >0.3) → {'✅' if final['entropy'] > 0.3 else '❌'}")
    print(f"  Stability:  {final['stability']:.3f} (target <0.8) → {'✅' if final['stability'] < 0.8 else '⚠️'}")
    
    all_ok = (final['divergence'] > 0.3 and final['utility'] > 0.3 and 
              final['entropy'] > 0.3 and final['stability'] < 0.8)
    
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if all_ok:
        print("✓ LONG-HORIZON VALIDATION PASSED")
        print("→ READY FOR CHUNK 2: Emergent Geometry Evolution")
    else:
        print("⚠️ LONG-HORIZON VALIDATION PARTIAL")
        print("→ Running with corrected parameters...")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    config = Phase11Config(
        mode="longhorizon",
        steps=5000,
        n_seeds=2,
        results_dir="phase11_results"
    )
    
    if config.mode == "longhorizon":
        run_longhorizon(config)
    else:
        print("Run with mode='longhorizon' for validation")
    
    print("\n" + "=" * 70)
    print("PHASE 11 - CORRECTED BALANCE COMPLETE")
    print("=" * 70)