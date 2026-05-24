import sys, types

# Mock matplotlib to avoid import errors
sys.modules['matplotlib'] = types.ModuleType('matplotlib')
sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')

# Mock audio libraries
sys.modules['soundfile'] = types.ModuleType('soundfile')
librosa = types.ModuleType('librosa')
librosa.feature = types.SimpleNamespace()
librosa.feature.mfcc = lambda y, sr, n_mfcc: __import__('numpy').zeros((n_mfcc, 1))
librosa.feature.spectral_centroid = lambda y, sr: __import__('numpy').zeros((1, 1))
librosa.feature.spectral_bandwidth = lambda y, sr: __import__('numpy').zeros((1, 1))
librosa.feature.zero_crossing_rate = lambda y: __import__('numpy').zeros((1, 1))
librosa.load = lambda filepath, sr: (__import__('numpy').zeros(sr), sr)
sys.modules['librosa'] = librosa

import phase6_complete as p6
import numpy as np

print('PHASE 6 SUSTAINED ERROR DETECTION - COMPREHENSIVE TEST')
print('='*60)

# Test patterns that were failing before
test_patterns = ['directional_push', 'parameter_shift', 'oscillation']
results_summary = {}

for pattern_name in test_patterns:
    print(f'\n--- Testing {pattern_name} ---')

    successes = 0
    total_runs = 5

    for run in range(total_runs):
        # Get the pattern enum
        pattern = getattr(p6.PerturbationPattern, pattern_name.upper())

        # Create proper PerturbationConfig object with stronger perturbation
        config = p6.PerturbationConfig(pattern, 2.0, 50, 200)  # stronger intensity, longer duration

        # Run single experiment
        suite = p6.Phase6VerificationSuite()
        result = suite._run_single_experiment(run, 1000 + run, config)

        if result.success:
            successes += 1

        print(f'  Run {run}: {"SUCCESS" if result.success else "FAILED"} | spike={result.spike_ratio:.1f}x | trans={result.mode_transitions}')

    success_rate = (successes / total_runs) * 100
    results_summary[pattern_name] = success_rate

    print(f'  SUCCESS RATE: {success_rate:.1f}% ({successes}/{total_runs})')

print('\n' + '='*60)
print('FINAL RESULTS SUMMARY:')
print('='*60)

for pattern, rate in results_summary.items():
    print(f'{pattern:15}: {rate:5.1f}% detection')

print('\nTARGET IMPROVEMENT:')
print('Expected: 70-90% detection for all structured perturbations')
print('Before fix: 0% detection for directional_push, parameter_shift, oscillation')
print('After fix: Should see significant improvement with sustained error detection')

# Check if sustained error detection is working
print('\nTESTING SUSTAINED ERROR DETECTION MECHANISM:')
print('-' * 50)

# Create a test scenario that should trigger sustained detection
classifier = p6.TrajectoryStateClassifier()

# Simulate sustained high error over many steps
print('Simulating sustained high error scenario...')
for step in range(15):  # More than the 10-step threshold
    error = 0.2  # Error above baseline * ratio (0.0253 * 1.3 = ~0.033)
    is_disturbed = classifier.classify_state(error, step)
    sustained_counter = classifier.sustained_error_counter
    print(f'Step {step}: error={error:.3f}, counter={sustained_counter}, disturbed={is_disturbed}')

print('\nExpected: Counter should reach 10 and trigger detection')