import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, lfilter, resample

def generate_normal_beat(samples=200, fs=360):
    """Generate a synthetic normal ECG beat"""
    t = np.linspace(0, samples/fs, samples)
    
    # Base signal with P wave, QRS complex, and T wave
    signal = np.zeros_like(t)
    
    # P wave
    p_center = 0.2 * samples/fs
    signal += 0.25 * np.exp(-((t - p_center) ** 2) / 0.002)
    
    # QRS complex
    qrs_center = 0.4 * samples/fs
    signal -= 0.2 * np.exp(-((t - qrs_center) ** 2) / 0.0005)  # Q
    signal += 1.0 * np.exp(-((t - (qrs_center + 0.02)) ** 2) / 0.0002)  # R
    signal -= 0.3 * np.exp(-((t - (qrs_center + 0.05)) ** 2) / 0.0008)  # S
    
    # T wave
    t_center = 0.7 * samples/fs
    signal += 0.3 * np.exp(-((t - t_center) ** 2) / 0.004)
    
    # Add slight baseline variation
    signal += 0.05 * np.sin(2 * np.pi * 0.3 * t)
    
    # Add some noise
    signal += 0.03 * np.random.randn(len(t))
    
    return signal

def generate_supraventricular_beat(samples=200, fs=360):
    """Generate a synthetic supraventricular beat"""
    t = np.linspace(0, samples/fs, samples)
    
    # Base signal
    signal = np.zeros_like(t)
    
    # No distinct P wave or inverted/abnormal P wave
    p_center = 0.2 * samples/fs
    signal += 0.15 * np.exp(-((t - p_center) ** 2) / 0.001)  # Less prominent P wave
    
    # Normal QRS complex
    qrs_center = 0.4 * samples/fs
    signal -= 0.2 * np.exp(-((t - qrs_center) ** 2) / 0.0005)  # Q
    signal += 1.0 * np.exp(-((t - (qrs_center + 0.02)) ** 2) / 0.0002)  # R
    signal -= 0.3 * np.exp(-((t - (qrs_center + 0.05)) ** 2) / 0.0008)  # S
    
    # T wave
    t_center = 0.7 * samples/fs
    signal += 0.3 * np.exp(-((t - t_center) ** 2) / 0.004)
    
    # Varying baseline
    signal += 0.1 * np.sin(2 * np.pi * 0.5 * t)
    
    # Add some noise
    signal += 0.05 * np.random.randn(len(t))
    
    return signal

def generate_ventricular_beat(samples=200, fs=360):
    """Generate a synthetic ventricular beat"""
    t = np.linspace(0, samples/fs, samples)
    
    # Base signal
    signal = np.zeros_like(t)
    
    # No distinct P wave
    
    # Wide QRS complex
    qrs_center = 0.4 * samples/fs
    signal -= 0.3 * np.exp(-((t - qrs_center) ** 2) / 0.002)  # Wider Q
    signal += 1.2 * np.exp(-((t - (qrs_center + 0.04)) ** 2) / 0.001)  # Higher R
    signal -= 0.5 * np.exp(-((t - (qrs_center + 0.09)) ** 2) / 0.003)  # Wider and deeper S
    
    # T wave often inverted
    t_center = 0.8 * samples/fs
    signal -= 0.4 * np.exp(-((t - t_center) ** 2) / 0.005)
    
    # Add some baseline variation
    signal += 0.1 * np.sin(2 * np.pi * 0.4 * t)
    
    # Add some noise
    signal += 0.05 * np.random.randn(len(t))
    
    return signal

def generate_fusion_beat(samples=200, fs=360):
    """Generate a synthetic fusion beat (fusion of ventricular and normal)"""
    t = np.linspace(0, samples/fs, samples)
    
    # Base signal
    signal = np.zeros_like(t)
    
    # Slight P wave
    p_center = 0.2 * samples/fs
    signal += 0.15 * np.exp(-((t - p_center) ** 2) / 0.001)
    
    # Fusion QRS - characteristics of both normal and ventricular
    qrs_center = 0.4 * samples/fs
    signal -= 0.2 * np.exp(-((t - qrs_center) ** 2) / 0.001)  # Q
    signal += 1.1 * np.exp(-((t - (qrs_center + 0.03)) ** 2) / 0.0005)  # R - taller than normal
    signal -= 0.4 * np.exp(-((t - (qrs_center + 0.07)) ** 2) / 0.002)  # S - wider than normal
    
    # T wave - may be abnormal
    t_center = 0.75 * samples/fs
    signal += 0.2 * np.exp(-((t - t_center) ** 2) / 0.004)
    
    # Add baseline variation
    signal += 0.08 * np.sin(2 * np.pi * 0.35 * t)
    
    # Add some noise
    signal += 0.04 * np.random.randn(len(t))
    
    return signal

def generate_unknown_beat(samples=200, fs=360):
    """Generate a synthetic unknown/aberrant beat"""
    t = np.linspace(0, samples/fs, samples)
    
    # Base signal - more chaotic and unpredictable
    signal = np.zeros_like(t)
    
    # Random components
    for i in range(3):
        center = np.random.uniform(0.2, 0.8) * samples/fs
        width = np.random.uniform(0.0005, 0.003)
        height = np.random.uniform(-0.8, 1.2)
        signal += height * np.exp(-((t - center) ** 2) / width)
    
    # Add some high-frequency components
    signal += 0.2 * np.sin(2 * np.pi * 25 * t)
    
    # Add some baseline wander
    signal += 0.15 * np.sin(2 * np.pi * 0.3 * t)
    
    # Add more noise
    signal += 0.08 * np.random.randn(len(t))
    
    return signal

def generate_ecg_sequence(beat_types, beat_count=10, sampling_rate=360):
    """Generate a sequence of ECG beats with the specified types"""
    beat_generators = {
        'Normal': generate_normal_beat,
        'Supraventricular': generate_supraventricular_beat,
        'Ventricular': generate_ventricular_beat,
        'Fusion': generate_fusion_beat,
        'Unknown': generate_unknown_beat
    }
    
    beat_length = 200  # samples per beat
    
    # Create the full sequence
    sequence = []
    true_labels = []
    
    for _ in range(beat_count):
        beat_type = np.random.choice(beat_types)
        beat = beat_generators[beat_type](samples=beat_length, fs=sampling_rate)
        sequence.extend(beat)
        true_labels.append(beat_type)
    
    # Add some continuous connection between beats
    sequence = np.array(sequence)
    
    # Apply a light filter to smooth connections
    b, a = butter(3, 0.1)
    sequence = lfilter(b, a, sequence)
    
    return sequence, true_labels

def generate_test_files():
    """Generate various test files for the app"""
    print("Generating test ECG data files...")
    
    # Create directory if it doesn't exist
    os.makedirs("test_data", exist_ok=True)
    
    # Generate different types of test files
    
    # 1. Normal ECG
    normal_ecg, _ = generate_ecg_sequence(['Normal'], beat_count=15)
    np.savetxt("test_data/normal_ecg.csv", normal_ecg, delimiter=',')
    
    # 2. Ventricular ECG
    ventricular_ecg, _ = generate_ecg_sequence(['Ventricular'], beat_count=15)
    np.savetxt("test_data/ventricular_ecg.csv", ventricular_ecg, delimiter=',')
    
    # 3. Mixed ECG - all types
    mixed_ecg, labels = generate_ecg_sequence(
        ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown'], 
        beat_count=20
    )
    np.savetxt("test_data/mixed_ecg.csv", mixed_ecg, delimiter=',')
    
    # 4. Mixed ECG with two columns (time and amplitude)
    time = np.linspace(0, len(mixed_ecg)/360, len(mixed_ecg))
    df = pd.DataFrame({
        'Time': time,
        'ECG': mixed_ecg
    })
    df.to_csv("test_data/ecg_with_time.csv", index=False)
    
    # 5. Create ECG with headers and multiple columns (adding a synthetic respiration signal)
    respiration = 0.5 * np.sin(2 * np.pi * 0.1 * time) + 0.1 * np.random.randn(len(time))
    df = pd.DataFrame({
        'Time(s)': time,
        'ECG(mV)': mixed_ecg,
        'Resp(V)': respiration
    })
    df.to_csv("test_data/multiparameter.csv", index=False)
    
    # Plot and save examples as PNGs for documentation
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(normal_ecg[:600])
    plt.title('Normal ECG')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(ventricular_ecg[:600])
    plt.title('Ventricular ECG')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(mixed_ecg[:600])
    plt.title('Mixed ECG')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("test_data/example_ecgs.png")
    
    # Create and save individual beat examples
    plt.figure(figsize=(15, 10))
    
    beat_generators = {
        'Normal': generate_normal_beat,
        'Supraventricular': generate_supraventricular_beat,
        'Ventricular': generate_ventricular_beat,
        'Fusion': generate_fusion_beat,
        'Unknown': generate_unknown_beat
    }
    
    for i, (name, generator) in enumerate(beat_generators.items()):
        plt.subplot(5, 1, i+1)
        beat = generator()
        plt.plot(beat)
        plt.title(f'{name} Beat')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("test_data/beat_examples.png")
    
    print("Test data generated successfully in 'test_data' directory")
    print("Test files created:")
    print("  - normal_ecg.csv: Normal ECG data")
    print("  - ventricular_ecg.csv: Ventricular arrhythmia data")
    print("  - mixed_ecg.csv: Mixed arrhythmia types")  
    print("  - ecg_with_time.csv: ECG with time column")
    print("  - multiparameter.csv: ECG with multiple physiological signals")
    print("  - example_ecgs.png: Visualization of test ECGs")
    print("  - beat_examples.png: Visualization of individual beat types")

if __name__ == "__main__":
    generate_test_files()
