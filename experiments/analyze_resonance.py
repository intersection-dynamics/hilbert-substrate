#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def analyze_resonance(json_path="gpu_mesoscape_organizer_collective_response.json"):
    print(f"Loading collective response data from {json_path}...\n")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{json_path}'.")
        return

    curves = data.get("response_curves", [])
    if not curves:
        print("Error: No response curves found in JSON.")
        return

    # Extract the Mutual Information shift over time
    steps = np.array([r["probe_step"] for r in curves])
    mi_shift = np.array([r["core_shell_mi_shift"] for r in curves])
    
    # We need a decent number of steps to do a meaningful FFT
    n_samples = len(mi_shift)
    if n_samples < 10:
        print("Warning: Not enough probe steps to run a reliable Fourier Transform.")
        
    # Remove any DC offset (baseline drift)
    mi_shift_centered = mi_shift - np.mean(mi_shift)

    # Perform Fast Fourier Transform (FFT)
    # Assuming dt=1 simulation step for the frequency domain
    yf = fft(mi_shift_centered)
    xf = fftfreq(n_samples, d=1.0)
    
    # Only take the positive frequencies
    positive_freqs = xf[:n_samples//2]
    amplitudes = np.abs(yf[:n_samples//2])
    
    # Find the dominant resonant frequency
    dominant_idx = np.argmax(amplitudes)
    dominant_freq = positive_freqs[dominant_idx]
    
    # Convert frequency to angular frequency (omega = 2 * pi * f)
    omega = 2.0 * np.pi * dominant_freq

    print("=" * 64)
    print("TOPOLOGICAL RESONANCE (FFT) ANALYSIS")
    print("=" * 64)
    print(f"Poke Mode Applied        : {data.get('poke_mode', 'Unknown')}")
    print(f"Total Probe Steps        : {n_samples}")
    print("-" * 64)
    print(f"Dominant Frequency (f)   : {dominant_freq:.4f} cycles / step")
    print(f"Angular Frequency (w)    : {omega:.4f} radians / step")
    print("=" * 64)
    
    # The physical "Mass" or "Energy" of the defect is proportional to omega
    print(f"Dimensionless Energy/Mass Signature (E ~ w) : {omega:.4f}")
    print("=" * 64)

    # Plot the original wave and the FFT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Time Domain
    ax1.plot(steps, mi_shift, color='purple', marker='.', linestyle='-')
    ax1.set_title("Time Domain: Core-Shell Mutual Information Ringing")
    ax1.set_xlabel("Simulation Step (dt)")
    ax1.set_ylabel("MI Shift")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Frequency Domain
    ax2.plot(positive_freqs, amplitudes, color='teal', linewidth=2)
    ax2.fill_between(positive_freqs, amplitudes, color='teal', alpha=0.3)
    ax2.axvline(dominant_freq, color='red', linestyle='--', label=f'Peak: {dominant_freq:.3f}')
    ax2.set_title("Frequency Domain (FFT): Defect Energy Signature")
    ax2.set_xlabel("Frequency (cycles/step)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("resonance_fft_analysis.png", dpi=300)
    print("Saved resonance plots to 'resonance_fft_analysis.png'")

if __name__ == "__main__":
    analyze_resonance()