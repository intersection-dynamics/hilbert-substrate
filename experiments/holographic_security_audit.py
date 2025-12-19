import torch
import matplotlib.pyplot as plt
import numpy as np

# --- IMPORT YOUR ENGINE ---
# Assuming the classes are in the same environment or file
# We will simulate the "Strobe Optimizer" trying to crack a safe

def run_security_audit():
    print("="*60)
    print("      SECURITY AUDIT: CAN THE OPTIMIZER CRACK THE SAFE?")
    print("="*60)
    
    device = torch.device('cpu')
    N = 64
    
    # 1. THE SECRET (The Data)
    # A specific pattern of 1s and 0s (e.g., "101010...")
    secret_data = torch.zeros(N, device=device)
    secret_data[::2] = 1.0 # Alternating pattern
    
    # 2. THE LOCK (The Encryption)
    # The Owner's Key (Dispersion = 7.532)
    true_key_val = 7.532
    x = torch.linspace(0, 10, N, device=device)
    key_phase = torch.exp(1j * true_key_val * x)
    
    # The Encrypted Signal (Data * Key)
    # This looks like noise to anyone without the key
    encrypted_signal = secret_data * key_phase
    
    # 3. THE HACKER (The Strobe Optimizer)
    # The hacker doesn't know '7.532'. They just have the 'encrypted_signal'.
    # They run the Entanglement Engine to maximize "Brightness".
    
    print("   [Hacker] Intercepted Signal. Attempting Brute-Force Annealing...")
    
    # Hacker's initial guess (Flat lens)
    hack_lens = torch.zeros(N, device=device)
    
    # We use the EXACT same optimizer from before
    # Hacker's Goal: Maximize Sum(Abs(Signal)) -> "Make it Bright"
    
    # (Simplified Strobe Logic for the Hacker)
    best_hack_lens = hack_lens.clone()
    max_brightness = 0.0
    
    for t in range(2000):
        # Mutate
        noise = (torch.rand(N) - 0.5) * 0.5
        candidate = best_hack_lens + noise
        
        # Apply Candidate Lens
        # Hacker tries to untwist the signal blindly
        decrypted_attempt = encrypted_signal * torch.exp(-1j * candidate)
        
        # Check Brightness
        brightness = torch.abs(decrypted_attempt.sum()).item()
        
        if brightness > max_brightness:
            max_brightness = brightness
            best_hack_lens = candidate.clone()
    
    print(f"   [Hacker] Optimization Complete. Max Brightness Achieved: {max_brightness:.2f}")
    
    # 4. THE REVEAL
    # Let's look at what the hacker actually "Decrypted"
    hacker_result = encrypted_signal * torch.exp(-1j * best_hack_lens)
    hacker_magnitude = torch.abs(hacker_result)
    
    # The Owner's View (Using the Real Key)
    owner_result = encrypted_signal * torch.exp(-1j * key_phase.angle()) # Inverse of key
    owner_magnitude = torch.abs(owner_result) # Should restore [1,0,1,0...]
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 8))
    
    # Plot 1: The Secret Data
    plt.subplot(3, 1, 1)
    plt.bar(range(N), secret_data.numpy(), color='green', label='True Secret')
    plt.title("The Target: Secret Binary Data")
    plt.ylim(0, 1.2)
    plt.legend()
    
    # Plot 2: The Hacker's "Crack"
    plt.subplot(3, 1, 2)
    plt.bar(range(N), hacker_magnitude.numpy(), color='red', label='Hacker Result')
    plt.title(f"The Hack: Optimized for Brightness (Energy={max_brightness:.1f})")
    plt.ylim(0, 2.0)
    plt.legend()
    
    # Plot 3: The Owner's Key
    plt.subplot(3, 1, 3)
    plt.bar(range(N), owner_magnitude.numpy(), color='blue', label='Owner Result')
    plt.title("The Owner: Correct Key Decryption")
    plt.ylim(0, 1.2)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('security_audit.png')
    print("\n   [Result] Audit Image Saved. Check 'security_audit.png'.")
    print("   Look at the Red Graph. Does it match the Green Graph?")

if __name__ == "__main__":
    run_security_audit()