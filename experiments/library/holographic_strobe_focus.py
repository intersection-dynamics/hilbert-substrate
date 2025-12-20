import torch
import numpy as np
import matplotlib.pyplot as plt

# --- 1. YOUR OPTIMIZER LOGIC (Adapted for PyTorch Tensors) ---
class StrobeOptimizer:
    """
    The 'Strobe' Engine adapted for Holographic Phase Alignment.
    Minimizes 'Locality Cost' (Destructive Interference).
    """
    def __init__(self, cost_function, initial_state, entropy_sensitivity=0.1):
        self.cost_func = cost_function
        self.state = initial_state.clone() # The Phase Correction Tensor
        self.sensitivity = entropy_sensitivity
        
        self.current_cost = self.cost_func(self.state)
        self.best_state = self.state.clone()
        self.min_cost = self.current_cost

    def step(self, t, mutation_strength=0.05):
        # 1. ENTROPY BUILD (The Fluctuation)
        # We apply random phase jitter ("Heat") to the connection
        noise = (torch.rand_like(self.state) - 0.5) * 2 * mutation_strength
        candidate_state = self.state + noise
        
        # Calculate Energy (Negative Coherence)
        candidate_cost = self.cost_func(candidate_state)
        delta_E = candidate_cost - self.current_cost
        
        # 2. THE STROBE (The Selection Rule)
        accepted = False
        
        # If lower energy (more coherent), we "Fall" into the state immediately.
        if delta_E < 0:
            accepted = True
        else:
            # If higher energy, we might accept it to escape local minima (Tunneling)
            # Cooling Schedule: Sensitivity drops as 't' increases
            temperature = self.sensitivity / (1.0 + t / 500.0) 
            probability = np.exp(-delta_E.item() / temperature)
            
            if np.random.rand() < probability:
                accepted = True

        if accepted:
            self.state = candidate_state
            self.current_cost = candidate_cost
            
            # 3. THE POP (Global Minimum Check)
            if self.current_cost < self.min_cost:
                self.min_cost = self.current_cost
                self.best_state = self.state.clone()
                return True, self.current_cost # "POP" event
                
        return False, self.current_cost

# --- 2. THE HARDWARE SIMULATION ---
def get_coherence_cost(correction_tensor, distorted_signal):
    """
    The Cost Function: Energy of Destructive Interference.
    We want MAX coherence, so we return MIN negative energy.
    """
    # Apply the lens (correction)
    lens = torch.exp(-1j * correction_tensor)
    corrected_signal = distorted_signal * lens
    
    # Measure total constructive interference (The "Glow")
    magnitude = torch.abs(corrected_signal.sum())
    
    # We minimize negative magnitude
    return -magnitude

def run_strobe_alignment():
    print("="*60)
    print("      STROBE OPTIMIZATION: ANNEALING THE CONNECTION")
    print("="*60)
    
    # Setup: 64 Channels, imperfect crystal
    device = torch.device('cpu')
    N = 64
    true_key = torch.linspace(0, 10, N)
    drift_noise = torch.randn(N) * 1.5 # Heavy distortion
    
    # The "Reality" (Signal + Noise)
    raw_signal = torch.exp(1j * (true_key + drift_noise))
    
    # Initial State: A flat lens (No correction)
    initial_lens = torch.zeros(N)
    
    # Create the Strobe Optimizer
    # We pass a lambda function to bind the specific signal to the cost check
    optimizer = StrobeOptimizer(
        cost_function=lambda x: get_coherence_cost(x, raw_signal),
        initial_state=initial_lens,
        entropy_sensitivity=5.0 # High initial heat
    )
    
    print(f"   [System] Initial Coherence: {-optimizer.current_cost.item():.2f}")
    
    history = []
    pops = []
    
    # Run the "Strobe" loop
    print("   [System] Strobe Cycle Initiated...")
    for t in range(2000):
        is_pop, cost = optimizer.step(t, mutation_strength=0.1)
        real_score = -cost.item()
        history.append(real_score)
        
        if is_pop:
            pops.append((t, real_score))
            if t % 100 == 0:
                print(f"      T={t}: POP! New Stable State Found (Energy: {real_score:.2f})")

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 5))
    plt.plot(history, color='#444444', alpha=0.6, label='System State')
    
    # Plot "Pop" events
    pop_x, pop_y = zip(*pops)
    plt.scatter(pop_x, pop_y, color='cyan', s=15, label='Pop Events (New Minima)')
    
    plt.title("Strobe Optimization: Thermodynamic Settling of Optical Phase")
    plt.xlabel("Time Steps (t)")
    plt.ylabel("Signal Coherence (Energy)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig('strobe_alignment.png')
    
    print("\n" + "="*60)
    print(f"   FINAL RESULT:")
    print(f"   Starting Coherence: {-history[0]:.2f}")
    print(f"   Final Coherence:    {history[-1]:.2f}")
    print(f"   Improvement Factor: {history[-1] / -history[0]:.1f}x")
    print("   Alignment Complete. Connection Locked.")
    print("="*60)

if __name__ == "__main__":
    run_strobe_alignment()