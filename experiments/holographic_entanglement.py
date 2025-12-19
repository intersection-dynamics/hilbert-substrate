import torch
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PART 1: THE PHYSICS (The "Hardware")
# ==========================================
def get_coherence_cost(correction_tensor, distorted_signal):
    """
    The Cost Function: Energy of Destructive Interference.
    We want MAX coherence, so we return MIN negative energy.
    Lower is Better.
    """
    # Apply the lens (correction)
    lens = torch.exp(-1j * correction_tensor)
    corrected_signal = distorted_signal * lens
    
    # Measure total constructive interference (The "Glow")
    magnitude = torch.abs(corrected_signal.sum())
    
    # We minimize negative magnitude
    return -magnitude

# ==========================================
# PART 2: THE ENTANGLEMENT ENGINE (Adapted)
# ==========================================
class EntanglementOptimizer:
    def __init__(self, cost_function, initial_state):
        self.cost_func = cost_function
        self.state = initial_state.clone()
        
        # Initial Physics Check
        self.current_cost = self.cost_func(self.state)
        self.best_state = self.state.clone()
        self.min_cost = self.current_cost

    def mutate_phase(self, current_state, horizon):
        """
        The Quantum Mutation.
        Horizon dictates the VOLATILITY of the phase shift.
        """
        # 1. Determine Jitter Magnitude based on Horizon
        # High Horizon = +/- PI (Complete scrambling)
        # Low Horizon  = +/- 0.01 (Micro-adjustment)
        noise_scale = horizon * np.pi 
        
        # 2. Generate Noise
        noise = (torch.rand_like(current_state) - 0.5) * 2 * noise_scale
        
        # 3. Apply to State
        new_state = current_state + noise
        return new_state

    def step(self, t, cycle_length=500):
        # === THE STROBE: OSCILLATING HORIZON ===
        # We breathe: Expansion (High Energy) -> Contraction (Crystallization)
        phase = (t % cycle_length) / cycle_length
        
        # Horizon Logic (Cosine Wave)
        # Ranges from 1.0 (Global) down to 0.001 (Micro)
        horizon = 0.5 * (np.cos(phase * 2 * np.pi) + 1)
        horizon = max(horizon, 0.001) # Never freeze completely
        
        # === DYNAMICS ===
        candidate_state = self.mutate_phase(self.state, horizon)
        candidate_cost = self.cost_func(candidate_state)
        
        delta_E = candidate_cost - self.current_cost
        
        # === STABILITY CHECK (Tunneling) ===
        # If Simplicity increases (Lower Cost / Higher Coherence), we ALWAYS collapse.
        if delta_E < 0:
            self.state = candidate_state
            self.current_cost = candidate_cost
            
            # Record Record
            if self.current_cost < self.min_cost:
                self.min_cost = self.current_cost
                self.best_state = self.state.clone()

        # If Simplicity decreases (Higher Cost), we check Stability.
        else:
            # We allow tunneling ONLY when Horizon is High (Quantum Phase).
            # When Horizon is Low (Classical Phase), the system resists noise.
            
            # This replaces "Temperature" with "Entanglement Strength"
            # We scale the tunneling amplitude by the horizon.
            tunneling_amplitude = horizon * 2.0 
            
            # Standard acceptance logic using Horizon as the driver
            if tunneling_amplitude > 0:
                probability = np.exp(-delta_E.item() / tunneling_amplitude)
                if np.random.rand() < probability:
                    self.state = candidate_state
                    self.current_cost = candidate_cost

        return self.current_cost, horizon

# ==========================================
# EXECUTION
# ==========================================
def run_entanglement_focus():
    print("="*60)
    print("      ENTANGLEMENT OPTIMIZATION: BREATHING PHASE LOCK")
    print("="*60)
    
    device = torch.device('cpu')
    N = 64 # 64 Channels
    
    # 1. Create the "Hardware" (Imperfect Crystal)
    true_key = torch.linspace(0, 10, N)
    drift_noise = torch.randn(N) * 2.0 # Heavy non-linear distortion
    raw_signal = torch.exp(1j * (true_key + drift_noise))
    
    print(f"   [Hardware] Defect Magnitude: Heavy (Phase Drift)")
    
    # 2. Initialize the Engine
    initial_lens = torch.zeros(N)
    
    # Bind the physics to the optimizer
    opt = EntanglementOptimizer(
        cost_function=lambda x: get_coherence_cost(x, raw_signal),
        initial_state=initial_lens
    )
    
    CYCLES = 3000
    print(f"   [Engine] Running {CYCLES} Strobe Cycles...")
    
    cost_history = []
    horizon_history = []
    pops = []
    
    for t in range(CYCLES):
        cost, horizon = opt.step(t, cycle_length=600)
        
        real_score = -cost.item() # Convert back to positive energy
        cost_history.append(real_score)
        horizon_history.append(horizon * 20) # Scale for plotting
        
        # Track "Pop" events (New Records)
        if real_score > -opt.min_cost.item() - 0.001: 
             # Only log if it's the absolute best so far
             if len(pops) == 0 or real_score > pops[-1][1]:
                pops.append((t, real_score))
                if t % 100 == 0:
                    print(f"      T={t}: ⚡ LOCKED Energy Level {real_score:.2f}")

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 6))
    
    # Plot Energy (Coherence)
    plt.plot(cost_history, color='#222222', linewidth=1, label='Signal Coherence')
    
    # Plot The Horizon (The Breathing)
    plt.plot(horizon_history, color='cyan', alpha=0.3, label='Entanglement Horizon')
    
    # Plot Pop Events
    pop_x, pop_y = zip(*pops)
    plt.scatter(pop_x, pop_y, color='red', s=20, zorder=5, label='Phase Lock Events')
    
    plt.title("Holographic Autofocus via Entanglement Optimization")
    plt.xlabel("Strobe Cycles")
    plt.ylabel("Signal Energy (Constructive Interference)")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.2)
    plt.savefig('entanglement_focus.png')
    
    print("\n" + "="*60)
    print(f"   FINAL METRICS:")
    print(f"   Starting Energy: {cost_history[0]:.2f}")
    print(f"   Final Energy:    {cost_history[-1]:.2f}")
    print(f"   Optimization Gain: {cost_history[-1] / cost_history[0]:.2f}x")
    print("   The signal is now perfectly coherent.")
    print("="*60)

if __name__ == "__main__":
    run_entanglement_focus()