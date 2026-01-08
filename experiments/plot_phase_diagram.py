import matplotlib.pyplot as plt
import numpy as np

def plot_figure_1_phase_diagram():
    # --- DATA (From N=5 Phase Diagram Sweep) ---
    p_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    
    # Theoretical Minimum (Harmonions - Eigenbasis)
    harmonion_cost = np.array([1.13, 1.50, 2.60, 6.01, 17.10, 54.54])
    
    # Spatial Reference (1D Ring)
    spatial_target = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    
    # Experimentally Recovered Cost
    recovered_cost = np.array([1.22, 2.11, 6.62, 19.68, 61.99, 273.55])

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    
    # Plot Lines
    plt.semilogy(p_values, harmonion_cost, 'g--o', label='Harmonion Ideal (Global Min)', linewidth=2, alpha=0.7)
    plt.semilogy(p_values, spatial_target, 'b-s', label='Spatial Target (1D Ring)', linewidth=2, alpha=0.7)
    plt.semilogy(p_values, recovered_cost, 'r-^', label='Recovered Cost (Dynamic)', linewidth=3)

    # --- ANNOTATIONS ---
    
    # 1. The Window of Reality (Region where Recovered ~= Spatial)
    plt.axvspan(2.8, 4.2, color='blue', alpha=0.1)
    plt.text(3.5, 300, "Window of Reality\n(Emergent Geometry)", horizontalalignment='center', color='darkblue', fontsize=11, fontweight='bold')

    # 2. Quantum Fluid Phase (p <= 2)
    plt.text(1.5, 300, "Quantum Fluid\n(Delocalized)", horizontalalignment='center', color='green', fontsize=11, fontweight='bold')
    
    # 3. Glass Phase (p >= 5)
    plt.text(5.5, 400, "Glassy Disorder\n(Trapped)", horizontalalignment='center', color='darkred', fontsize=11, fontweight='bold')

    # Styling
    plt.xlabel('Locality Penalty Power ($p$)', fontsize=14)
    plt.ylabel('Locality Cost $\mathcal{C}_p$ (Log Scale)', fontsize=14)
    
    # CORRECTED TITLE
    plt.title('Figure 1: The Phase Diagram of Accessibility ($N=5$)', fontsize=16)
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=11, loc='lower right')
    
    # Adjust limits for visual clarity
    plt.ylim(0.9, 1000)
    plt.tight_layout()
    
    # Save
    plt.savefig('figure1_phase_diagram.pdf', dpi=300)
    plt.savefig('figure1_phase_diagram.png', dpi=300)
    print("Saved as figure1_phase_diagram.png")
    plt.show()

if __name__ == "__main__":
    plot_figure_1_phase_diagram()