import torch
import matplotlib.pyplot as plt
import os
from training.train_jepa import train_jepa_run

def run_ablations():
    mask_ratios = [0.3, 0.5, 0.7] # 3 ratios
    final_losses = []
    
    print("Starting Mask Ratio Ablations...")
    print(f"Ratios: {mask_ratios}")
    
    for ratio in mask_ratios:
        # Train for short duration (e.g. 2 epochs) as per requirements/constraints
        # "Train JEPA predictor with at least 3 mask ratios"
        loss = train_jepa_run(
            mask_ratio=ratio,
            epochs=2,
            save_name=f"jepa_loss_ratio_{ratio}"
        )
        final_losses.append(loss)
        print(f"Finished ratio {ratio}: Final Loss = {loss:.4f}")
        
    # Plot Comparison
    plt.figure()
    plt.plot(mask_ratios, final_losses, marker='o')
    plt.xlabel("Mask Ratio")
    plt.ylabel("Final Loss")
    plt.title("Ablation: Mask Ratio vs Final Loss")
    plt.grid(True)
    plt.savefig("plots/ablation_mask_ratio.png")
    plt.close()
    
    print("Ablation complete. Summary plot: plots/ablation_mask_ratio.png")

if __name__ == "__main__":
    run_ablations()
