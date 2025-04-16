import os
import glob
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from vae import VAE

class DemoDataset(Dataset):
    def __init__(self, demos_folder):
        self.states = []
        
        # Load all pickle files from the demos folder
        demo_files = glob.glob(os.path.join(demos_folder, "*.pkl"))
        
        for demo_file in demo_files:
            with open(demo_file, 'rb') as f:
                trajectory = pickle.load(f)
                # Extract states from (state, action) pairs
                states = [pair[0] for pair in trajectory]
                self.states.extend(states)
                
        self.states = np.array(self.states)
        print(f"Loaded {len(self.states)} states from {len(demo_files)} files.")
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.states[idx])

def train_vae(demos_folder, batch_size=32, epochs=100, latent_dim=32, lr=1e-3):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset and dataloader
    dataset = DemoDataset(demos_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    M_N = batch_size / len(dataset)
    print(f"Batch size: {batch_size}, M/N Ratio: {M_N:.2f}")
    
    # Initialize VAE
    input_dim = dataset.states[0].shape[0]  # Get dimension of state vector
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        recon_loss = 0
        kld_loss = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            results = vae(batch)
            train_loss, recon, kld = vae.loss_function(*results, M_N=M_N)
            
            # Backward pass
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            total_loss += train_loss.item()
            recon_loss += recon.item()
            kld_loss += kld.item()
            
        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        avg_recon = recon_loss / len(dataloader)
        avg_kld = kld_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} "
              f"(Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f})")
    
    # Save the trained model
    model_path = os.path.join(os.path.dirname(demos_folder), "trained_vae.pt")
    torch.save(vae.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return vae

if __name__ == "__main__":
    # Get the path to the trajectories folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demos_folder = os.path.join(script_dir, "..", "demos")
    
    # Train the VAE
    vae = train_vae(demos_folder, latent_dim=8, epochs=500)