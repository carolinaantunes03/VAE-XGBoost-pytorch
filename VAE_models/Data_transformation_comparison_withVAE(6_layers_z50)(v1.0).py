import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
import matplotlib.pyplot as plt

# 1. Configuration e Hyperparameters
CONFIG = {
    'layer_dims': [6000, 3000, 1000, 500, 100],
    #'layer_dims': [600, 300, 100, 50, 20],
    'latent_dim': 50,
    'batch_size': 64,
    #'batch_size': None,
    'epochs': 400,
    'learning_rate': 0.002,
    'test_split': 0.1,
    'seed': 42,
    'alpha': 1.0, # KL weight multiplier
    'kappa': 0.002, 
#    'device': torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu") 

}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])


# 2. Data Loading & Preprocessing
def load_and_process_data(filepath):
    print(f"Loading data from: {filepath}")
    
    
    if not os.path.exists(filepath):
        print("Warning: File not found. Generating dummy data for demonstration.")
        
        data = pd.DataFrame(np.random.rand(100, 20531)) 
    else:
        data = pd.read_csv(filepath, sep="\t", index_col=0)
        data = data.dropna(axis='columns')

    print(f"Input dimensions: {data.shape}")
    
    scaler = preprocessing.MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Convert to DataFrame to keep indices
    df_scaled = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
    
    # Split Data
    test_data = df_scaled.sample(frac=CONFIG['test_split'], random_state=CONFIG['seed'])
    train_data = df_scaled.drop(test_data.index)
    
    return train_data, test_data, df_scaled.shape[1], df_scaled.index, test_data.index

# 3. Model Definition
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
        
        # --- Encoder ---
        # Matches Keras: Dense -> BatchNorm -> ReLU
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim, eps=1e-5, momentum=0.99)) 
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
            
        self.encoder_body = nn.Sequential(*encoder_layers)
        
        # Latent space heads
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # --- Decoder ---
        # Matches Keras: Dense -> ReLU (No Batch Norm in original decoder code)
        decoder_layers = []
        reversed_hidden = hidden_dims[::-1]
        
        # Connect latent to first hidden layer
        decoder_layers.append(nn.Linear(latent_dim, reversed_hidden[0]))
        decoder_layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(reversed_hidden) - 1):
            decoder_layers.append(nn.Linear(reversed_hidden[i], reversed_hidden[i+1]))
            decoder_layers.append(nn.ReLU())
            
        # Output layer (reconstructing original_dim)
        decoder_layers.append(nn.Linear(reversed_hidden[-1], input_dim))
        decoder_layers.append(nn.ReLU()) # Original code used ReLU at output
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        h = self.encoder_body(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z

# 4. Custom Loss Function
def loss_function(recon_x, x, mu, logvar, beta, alpha=1.0):
    # Reconstruction loss: Mean Absolute Error per sample, summed over features
    # Keras logic: original_dim * mean(abs(x - recon)) -> effectively Sum(abs(x-recon))
    recon_loss = torch.sum(torch.abs(recon_x - x), dim=1).mean()
    
    # KL Divergence
    # -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    return recon_loss + (alpha * beta * kl_loss), recon_loss, kl_loss

# 5. Main Execution
def main():
  
    file_path = 'TCGA_BRCA_VSTnorm_count_expr_clinical_data.txt' # Path to data file
    
    train_df, test_df, input_dim, all_indices, test_indices = load_and_process_data(file_path)

    #CONFIG['batch_size'] = len(train_df) #changed due to computational limits

    # Convert to Tensors
    train_tensor = torch.FloatTensor(train_df.values).to(CONFIG['device'])
    test_tensor = torch.FloatTensor(test_df.values).to(CONFIG['device'])
    
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=CONFIG['batch_size'], shuffle=True)

    
    # Initialize Model 
    model = VAE(input_dim, CONFIG['layer_dims'], CONFIG['latent_dim']).to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    print(f"Model architecture on {CONFIG['device']}:")
    print(model)

    # Training Loop
    print("\nStarting Training...")
    beta = 0.0 # Warmup variable
    start_time = time.time()
    history = {'loss': [], 'recon_loss': [], 'kl_loss': []}

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        
        beta = min(1.0, epoch * CONFIG['kappa'])
        
        # Update Beta (Warmup)
        if beta <= 1.0:
            beta += CONFIG['kappa']
            
        for batch in train_loader:
            x_batch = batch[0]
            
            optimizer.zero_grad()
            recon_batch, mu, logvar, _ = model(x_batch)
            
            loss, recon, kl = loss_function(recon_batch, x_batch, mu, logvar, beta, CONFIG['alpha'])
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Calculate average loss per epoch
        avg_loss = train_loss / len(train_loader)
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 1 == 0:
            print(f'Epoch {epoch+1}/{CONFIG["epochs"]} | Loss: {avg_loss:.4f} | Beta: {beta:.3f}')

    print(f"Training time: {time.time() - start_time:.2f} seconds")

    #  Evaluation & Extraction 
    model.eval()
    with torch.no_grad():
        # Encode ALL samples (train + test)
        full_df = pd.concat([train_df, test_df])  # ← use concat instead of append
        full_tensor = torch.FloatTensor(full_df.values).to(CONFIG['device'])
        _, _, _, z_all = model(full_tensor)

        # Get reconstruction for test set (optional)
        recon_test, _, _, _ = model(torch.FloatTensor(test_df.values).to(CONFIG['device']))

        reconstruction_fidelity = torch.abs(torch.FloatTensor(test_df.values).to(CONFIG['device']) - recon_test).cpu().numpy()
        mean_recon_loss = np.mean(reconstruction_fidelity)
        print(f"\nMean Reconstruction Loss (Test): {mean_recon_loss:.6f}")

    #  Save Latent Space 
    z_df = pd.DataFrame(z_all.cpu().numpy(), index=full_df.index)
    z_df.columns = z_df.columns + 1
    z_df.columns.name = 'sample_id'
    
    out_dir = os.path.join("counts_data", "vae_compressed")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "encoded_BRCA_VAE_z50_pytorch_exp3.tsv")
    
    z_df.to_csv(save_path, sep='\t')
    print(f"Encoded data saved to: {save_path}")
    print(f"Output shape: {z_df.shape}")

    # Plotting History
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Total Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("training_history.pdf")
    plt.show()



if __name__ == "__main__":
    main()