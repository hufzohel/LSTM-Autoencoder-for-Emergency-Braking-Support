import torch
import torch.nn as nn

class LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=4):
        super(LSTM_Autoencoder, self).__init__()
        
        # 1. THE ENCODER LSTM
        # Math: Allocates matrices W_ii, W_if, W_ig, W_io (Input, Forget, Cell, Output gates)
        # Dimensions: W are roughly [input_dim x hidden_dim]
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        
        # 2. THE BOTTLENECK (Compression)
        # Math: z = W_e * h + b_e
        # We are projecting from 64 dimensions (Hidden) down to 16 (Latent)
        self.encoder_linear = nn.Linear(hidden_dim, latent_dim)

        # 3. THE DECODER SETUP (Decompression)
        # Math: h_prime = W_d * z + b_d
        # We project from 16 (Latent) back up to 64 (Hidden) to start the reconstruction
        self.decoder_linear = nn.Linear(latent_dim, hidden_dim)
        
        # 4. THE DECODER LSTM
        # Math: Allocates new Weight matrices for the reconstruction phase
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # 5. THE OUTPUT MAPPING
        # Math: x_hat = W_out * h_decoded + b_out
        # Maps the hidden state (64) back to the physical sensors (3: Speed, Gas, Brake)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: [Batch_Size, Sequence_Len, 3]

        # We only care about the FINAL hidden state (h_n) of the sequence
        # h_n shape: [1, Batch_Size, Hidden_Dim]
        _, (h_n, _) = self.encoder_lstm(x)
        h_n = h_n.squeeze(0)
        latent = self.encoder_linear(h_n) 
        hidden_decoded = self.decoder_linear(latent)
        
        # Expand to match sequence length (Repeat the vector)
        # We repeat the context for every time step
        seq_len = x.shape[1]       
        repeated_input = hidden_decoded.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_output, _ = self.decoder_lstm(repeated_input)
        
        # 4. FINAL MAPPING
        # Map pixels back to GBSA values
        reconstructed = self.output_layer(decoder_output)
        
        return reconstructed