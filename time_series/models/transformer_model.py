import torch
import torch.nn as nn
import math
import numpy as np
import logging
from time_series.utils.evaluation import time_model_execution

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: [seq_len, batch_size, d_model]"""
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, output_size, dropout=0.1, seq_length=7):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.encoder = nn.Linear(input_size, d_model) # Input embedding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length + 20) # Increased max_len margin slightly for safety
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(d_model, output_size) # New: Use output of one token (e.g., first or CLS equivalent)
        self.init_weights()
        self.seq_length = seq_length # seq_length is used by pos_encoder, not directly here anymore for decoder input size

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """src: [batch_size, seq_len, input_features]"""
        src = self.encoder(src) * math.sqrt(self.d_model)
        
        src = src.permute(1, 0, 2) # [seq_len, batch_size, d_model]
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2) # [batch_size, seq_len, d_model]
        
        output = self.transformer_encoder(src)
        # output is [batch_size, seq_len, d_model]
        
        # Use the output of the first token for prediction
        # This assumes the first token can learn to aggregate sequence information for the task
        output = self.decoder(output[:, 0, :]) # Using output of the first token in the sequence
        # Alternative: use the last token output[:, -1, :]
        return output

class TransformerPredictor:
    def __init__(self, input_size, d_model=64, nhead=4, num_encoder_layers=2, 
                 dim_feedforward=128, output_size=1, dropout=0.1, seq_length=7, learning_rate=0.001, device=None):
        self.logger = logging.getLogger('transformer_model')
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerModel(input_size, d_model, nhead, num_encoder_layers, 
                                      dim_feedforward, output_size, dropout, seq_length).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.logger.info(f"TransformerModel initialized on {self.device}")
        self.logger.info(f"Model params: input_size={input_size}, d_model={d_model}, nhead={nhead}, layers={num_encoder_layers}, ff_dim={dim_feedforward}, output_size={output_size}, seq_len={seq_length}")

    @time_model_execution
    def train(self, train_loader, val_loader, epochs=50):
        self.logger.info(f"Starting Transformer training for {epochs} epochs.")
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            self.logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict()
                self.logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving model state.")

        if best_model_state:
            self.model.load_state_dict(best_model_state)
            self.logger.info("Loaded best model state for Transformer.")
        
        return {'train_losses': train_losses, 'val_losses': val_losses, 'best_val_loss': best_val_loss}

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                # Squeeze might be problematic if batch_size is 1 and output_size is 1
                # Ensure outputs are correctly shaped before extending
                if outputs.ndim == 1: # if outputs is [batch_size] after squeeze for single output value
                    outputs = outputs.unsqueeze(1) # make it [batch_size, 1]
                if y_batch.ndim == 1:
                    y_batch = y_batch.unsqueeze(1)

                predictions.extend(outputs.cpu().numpy().tolist()) # .tolist() will handle nested lists if any
                actuals.extend(y_batch.cpu().numpy().tolist())
        
        # Convert to flat numpy arrays then reshape to (N,1)
        # This handles cases where elements might be single-element lists from .tolist()
        predictions_np = np.array([p[0] if isinstance(p, list) else p for p in predictions]).reshape(-1, 1)
        actuals_np = np.array([a[0] if isinstance(a, list) else a for a in actuals]).reshape(-1, 1)
            
        return predictions_np, actuals_np 