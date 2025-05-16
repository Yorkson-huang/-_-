import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from time_series.utils.evaluation import time_model_execution
import numpy as np

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size) # For Bahdanau-style
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_outputs, lstm_hidden_state):
        # lstm_outputs: (batch_size, seq_len, hidden_size)
        # lstm_hidden_state: (num_layers * num_directions, batch_size, hidden_size)
        # Assuming num_layers=1 and num_directions=1 for simplicity, take the last hidden state
        hidden = lstm_hidden_state[-1].unsqueeze(1) # (batch_size, 1, hidden_size)
        
        seq_len = lstm_outputs.size(1)
        # Repeat hidden state seq_len times
        hidden_repeated = hidden.repeat(1, seq_len, 1) # (batch_size, seq_len, hidden_size)
        
        # Calculate energy scores
        # Energy: tanh(W[h_prev, s_i])
        energy_input = torch.cat((hidden_repeated, lstm_outputs), dim=2) # (batch_size, seq_len, hidden_size * 2)
        energy = torch.tanh(self.attn(energy_input)) # (batch_size, seq_len, hidden_size)
        
        # Attention scores
        attention_scores = self.v(energy).squeeze(2) # (batch_size, seq_len)
        alpha = F.softmax(attention_scores, dim=1) # (batch_size, seq_len)
        
        # Context vector
        alpha = alpha.unsqueeze(1) # (batch_size, 1, seq_len)
        context_vector = torch.bmm(alpha, lstm_outputs).squeeze(1) # (batch_size, hidden_size)
        
        return context_vector, alpha.squeeze(1)

class CNNLSTMAttentionModel(nn.Module):
    def __init__(self, input_channels, seq_length, lstm_hidden_size, lstm_num_layers, output_size, 
                 cnn_out_channels=16, kernel_size=3):
        super(CNNLSTMAttentionModel, self).__init__()
        self.logger = logging.getLogger('cnn_lstm_attention_model')
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=cnn_out_channels, kernel_size=kernel_size, padding=(kernel_size -1)//2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) # Optional pooling
        
        # Calculate CNN output size for LSTM
        # Assuming input to CNN is (N, C_in, L_in) = (batch, input_channels, seq_length)
        # After Conv1d: (N, cnn_out_channels, L_in) (with padding)
        # After MaxPool1d(2,2): (N, cnn_out_channels, L_in // 2 + L_in % 2)
        # Corrected cnn_output_seq_len calculation for MaxPool1d
        if self.pool:
            self.cnn_output_seq_len = (seq_length + self.pool.padding * 2 - self.pool.kernel_size) // self.pool.stride + 1
        else:
            self.cnn_output_seq_len = seq_length
        
        self.lstm_input_size = cnn_out_channels

        self.lstm = nn.LSTM(input_size=self.lstm_input_size, 
                              hidden_size=lstm_hidden_size, 
                              num_layers=lstm_num_layers, 
                              batch_first=True)
        self.attention = Attention(lstm_hidden_size)
        self.fc = nn.Linear(lstm_hidden_size, output_size) # FC layer after attention

    def forward(self, x):
        # x shape: (batch_size, input_channels, seq_length)
        # self.logger.debug(f"Initial x shape: {x.shape}")
        x = self.conv1(x)
        x = self.relu(x)
        # self.logger.debug(f"After Conv1: {x.shape}")
        if self.pool:
             x = self.pool(x)
            # self.logger.debug(f"After Pool: {x.shape}")

        x = x.permute(0, 2, 1) # (batch_size, cnn_output_seq_len, cnn_out_channels)
        # self.logger.debug(f"After Permute for LSTM: {x.shape}")
        
        # Initialize LSTM hidden state (optional, LSTM does it by default if not provided)
        # h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        # c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        # lstm_out, (h_n, c_n) = self.lstm(x, (h0,c0))
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        context_vector, attention_weights = self.attention(lstm_out, h_n)
        # self.logger.debug(f"Context vector shape: {context_vector.shape}")
        
        out = self.fc(context_vector)
        return out

class CNNLSTMAttentionPredictor:
    def __init__(self, input_channels, seq_length, lstm_hidden_size=64, lstm_num_layers=1, 
                 output_size=1, cnn_out_channels=16, kernel_size=3, learning_rate=0.001, device=None):
        self.logger = logging.getLogger('cnn_lstm_attention_model')
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNLSTMAttentionModel(input_channels, seq_length, lstm_hidden_size, lstm_num_layers, 
                                           output_size, cnn_out_channels, kernel_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.logger.info(f"CNNLSTMAttentionModel initialized on {self.device}")
        self.logger.info(f"Model params: input_channels={input_channels}, seq_length={seq_length}, lstm_hidden={lstm_hidden_size}, cnn_out={cnn_out_channels}")

    @time_model_execution
    def train(self, train_loader, val_loader, epochs=50):
        self.logger.info(f"Starting CNN-LSTM-Attention training for {epochs} epochs.")
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            epoch_train_loss = 0
            for X_batch, y_batch in train_loader:
                # X_batch for CNN should be [batch, channels, seq_len]
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
            self.logger.info("Loaded best model state for CNN-LSTM-Attention.")
        
        return {'train_losses': train_losses, 'val_losses': val_losses, 'best_val_loss': best_val_loss}

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.extend(outputs.squeeze().cpu().numpy().tolist())
                actuals.extend(y_batch.squeeze().cpu().numpy().tolist())
        
        if not isinstance(predictions, list) or (predictions and not isinstance(predictions[0], (int, float))):
            predictions = np.array(predictions).flatten().tolist()
        if not isinstance(actuals, list) or (actuals and not isinstance(actuals[0], (int, float))):
            actuals = np.array(actuals).flatten().tolist()
            
        return np.array(predictions).reshape(-1,1), np.array(actuals).reshape(-1,1) 