import torch
import torch.nn as nn
import numpy as np

class LSTMRegressor(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_layers=2, out_features=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_features)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class GRURegressor(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_layers=2, out_features=3, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_features)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class CNNBiLSTMRegressor(nn.Module):
    def __init__(self, in_features, cnn_out=64, lstm_hidden=32, out_features=3, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=cnn_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_out)
        self.relu = nn.ReLU()
        self.bilstm = nn.LSTM(input_size=cnn_out, hidden_size=lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, out_features)
        )
    def forward(self, x):
        x_trans = x.transpose(1, 2)
        conv_feat = self.relu(self.bn1(self.conv1(x_trans)))
        conv_feat = conv_feat.transpose(1, 2)
        out, _ = self.bilstm(conv_feat)
        return self.fc(out[:, -1, :])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerRegressor(nn.Module):
    def __init__(self, in_features, d_model=64, nhead=4, num_layers=2, out_features=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, out_features)
        )
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        trans_out = self.transformer(x)
        pooled = torch.mean(trans_out, dim=1)
        return self.fc(pooled)

class ResidualMLPRegressor(nn.Module):
    def __init__(self, in_features, window_size=12, hidden_dim=128, out_features=3, dropout=0.15):
        super().__init__()
        self.flatten_dim = in_features * window_size
        self.in_proj = nn.Linear(self.flatten_dim, hidden_dim)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        
        self.out_head = nn.Linear(hidden_dim, out_features)
        
    def forward(self, x):
        b = x.shape[0]
        h = self.in_proj(x.view(b, -1))
        res1 = h
        h = self.drop1(self.act1(self.fc1(self.ln1(h)))) + res1
        res2 = h
        h = self.drop2(self.act2(self.fc2(self.ln2(h)))) + res2
        return self.out_head(h)
