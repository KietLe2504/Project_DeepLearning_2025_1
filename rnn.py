import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RNNRegression:
    def __init__(self, sequence_length=30, hidden_size=64, num_layers=2,
                 epochs=50, batch_size=32, learning_rate=0.001):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.last_sequence = None

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i+self.sequence_length])
            y_seq.append(y[i+self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values.reshape(-1, 1)
        elif len(y.shape) == 1:
            y = y.reshape(-1, 1)

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        if len(X_scaled) >= self.sequence_length:
            self.last_sequence = X_scaled[-self.sequence_length:]

        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)

        input_size = X.shape[1]
        self.model = RNNModel(input_size, self.hidden_size,
                             self.num_layers, output_size=1).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        dataset = TimeSeriesDataset(X_seq, y_seq)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}')

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if len(X) < self.sequence_length:
            if self.last_sequence is None:
                raise ValueError(f"Need at least {self.sequence_length} samples for prediction. "
                               f"Got {len(X)} samples and no stored sequence from training.")

            X_scaled = self.scaler_X.transform(X)

            current_sequence = self.last_sequence.copy()
            predictions = []

            for i in range(len(X_scaled)):
                seq_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)

                self.model.eval()
                with torch.no_grad():
                    pred_scaled = self.model(seq_tensor).cpu().numpy()

                predictions.append(pred_scaled[0])

                current_sequence = np.vstack([current_sequence[1:], X_scaled[i]])

            predictions = np.array(predictions)
            predictions = self.scaler_y.inverse_transform(predictions)
            return predictions.flatten()

        X_scaled = self.scaler_X.transform(X)

        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length + 1):
            X_seq.append(X_scaled[i:i+self.sequence_length])

        if len(X_seq) == 0:
            raise ValueError(f"Not enough data to create sequences. Need at least {self.sequence_length} samples.")

        X_seq = np.array(X_seq)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            predictions_scaled = self.model(X_tensor).cpu().numpy()

        predictions = self.scaler_y.inverse_transform(predictions_scaled)

        if len(predictions) < len(X):
            padding = np.full((len(X) - len(predictions), 1), predictions[-1])
            predictions = np.vstack([predictions, padding])

        return predictions.flatten()