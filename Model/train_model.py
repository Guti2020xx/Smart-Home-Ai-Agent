import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load dataset
df = pd.read_csv(r"../Data/smart_home_data.csv")
df.head()

action_vectors = [
    "Activate dehumidifier", 
    "No action needed", 
    "Set home to eco mode", 
    "Turn on AC",
    "Turn on heater"
]

action_to_idx = {
    "Activate dehumidifier": 0,
    "No action needed": 1,
    "Set home to eco mode": 2,
    "Turn on AC": 3,
    "Turn on heater": 4
}
df['action_idx'] = df['action'].map(action_to_idx)

seq_len = 15
features = ["inside_temp", "outside_temp", "humidity", "occupancy"]

X = []
y = []

for i in range(len(df) - seq_len):
    seq = df[features].iloc[i:i+seq_len].values   # shape: (seq_len, num_features)
    target = df['action_idx'].iloc[i + seq_len]   # next action
    X.append(seq)
    y.append(target)

X = np.array(X)  # (num_samples, seq_len, num_features)
y = np.array(y)  # (num_samples,)
print("X shape:", X.shape, "y shape:", y.shape)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define neural network
class SmartHomeTransformer(nn.Module):
    def __init__(self, input_dim=4, seq_len=15, d_model=64, nhead=4, num_layers=2, num_actions=5):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(d_model, num_actions)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_fc(x)            # (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)          # Transformer expects (seq_len, batch_size, d_model)
        x = self.transformer(x)
        x = x[-1, :, :]                 # take the last time step
        out = self.output_fc(x)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmartHomeTransformer(input_dim=4, seq_len=seq_len, num_actions=len(action_vectors)).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#actual training with batch size = 32 and 20 epochs
n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.4f}")


#exporting to onnx
dummy_input = torch.randn(1, seq_len, 4).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "smart_home_transformer.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=14
)
print("Model exported to ONNX!")
