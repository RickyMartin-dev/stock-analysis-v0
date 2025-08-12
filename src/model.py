import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden*2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden*2, hidden*2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden*2, hidden),
            nn.LeakyReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, 1),  # logits
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


# class Net(nn.Module):
#     def __init__(self, input_dim, hidden=64, p:float=0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden),
#             nn.ReLU(),
#             nn.Dropout(p),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Dropout(p),
#             nn.Linear(hidden, 1),  # logits
#         )
#     def forward(self, x):
#         return self.net(x).squeeze(-1)

#     def fit(self, X, y, epochs=20, batch_size=64, lr=1e-3):
#         ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.float32))
#         dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
#         opt = torch.optim.Adam(self.parameters(), lr=lr)
#         bce = nn.BCEWithLogitsLoss()
#         self.train()
#         for _ in range(epochs):
#             if _ % 10 == 0: print(f'update {_}/{epochs}')
#             for xb, yb in dl:
#                 opt.zero_grad()
#                 loss = bce(self(xb), yb)
#                 loss.backward(); opt.step()
#         return self

# class CalibratedSigmoid(nn.Module):
#     def __init__(self, base):
#         super().__init__()
#         self.base = base
#         self.a = nn.Parameter(torch.tensor(1.0))
#         self.b = nn.Parameter(torch.tensor(0.0))
#     def forward(self, x):
#         z = self.base(x)
#         return self.a * z + self.b
#     def predict_proba(self, X):
#         with torch.no_grad():
#             x = torch.tensor(X, dtype=torch.float32)
#             p = torch.sigmoid(self.forward(x)).numpy()
#         return p
#     def fit(self, X, y):
#         # quick platt-like calibration on logits
#         self.base.eval()
#         x = torch.tensor(X, dtype=torch.float32)
#         with torch.no_grad():
#             z = self.base(x)
#         y = torch.tensor(y, dtype=torch.float32)
#         opt = torch.optim.LBFGS([self.a, self.b], lr=0.1, max_iter=50)
#         def closure():
#             opt.zero_grad()
#             loss = nn.BCEWithLogitsLoss()(self.a*z + self.b, y)
#             loss.backward(); return loss
#         opt.step(closure)
#         return self