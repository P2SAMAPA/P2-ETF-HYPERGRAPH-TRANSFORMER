"""
Hypergraph Transformer using attention over hyperedges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HypergraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        # Project input to out_dim if dimensions differ (needed for residual connection)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.q_linear = nn.Linear(out_dim, out_dim)
        self.k_linear = nn.Linear(out_dim, out_dim)
        self.v_linear = nn.Linear(out_dim, out_dim)
        self.out_linear = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x_etf, hyperedge_index):
        # x_etf: (N, in_dim)  -> project to out_dim
        x_proj = self.proj(x_etf)   # (N, out_dim)
        N = x_proj.size(0)
        device = x_proj.device
        num_hedges = len(hyperedge_index)

        # Hyperedge features via mean pooling of member ETFs
        hedge_feats = []
        for members in hyperedge_index:
            if members:
                member_feats = x_proj[members]  # (|members|, out_dim)
                hedge_feats.append(member_feats.mean(dim=0))
            else:
                hedge_feats.append(torch.zeros(self.out_dim, device=device))
        hedge_feats = torch.stack(hedge_feats, dim=0)  # (num_hedges, out_dim)

        # Multi-head attention
        q = self.q_linear(x_proj).view(N, self.num_heads, self.head_dim)         # (N, H, D)
        k = self.k_linear(hedge_feats).view(num_hedges, self.num_heads, self.head_dim)  # (M, H, D)
        v = self.v_linear(hedge_feats).view(num_hedges, self.num_heads, self.head_dim)

        new_x = torch.zeros(N, self.out_dim, device=device)
        for i in range(N):
            hedge_indices = [h for h, members in enumerate(hyperedge_index) if i in members]
            if not hedge_indices:
                new_x[i] = x_proj[i]
                continue
            k_i = k[hedge_indices]   # (degree, H, D)
            v_i = v[hedge_indices]
            q_i = q[i:i+1]           # (1, H, D)

            attn_scores = torch.einsum('nhd,mhd->nhm', q_i, k_i) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out_i = torch.einsum('nhm,mhd->nhd', attn_weights, v_i)  # (1, H, D)
            out_i = out_i.reshape(1, self.out_dim)
            new_x[i] = self.out_linear(out_i)

        x_out = self.layer_norm(x_proj + new_x)
        return x_out

class HypergraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HypergraphAttentionLayer(in_channels, hidden_channels, num_heads, dropout))
        for _ in range(num_layers - 1):
            self.layers.append(HypergraphAttentionLayer(hidden_channels, hidden_channels, num_heads, dropout))
        self.pred = nn.Linear(hidden_channels, 1)

    def forward(self, x_etf, hyperedge_index):
        for layer in self.layers:
            x_etf = layer(x_etf, hyperedge_index)
        return self.pred(x_etf).squeeze(-1)  # (N,)

class HGRunner:
    def __init__(self, in_channels, hidden_channels=64, num_heads=4, num_layers=2,
                 dropout=0.1, lr=0.001, seed=42):
        torch.manual_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HypergraphTransformer(in_channels, hidden_channels, num_heads, num_layers, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_snapshots(self, snapshots, epochs=80):
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            total_loss = 0.0
            for snap in snapshots:
                x = snap['x'].to(self.device)
                y = snap['y'].to(self.device)
                hedges = snap['hyperedge_index']
                pred = self.model(x, hedges)
                loss = self.criterion(pred, y)
                loss.backward()
                total_loss += loss.item()
            self.optimizer.step()
            avg_loss = total_loss / len(snapshots)
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    def predict_latest(self, snapshots):
        self.model.eval()
        snap = snapshots[-1]
        x = snap['x'].to(self.device)
        hedges = snap['hyperedge_index']
        with torch.no_grad():
            pred = self.model(x, hedges)
        return pred.cpu().numpy()
