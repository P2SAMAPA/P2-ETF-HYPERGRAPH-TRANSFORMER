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

        self.q_linear = nn.Linear(in_dim, out_dim)
        self.k_linear = nn.Linear(in_dim, out_dim)
        self.v_linear = nn.Linear(in_dim, out_dim)
        self.out_linear = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x_etf, hyperedge_index):
        # x_etf: (N, in_dim)
        # hyperedge_index: list of lists of ETF indices per hyperedge
        # Compute hyperedge features by aggregating member ETF features (mean + attention)
        N = x_etf.size(0)
        device = x_etf.device
        num_hedges = len(hyperedge_index)

        # Build hyperedge-to-ETF bipartite edges for message passing
        # We'll compute attention for each ETF over its hyperedges
        # First, compute hyperedge features via simple mean pooling of member ETFs
        hedge_feats = []
        for members in hyperedge_index:
            if members:
                member_feats = x_etf[members]  # (|members|, in_dim)
                hedge_feats.append(member_feats.mean(dim=0))
            else:
                hedge_feats.append(torch.zeros(x_etf.size(1), device=device))
        hedge_feats = torch.stack(hedge_feats, dim=0)  # (num_hedges, in_dim)

        # Update ETF features via attention over connected hyperedges
        # For each ETF, gather hyperedges it belongs to
        # We'll compute new ETF features as weighted sum of hyperedge features
        # Use multi-head attention: query = ETF, key/value = hyperedge
        q = self.q_linear(x_etf).view(N, self.num_heads, self.head_dim)  # (N, H, D)
        k = self.k_linear(hedge_feats).view(num_hedges, self.num_heads, self.head_dim)  # (M, H, D)
        v = self.v_linear(hedge_feats).view(num_hedges, self.num_heads, self.head_dim)

        # Build adjacency mask: for each ETF, which hyperedges it belongs to
        # We'll compute attention in a loop over ETFs due to varying degrees
        new_x = torch.zeros_like(x_etf)
        for i in range(N):
            # Find hyperedges containing ETF i
            hedge_indices = [h for h, members in enumerate(hyperedge_index) if i in members]
            if not hedge_indices:
                new_x[i] = x_etf[i]
                continue
            # Gather keys/values for those hyperedges
            k_i = k[hedge_indices]  # (degree, H, D)
            v_i = v[hedge_indices]
            q_i = q[i:i+1]          # (1, H, D)

            # Scaled dot-product attention
            attn_scores = torch.einsum('nhd,mhd->nhm', q_i, k_i) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)  # (1, H, degree)
            attn_weights = self.dropout(attn_weights)
            out_i = torch.einsum('nhm,mhd->nhd', attn_weights, v_i)  # (1, H, D)
            out_i = out_i.reshape(-1, self.out_dim)
            new_x[i] = self.out_linear(out_i)

        x_etf = self.layer_norm(x_etf + new_x)
        return x_etf

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
        # Move all snapshots to device (won't store all in GPU, we process one by one)
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
