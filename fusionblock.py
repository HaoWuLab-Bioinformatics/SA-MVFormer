import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv



class GNNFusion(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, gnn_type="gcn", heads=1, dropout=0.5, 
                 gnn_layer=2, gcn_residual=True):
        super(GNNFusion, self).__init__()
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        self.gnn_layer = gnn_layer
        self.gcn_residual = gcn_residual

        if self.gnn_type == "gcn":
            self.gcn_convs = nn.ModuleList()
            self.gcn_dropouts = nn.ModuleList()
            
            self.gcn_convs.append(GCNConv(in_dim, hidden_dim, cached=True, normalize=True))
            self.gcn_dropouts.append(nn.Dropout(dropout))
            
            for _ in range(gnn_layer - 2):
                self.gcn_convs.append(GCNConv(hidden_dim, hidden_dim, cached=True, normalize=True))
                self.gcn_dropouts.append(nn.Dropout(dropout))
            
            if gnn_layer > 1:
                self.gcn_convs.append(GCNConv(hidden_dim, out_dim, cached=True, normalize=True))
                self.gcn_dropouts.append(nn.Dropout(dropout))
            
            if gcn_residual and in_dim != out_dim:
                self.residual_proj = nn.Linear(in_dim, out_dim)
            else:
                self.residual_proj = None

        elif self.gnn_type == "sage":
            self.sage_layers = nn.ModuleList()
            dims = [in_dim] + [hidden_dim] * max(0, gnn_layer - 1) + [out_dim]
            for i in range(len(dims) - 1):
                self.sage_layers.append(SAGEConv(dims[i], dims[i + 1]))

        else:
            raise ValueError(f"Unsupported gnn_type: {gnn_type}. Choose from ['gcn', 'gat', 'sage'].")

    def forward(self, x, edge_index_or_adj):
        if self.gnn_type == "gcn":
            residual = x
            

            for i, (conv, dropout) in enumerate(zip(self.gcn_convs, self.gcn_dropouts)):
                x = conv(x, edge_index_or_adj)
                
                if i < len(self.gcn_convs) - 1:
                    x = F.relu(x)
                    x = dropout(x)
            

            if self.gcn_residual:
                if self.residual_proj is not None:
                    residual = self.residual_proj(residual)
                x = x + residual
            
            return x

        elif self.gnn_type == "sage":
            for i, conv in enumerate(self.sage_layers):
                x = conv(x, edge_index_or_adj)
                if i < len(self.sage_layers) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            return x
        else:
            raise ValueError(f"Unsupported gnn_type: {self.gnn_type}")


class fusionblock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, gnn_type="gcn", heads=1, dropout=0.5,
                 fusion="weighted", graph_weight=0.8, gnn_layer=2, gcn_residual=True,
                 fusion_norm_before: bool = False):
        super(fusionblock, self).__init__()
        self.gnn = GNNFusion(in_dim, hidden_dim, out_dim, gnn_type, heads, dropout, 
                           gnn_layer, gcn_residual)
        self.fusion = fusion
        self.graph_weight = graph_weight
        self.fusion_norm_before = fusion_norm_before

        if self.fusion_norm_before:
            self.ln_gnn = nn.LayerNorm(out_dim)
            self.ln_attn = nn.LayerNorm(out_dim)

        if fusion == "concat":
            self.proj = nn.Linear(2 * out_dim, out_dim)

    def forward(self, x, edge_index, attn_out):
        gnn_out = self.gnn(x, edge_index)

        if self.fusion_norm_before:
            gnn_out = self.ln_gnn(gnn_out)
            attn_out = self.ln_attn(attn_out)

        if self.fusion == "concat":
            out = torch.cat([gnn_out, attn_out], dim=-1)
            out = self.proj(out)

        elif self.fusion == "weighted":
            out = self.graph_weight * gnn_out + (1 - self.graph_weight) * attn_out

        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion}")

        return out

