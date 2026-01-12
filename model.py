import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from fusionblock import fusionblock
import random
import numpy as np


class GlobalAttn(torch.nn.Module):
    def __init__(self, hidden_channels, heads, num_layers, beta, dropout, qk_shared=True):
        super(GlobalAttn, self).__init__()
        self.hidden_channels = hidden_channels 
        self.heads = heads 
        self.num_layers = num_layers
        self.beta = beta 
        self.dropout = dropout
        self.qk_shared = qk_shared
        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(num_layers, heads*hidden_channels))
        else:
            self.betas = torch.nn.Parameter(torch.ones(num_layers, heads*hidden_channels)*self.beta)
        self.h_lins = torch.nn.ModuleList()
        if not self.qk_shared:
            self.q_lins = torch.nn.ModuleList()
        self.k_lins = torch.nn.ModuleList()
        self.v_lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        for i in range(num_layers):
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            if not self.qk_shared:
                self.q_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.k_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.v_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
        self.lin_out = torch.nn.Linear(heads*hidden_channels, heads*hidden_channels)

    def reset_parameters(self):
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        if not self.qk_shared:
            for q_lin in self.q_lins:
                q_lin.reset_parameters()
        for k_lin in self.k_lins:
            k_lin.reset_parameters()
        for v_lin in self.v_lins:
            v_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)
        self.lin_out.reset_parameters()

    def forward(self, x):
        seq_len, _ = x.size()
        for i in range(self.num_layers):
            h = self.h_lins[i](x)
            k = F.sigmoid(self.k_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
            if self.qk_shared:
                q = k
            else:
                q = F.sigmoid(self.q_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
            v = self.v_lins[i](x).view(seq_len, self.hidden_channels, self.heads)

            # numerator
            kv = torch.einsum('ndh, nmh -> dmh', k, v)
            num = torch.einsum('ndh, dmh -> nmh', q, kv)

            # denominator
            k_sum = torch.einsum('ndh -> dh', k)
            den = torch.einsum('ndh, dh -> nh', q, k_sum).unsqueeze(1)

            # linear global attention based on kernel trick
            if self.beta < 0:
                beta = F.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (num/den).reshape(seq_len, -1)
            x = self.lns[i](x) * (h+beta)
            x = F.relu(self.lin_out(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

class SA_MVFormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3, global_layers=2,
            in_dropout=0.15, dropout=0.5, global_dropout=0.5, heads=1, beta=-1, pre_ln=False,
            fusion_method="weighted", fusion_gnn_type="gcn", fusion_gnn_heads=1,
            fusion_graph_weight=0.8, fusion_dropout=None, fusion_feature_source="proj",
            fusion_gnn_layer=2, fusion_gcn_residual=True, fusion_norm_before: bool = False,
            # view fusion controls (original + KNN)
            view_fusion_method: str = 'weighted',
            view_weights: list = None, view_knn_k: int = 10,
            view_knn_metric: str = 'cosine',
            knn_no_dropout: bool = False,
            view_gate_init: float = 0.0):
        super(SA_MVFormer, self).__init__()

        self._global = False        
        self.in_drop = in_dropout    
        self.dropout = dropout       
        self.pre_ln = pre_ln      
        self.fusion_method = fusion_method
        self.fusion_gnn_type = fusion_gnn_type
        self.fusion_gnn_heads = fusion_gnn_heads
        self.fusion_graph_weight = fusion_graph_weight
        self.fusion_dropout = dropout if fusion_dropout is None else fusion_dropout
        self.fusion_feature_source = fusion_feature_source
        self.fusion_gnn_layer = fusion_gnn_layer
        self.fusion_gcn_residual = fusion_gcn_residual
        self.fusion_norm_before = fusion_norm_before
        self.view_fusion_method = view_fusion_method
        self.view_weights = view_weights if view_weights is not None else [0.5, 0.5]
        self.view_knn_k = view_knn_k
        self.view_knn_metric = view_knn_metric
        self.knn_no_dropout = knn_no_dropout
        self.view_gate_init = view_gate_init
        
        self.cached_knn_index = None
        self.knn_cache_key = None 

        ## Two initialization strategies on beta
        self.beta = beta
        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(local_layers,heads*hidden_channels))
        else:
            self.betas = torch.nn.Parameter(torch.ones(local_layers,heads*hidden_channels)*self.beta)


        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()

        for _ in range(local_layers): 
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.local_convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads,
                concat=True, add_self_loops=False, bias=False))
            self.lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(heads*hidden_channels))


        self.lin_in = torch.nn.Linear(in_channels, heads*hidden_channels)
        self.ln = torch.nn.LayerNorm(heads*hidden_channels)
        self.global_attn = GlobalAttn(hidden_channels, heads, global_layers, beta, global_dropout)
        if self.view_fusion_method == 'residual':
            self.knn_gate = torch.nn.Parameter(torch.tensor([self.view_gate_init]))
        self.pred_local = torch.nn.Linear(heads*hidden_channels, out_channels)
        self.pred_global = torch.nn.Linear(heads*hidden_channels, out_channels)
        self.fusion_block = fusionblock(
            in_dim=heads*hidden_channels,
            hidden_dim=heads*hidden_channels,
            out_dim=heads*hidden_channels,
            gnn_type=self.fusion_gnn_type,
            heads=self.fusion_gnn_heads,
            dropout=self.fusion_dropout,
            fusion=self.fusion_method,
            graph_weight=self.fusion_graph_weight,
            gnn_layer=self.fusion_gnn_layer,
            gcn_residual=self.fusion_gcn_residual,
            fusion_norm_before=self.fusion_norm_before,
        )
        self.pred_fused = torch.nn.Linear(heads*hidden_channels, out_channels)
        # Projection layer for concat fusion method to handle 2x dimension
        if self.view_fusion_method == 'concat':
            self.concat_proj = torch.nn.Linear(2*heads*hidden_channels, heads*hidden_channels)
        else:
            self.concat_proj = None

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.global_attn.reset_parameters()
        if self.view_fusion_method == 'residual':
            torch.nn.init.constant_(self.knn_gate, self.view_gate_init)
        self.pred_local.reset_parameters()
        self.pred_global.reset_parameters()
        for m in self.fusion_block.modules():
            if hasattr(m, 'reset_parameters'):
                try:
                    m.reset_parameters()
                except Exception:
                    pass
        self.pred_fused.reset_parameters()
        if self.concat_proj is not None:
            self.concat_proj.reset_parameters()
        
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)

    def encode(self, x, edge_index, adj=None, no_dropout=False):
        x = F.dropout(x, p=self.in_drop if not no_dropout else 0.0, training=self.training)
        x = self.lin_in(x)
        x = F.dropout(x, p=self.dropout if not no_dropout else 0.0, training=self.training)

        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            h = self.h_lins[i](x)
            h = F.relu(h)
            x = local_conv(x, edge_index) + self.lins[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout if not no_dropout else 0.0, training=self.training)
            if self.beta < 0:
                beta = F.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (1 - beta) * self.lns[i](h * x) + beta * x
            x_local = x_local + x

        if self._global:
            attn_out = self.global_attn(self.ln(x_local))
        else:
            attn_out = x_local
        return attn_out

    def forward(self, x, edge_index, adj=None):
        num_nodes = x.size(0)
        
        x = F.dropout(x, p=self.in_drop, training=self.training)
        x_raw = x
        x = self.lin_in(x)
        x_proj = x
        
        
        
        x = F.dropout(x, p=self.dropout, training=self.training)

        # equivariant local attention
        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            h = self.h_lins[i](x)
            h = F.relu(h)
            x = local_conv(x, edge_index) + self.lins[i](x)
            x = F.relu(x) 
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.beta < 0:
                beta = F.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (1-beta)*self.lns[i](h*x) + beta*x
            x_local = x_local + x

        # equivariant global attention
        if self._global:
            attn_out = self.global_attn(self.ln(x_local))
        else:
            attn_out = x_local

        
        views = [attn_out] 
        view_names = ['original']
        
        # Build KNN view
        cache_key = f"{x_raw.shape[0]}_{self.view_knn_k}_{self.view_knn_metric}"
        
        if self.cached_knn_index is None or self.knn_cache_key != cache_key:
            from knn_graph import build_knn_graph
            with torch.no_grad():
                ei_knn, _ = build_knn_graph(x_raw.cpu(),
                                         k=self.view_knn_k,
                                         metric=self.view_knn_metric,
                                         normalize=True,
                                         device='cpu')
                self.cached_knn_index = ei_knn.to(attn_out.device)
                self.knn_cache_key = cache_key
                if not self.training:
                    print(f"The KNN graph has been constructed and cached (number of nodes N={x_raw.shape[0]}, K={self.view_knn_k}, metric={self.view_knn_metric})")
        
        knn_edge_index = self.cached_knn_index
        attn_out_knn = self.encode(x_raw, knn_edge_index, adj=None, no_dropout=self.knn_no_dropout)
        views.append(attn_out_knn)
        view_names.append('knn')
        
        # Fuse two views
        if len(views) == 2:
            if self.view_fusion_method == 'weighted':
                weights = torch.tensor(self.view_weights, device=attn_out.device)
                weights = weights / weights.sum()
                attn_out = sum(w * v for w, v in zip(weights, views))
            elif self.view_fusion_method == 'residual':
                attn_out = views[0]
                attn_out_knn_for_residual = views[1]
            else:  # concat
                attn_out_concat = torch.cat(views, dim=-1)
                # Project concat features back to original dimension for fusion_block
                attn_out = self.concat_proj(attn_out_concat)

        # Always use fusion block
        if self.fusion_feature_source == 'proj':
            fusion_x = x_proj
        elif self.fusion_feature_source == 'local':
            fusion_x = x_local
        else:
            raise ValueError(f"Unsupported fusion_feature_source: {self.fusion_feature_source}")

        graph_struct = adj if (self.fusion_gnn_type == 'dense' and adj is not None) else edge_index
        fused = self.fusion_block(fusion_x, graph_struct, attn_out)
        out = self.pred_fused(fused)

        # Handle residual fusion for two views
        if (self.view_fusion_method == 'residual' and 
            len(views) == 2 and 
            'attn_out_knn_for_residual' in locals()):
            
            if self._global:
                out_knn = self.pred_global(attn_out_knn_for_residual)
            else:
                out_knn = self.pred_local(attn_out_knn_for_residual)
            
            gate_weight = self.knn_gate
            out = out + gate_weight * (out_knn - out)

        return out
