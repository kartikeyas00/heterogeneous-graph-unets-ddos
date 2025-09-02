import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import HeteroGraphConv, GraphConv, SAGEConv, GATConv
import numpy as np
import random
import gc
from tqdm import tqdm

##########################################
# Heterogeneous Attention-based Pooling
##########################################
class HeteroAttentionPooling(nn.Module):
    def __init__(self, in_feats_dict, ratio=0.5, attention_heads=4, attn_dropout=0.1):
        """
        Heterogeneous Attention-based Pooling layer.

        Parameters:
        - in_feats_dict (dict): Mapping from node types to feature sizes.
        - ratio (float): Fraction of nodes to keep.
        - attention_heads (int): Number of attention heads.
        - attn_dropout (float): Dropout rate applied to attention scores.
        """
        super(HeteroAttentionPooling, self).__init__()
        self.ratio = ratio
        self.attention_heads = attention_heads
        
        # Multi-head attention scoring for each node type.
        self.score_layers = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(in_feats, attention_heads * in_feats),
                nn.LeakyReLU(0.2),
                nn.Dropout(attn_dropout),
                nn.Linear(attention_heads * in_feats, attention_heads)
            ) for ntype, in_feats in in_feats_dict.items()
        })
        
        # Optionally, you could also add edge scoring if needed.
        self.edge_score = nn.ModuleDict({
            ntype: nn.Linear(in_feats, 1)
            for ntype, in_feats in in_feats_dict.items()
        })

    def forward(self, graph, feat_dict):
        with graph.local_scope():
            topk_indices = {}
            new_feat_dict = {}
            scores_dict = {}
            
            # Compute node-level attention scores for each node type.
            for ntype in graph.ntypes:
                if graph.num_nodes(ntype) == 0:
                    continue
                feat = feat_dict[ntype]
                # Multi-head scoring produces a score per head.
                attn_scores = self.score_layers[ntype](feat)  # [N, heads]
                # Average scores over heads for stability.
                attn_scores = attn_scores.mean(dim=1)
                
                # Add a tiny random noise for tie-breaking during training.
                if self.training:
                    attn_scores = attn_scores + torch.randn_like(attn_scores) * 1e-5

                # Use sigmoid (or another bounded activation) to constrain scores between 0 and 1.
                scores = torch.sigmoid(attn_scores)
                scores_dict[ntype] = scores

                # Determine the number of nodes to retain.
                k = max(1, int(self.ratio * graph.num_nodes(ntype)))
                _, idx = torch.topk(scores, k)
                topk_indices[ntype] = idx
            
            # Create a subgraph using the selected nodes.
            subgraph = dgl.node_subgraph(graph, topk_indices)
            
            # Update features by scaling with the attention score (with a residual term).
            for ntype in subgraph.ntypes:
                if ntype not in topk_indices or topk_indices[ntype].numel() == 0:
                    new_feat_dict[ntype] = torch.zeros((0, feat_dict[ntype].shape[1]), 
                                                       device=feat_dict[ntype].device)
                    continue

                idx = topk_indices[ntype]
                node_feat = feat_dict[ntype][idx]
                node_scores = scores_dict[ntype][idx].unsqueeze(-1)
                scaled_feat = node_feat * (1.0 + node_scores)  # Residual scaling
                new_feat_dict[ntype] = scaled_feat

            return subgraph, new_feat_dict, topk_indices



class ResidualHeteroGATConv(nn.Module):
    """Residual Heterogeneous Graph Convolution layer using Graph Attention (GATConv) with skip connections."""
    def __init__(self, in_feats_dict, hidden_feats, rel_names, canonical_etypes, dropout=0.1, num_heads=4):
        super(ResidualHeteroGATConv, self).__init__()
        self.num_heads = num_heads
        
        # Build a heterograph convolution where each relation uses GATConv.
        self.conv = HeteroGraphConv({
            rel: GATConv(
                in_feats_dict[stype],
                hidden_feats,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=False  # We'll handle residuals separately.
            )
            for rel, (stype, _, _) in zip(rel_names, canonical_etypes)
        }, aggregate='sum')
        
        # Projection layers for residual connections when dimensions don't match.
        self.projections = nn.ModuleDict({
            ntype: nn.Linear(in_feats, hidden_feats)
            for ntype, in_feats in in_feats_dict.items()
            if in_feats != hidden_feats
        })
        
        # Layer normalization for each node type.
        self.norm = nn.ModuleDict({
            ntype: nn.LayerNorm(hidden_feats)
            for ntype in in_feats_dict.keys()
        })
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, graph, feat_dict):
        # Compute multi-head attention outputs.
        new_feat_dict = self.conv(graph, feat_dict)
        # GATConv returns a tensor of shape [N, num_heads, hidden_feats]. Average over heads.
        for ntype in new_feat_dict:
            new_feat_dict[ntype] = new_feat_dict[ntype].mean(dim=1)
        
        # Apply residual connections.
        for ntype, feat in feat_dict.items():
            if ntype in new_feat_dict:
                if feat.shape[1] != new_feat_dict[ntype].shape[1]:
                    residual = self.projections[ntype](feat) if ntype in self.projections else feat
                else:
                    residual = feat
                # Combine and apply normalization, activation, and dropout.
                combined = new_feat_dict[ntype] + residual
                new_feat_dict[ntype] = self.dropout(F.gelu(self.norm[ntype](combined)))
        return new_feat_dict

##########################################
# Residual Heterogeneous Graph Convolution
##########################################
class ResidualHeteroConv(nn.Module):
    """Residual Heterogeneous Graph Convolution with skip connections."""
    def __init__(self, in_feats_dict, hidden_feats, rel_names, canonical_etypes, dropout=0.1):
        super(ResidualHeteroConv, self).__init__()
        # Build hetero convolution for each relation type using SAGEConv.
        self.conv = HeteroGraphConv({
            rel: SAGEConv(
                in_feats_dict[stype],
                hidden_feats,
                aggregator_type='mean',
                feat_drop=dropout
            )
            for rel, (stype, _, _) in zip(rel_names, canonical_etypes)
        }, aggregate='sum')
        
        # If input and output dimensions differ, create projection layers.
        self.projections = nn.ModuleDict({
            ntype: nn.Linear(in_feats, hidden_feats)
            for ntype, in_feats in in_feats_dict.items()
            if in_feats != hidden_feats
        })
        
        self.norm = nn.ModuleDict({
            ntype: nn.LayerNorm(hidden_feats)
            for ntype in in_feats_dict.keys()
        })
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, feat_dict):
        new_feat_dict = self.conv(graph, feat_dict)
        # Apply residual connections.
        for ntype, feat in feat_dict.items():
            if ntype in new_feat_dict:
                if feat.shape[1] != new_feat_dict[ntype].shape[1]:
                    residual = self.projections[ntype](feat) if ntype in self.projections else feat
                else:
                    residual = feat
                # Combine, normalize, activate and dropout.
                combined = new_feat_dict[ntype] + residual
                new_feat_dict[ntype] = self.dropout(F.gelu(self.norm[ntype](combined)))
        return new_feat_dict


##########################################
# Heterogeneous Graph U-Net
##########################################
class HeteroGraphUNet(nn.Module):
    def __init__(self, in_feats_dict, hidden_feats, out_feats_dict, rel_names, canonical_etypes,
                 depth=3, pool_ratio=0.5, dropout=0.2, use_gat=True):
        """
        A U-Net style architecture for heterogeneous graphs.

        Parameters:
        - in_feats_dict (dict): Mapping from node types to input feature sizes.
        - hidden_feats (int): Hidden feature dimension.
        - out_feats_dict (dict): Mapping from node types (e.g., 'flow') to output sizes.
        - rel_names (list): List of relation names.
        - canonical_etypes (list): List of canonical edge types (tuples: (src, rel, dst)).
        - depth (int): Number of down/up-sampling layers.
        - pool_ratio (float): Base pooling ratio (may decay per layer).
        - dropout (float): Dropout rate.
        - use_gat (bool): Whether to use GAT in the bottleneck.
        """
        super(HeteroGraphUNet, self).__init__()
        self.depth = depth
        self.pool_ratio = pool_ratio
        self.hidden_feats = hidden_feats
        self.dropout = dropout
        self.use_gat = use_gat

        # Initial feature embedding for each node type.
        self.node_embeddings = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(in_feats, hidden_feats),
                nn.LayerNorm(hidden_feats),
                nn.GELU()
            ) for ntype, in_feats in in_feats_dict.items()
        })

        # Build encoder (downsampling) path.
        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        curr_in_feats_dict = {ntype: hidden_feats for ntype in in_feats_dict.keys()}
        for i in range(depth):
            self.down_convs.append(
                #ResidualHeteroConv(curr_in_feats_dict, hidden_feats, rel_names, canonical_etypes, dropout=dropout)
                ResidualHeteroGATConv(curr_in_feats_dict, hidden_feats, rel_names, canonical_etypes, dropout=dropout)

            )
            # Decaying pool ratio for deeper layers.
            pool = HeteroAttentionPooling(
                {ntype: hidden_feats for ntype in in_feats_dict.keys()},
                ratio=pool_ratio * (0.8 ** i)
            )
            self.pools.append(pool)

        # Bottleneck: use GAT for improved message passing if desired.
        if use_gat:
            self.bottom_conv = HeteroGraphConv({
                rel: GATConv(
                    hidden_feats,
                    hidden_feats,
                    num_heads=4,
                    feat_drop=dropout,
                    attn_drop=dropout
                ) for rel in rel_names
            }, aggregate='sum')
        else:
            self.bottom_conv = ResidualHeteroConv(
                {ntype: hidden_feats for ntype in in_feats_dict.keys()},
                hidden_feats,
                rel_names,
                canonical_etypes,
                dropout=dropout
            )

        # Build decoder (upsampling) path.
        self.up_convs = nn.ModuleList()
        for i in range(depth):
            # For the first up conv after bottleneck, handle potential multi-head outputs.
            if use_gat and i == 0:
                conv = HeteroGraphConv({
                    rel: GraphConv(hidden_feats * 2, hidden_feats)
                    for rel in rel_names
                }, aggregate='sum')
            else:
                conv = ResidualHeteroConv(
                    {ntype: hidden_feats * 2 for ntype in in_feats_dict.keys()},
                    hidden_feats,
                    rel_names,
                    canonical_etypes,
                    dropout=dropout
                )
            self.up_convs.append(conv)

        # Output classifier for binary classification on 'flow' nodes.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats * 2),
            nn.LayerNorm(hidden_feats * 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_feats * 2, hidden_feats),
            nn.LayerNorm(hidden_feats),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_feats, out_feats_dict['flow'])
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _handle_empty_subgraphs(self, subgraph, feat_dict):
        # Ensure each node type exists in the feature dict.
        for ntype in subgraph.ntypes:
            if subgraph.num_nodes(ntype) == 0 and ntype not in feat_dict:
                feat_dict[ntype] = torch.zeros((0, self.hidden_feats), device=next(self.parameters()).device)
        return feat_dict

    def forward(self, graph, feat_dict):
        # Initial node embedding.
        feat_dict = {ntype: self.node_embeddings[ntype](feat) for ntype, feat in feat_dict.items()}

        encoder_graphs = []
        encoder_feats = []
        pooling_indices = []  # Store mapping for unpooling.

        curr_graph = graph
        # Encoder (downsampling) path.
        for i in range(self.depth):
            feat_dict = self.down_convs[i](curr_graph, feat_dict)
            encoder_graphs.append(curr_graph)
            # Save a copy of features for skip connection.
            encoder_feats.append({k: v.clone() for k, v in feat_dict.items()})
            # Apply pooling.
            curr_graph, feat_dict, idx = self.pools[i](curr_graph, feat_dict)
            feat_dict = self._handle_empty_subgraphs(curr_graph, feat_dict)
            pooling_indices.append(idx)

        # Bottleneck processing.
        if self.use_gat:
            gat_outputs = self.bottom_conv(curr_graph, feat_dict)
            # Average multi-head outputs if necessary.
            feat_dict = {ntype: (feat.mean(dim=1) if feat.dim() > 2 else feat)
                         for ntype, feat in gat_outputs.items()}
        else:
            feat_dict = self.bottom_conv(curr_graph, feat_dict)

        # Decoder (upsampling) path.
        for i in range(self.depth):
            idx = self.depth - i - 1  # Reverse order for skip connections.
            encoder_graph = encoder_graphs[idx]
            encoder_feat = encoder_feats[idx]
            # Upsample: create an empty tensor for each node type and place the pooled features back
            up_feat_dict = {}
            for ntype in encoder_graph.ntypes:
                # Start with the original encoder features.
                base_feat = encoder_feat[ntype]
                # If pooling was applied to this node type, restore the features from the pooled subgraph.
                if ntype in pooling_indices[idx]:
                    pool_idx = pooling_indices[idx][ntype]
                    # Create a tensor to hold the upsampled features (default zeros).
                    up_feat = torch.zeros_like(base_feat)
                    # Only update positions corresponding to the pooled nodes.
                    if feat_dict.get(ntype) is not None and pool_idx.numel() > 0 and feat_dict[ntype].shape[0] > 0:
                        up_feat[pool_idx] = feat_dict[ntype]
                    # Concatenate along feature dimension for skip connection.
                    up_feat = torch.cat([base_feat, up_feat], dim=1)
                else:
                    up_feat = base_feat
                up_feat_dict[ntype] = up_feat

            feat_dict = self.up_convs[i](encoder_graph, up_feat_dict)
            curr_graph = encoder_graph

        # Final classification on 'flow' nodes.
        logits = self.classifier(feat_dict['flow'])
        return logits
