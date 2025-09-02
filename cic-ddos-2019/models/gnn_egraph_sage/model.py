import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class EGraphSAGELayer(nn.Module):
    """
    E-GraphSAGE layer implementation that matches the original paper's implementation.
    """
    def __init__(self, node_in_dim, edge_dim, node_out_dim, activation):
        super(EGraphSAGELayer, self).__init__()
        # Message transformation: combines node and edge features
        self.W_msg = nn.Linear(node_in_dim + edge_dim, node_out_dim)
        # Node transformation: combines node features with aggregated messages
        self.W_apply = nn.Linear(node_in_dim + node_out_dim, node_out_dim)
        self.activation = activation

    def message_func(self, edges):
        """
        Message function that combines source node features with edge features.
        Handles batched inputs (3D tensors).
        """
        return {'m': self.W_msg(torch.cat([edges.src['h'], edges.data['feat']], -1))}

    def forward(self, g, nfeats, efeats):
        """
        Forward pass handling batched inputs.
        
        Args:
            g: DGL graph
            nfeats: Node features [batch_size, num_nodes, node_feat_dim]
            efeats: Edge features [batch_size, num_edges, edge_feat_dim]
        """
        with g.local_scope():
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            
            # Aggregate messages from neighbors
            g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
            
            # Combine with original node features
            h = torch.cat([g.ndata['h'], g.ndata['h_neigh']], -1)
            h = self.W_apply(h)
            
            return self.activation(h)

class EdgePredictor(nn.Module):
    """
    Edge prediction module that handles batched inputs.
    """
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.W = nn.Linear(in_dim * 2, num_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], -1))
        return {'score': score}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

class EGraphSAGE(nn.Module):
    """
    Full E-GraphSAGE model with proper batch handling.
    """
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_classes,num_layers=2, dropout=0.2):
        super(EGraphSAGE, self).__init__()
        
        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(EGraphSAGELayer(
            node_in_dim, edge_in_dim, hidden_dim, F.relu
        ))
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(EGraphSAGELayer(
                hidden_dim, edge_in_dim, hidden_dim, F.relu
            ))
        
        # Edge predictor
        self.predictor = EdgePredictor(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g):
        """
        Forward pass with batched inputs.
        
        Args:
            g: DGL graph
            nfeats: Initial node features [batch_size, num_nodes, node_feat_dim]
            efeats: Edge features [batch_size, num_edges, edge_feat_dim]
        """
        h = g.ndata['feat']
        
        # Message passing layers
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, g.edata['feat'])
        
        # Predict edge scores
        return self.predictor(g, h)

    def get_edge_embeddings(self, g):
        """
        Get edge embeddings for visualization/analysis.
        Handles batched inputs.
        """
        h = g.ndata['feat']
        
        # Get final node representations
        for layer in self.layers:
            h = layer(g, h, g.edata['feat'])
            
        # Compute edge embeddings
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(lambda edges: {
                'embedding': torch.cat([edges.src['h'], edges.dst['h']], -1)
            })
            return g.edata['embedding']