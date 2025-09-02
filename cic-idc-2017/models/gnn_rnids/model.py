import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class GNN_NIDS(nn.Module):
    def __init__(self, in_feats_host, in_feats_flow, hidden_size, num_classes, num_iterations):
        super(GNN_NIDS, self).__init__()
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations
        
        self.host_fc = nn.Linear(in_feats_host, hidden_size)
        self.flow_fc = nn.Linear(in_feats_flow, hidden_size)
        
        self.msg_from_host = nn.Linear(2 * hidden_size, hidden_size)  # host to flow
        self.msg_to_host = nn.Linear(2 * hidden_size, hidden_size)    # flow to host
        
        self.host_gru = nn.GRUCell(hidden_size, hidden_size)
        self.flow_gru = nn.GRUCell(hidden_size, hidden_size)
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
        
    def message_func_from_host(self, edges):
        msg = F.relu(F.dropout(self.msg_from_host(torch.cat([edges.src['h'], edges.dst['h']], dim=1)), 0.5))
        return {'msg': msg}

    def message_func_to_host(self, edges):
        msg = F.relu(F.dropout(self.msg_to_host(torch.cat([edges.src['h'], edges.dst['h']], dim=1)), 0.5))
        return {'msg': msg}

    def reduce_func(self, nodes):
        return {'agg': torch.sum(nodes.mailbox['msg'], dim=1)}
    
    def forward(self, g, host_features, flow_features):
        host_h = torch.cat(
            [host_features, torch.ones(host_features.shape[0], self.hidden_size - host_features.shape[1], device=host_features.device)],
            dim=1
        )
        flow_h = torch.cat(
            [flow_features, torch.zeros(flow_features.shape[0], self.hidden_size - flow_features.shape[1], device=flow_features.device)],
            dim=1
        )
    
        g.nodes['host'].data['h'] = host_h
        g.nodes['flow'].data['h'] = flow_h
    
        for _ in range(self.num_iterations):
            for ntype in ['host', 'flow']:
                if 'agg' in g.nodes[ntype].data:
                    del g.nodes[ntype].data['agg']
    
            g.multi_update_all(
                {
                    ('host', 'from_', 'flow'): (self.message_func_from_host, self.reduce_func),
                    ('flow', 'to_', 'host'): (self.message_func_to_host, self.reduce_func)
                },
                cross_reducer='mean'
            )
    
            for ntype in ['host', 'flow']:
                if 'agg' not in g.nodes[ntype].data:
                    h_shape = g.nodes[ntype].data['h'].shape
                    g.nodes[ntype].data['agg'] = torch.zeros(
                        h_shape[0], self.hidden_size, device=g.device
                    )
    
            host_h_new = self.host_gru(
                g.nodes['host'].data['agg'],
                g.nodes['host'].data['h']
            )
            flow_h_new = self.flow_gru(
                g.nodes['flow'].data['agg'],
                g.nodes['flow'].data['h']
            )
    
            g.nodes['host'].data['h'] = host_h_new
            g.nodes['flow'].data['h'] = flow_h_new
    
        flow_h = g.nodes['flow'].data['h']
        return self.readout(flow_h)