import torch
from torch.utils.data import Dataset, DataLoader
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FlowNodeDataset(Dataset):
    def __init__(self, graph, flow_node_batches, batch_size=16, shuffle=True):
        self.graph = graph
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flow_node_batches = flow_node_batches
        self.number_of_node_batches = len(self.flow_node_batches)
        
        if self.shuffle:
            self.flow_node_batches = self.flow_node_batches[torch.randperm(self.number_of_node_batches)]
        
        # Pre-compute connected nodes
        #self.connected_nodes = self._precompute_connected_nodes()
        self.subgraphs = self._precompute_sub_graphs()

    def __len__(self):
        return (self.number_of_node_batches + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.number_of_node_batches)
        #batch_flow_nodes = torch.cat(self.flow_node_batches[start:end])
        #return batch_flow_nodes
        return self.flow_node_batches[start:end],self.subgraphs[idx]
    
    def _precompute_connected_nodes(self):
        connected_nodes = {}
        for batch in self.flow_node_batches:
            _, to_host_nodes = self.graph.out_edges(batch, etype=('flow', 'to_', 'host'))
            from_host_nodes, _ = self.graph.in_edges(batch, etype=('host', 'from_', 'flow'))
            host_nodes = torch.cat([to_host_nodes, from_host_nodes]).unique()
            connected_nodes[batch] = host_nodes
        return connected_nodes

    
    def _precompute_sub_graphs(self):
        batch_sub_graphs = {}
        for idx,i in enumerate(range(0,len(self.flow_node_batches),self.batch_size)):
            flow_node_batches_i = torch.cat(self.flow_node_batches[i:i+self.batch_size])
            _, to_host_nodes = self.graph.out_edges(flow_node_batches_i, etype=('flow', 'to_', 'host'))
            from_host_nodes, _ = self.graph.in_edges(flow_node_batches_i, etype=('host', 'from_', 'flow'))
            host_nodes = torch.cat([to_host_nodes, from_host_nodes]).unique()
            nodes_dict = {'flow': flow_node_batches_i, 'host': host_nodes}
            subgraph = self.graph.subgraph(nodes_dict)
            batch_sub_graphs[idx] = subgraph
        return batch_sub_graphs
    


def create_datalaoder(g, batched_flow_nodes, batch_size, shuffle, num_workers=4):
    dataset = FlowNodeDataset(g, batched_flow_nodes, batch_size=batch_size, shuffle=shuffle)
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
    )
    return dataloader