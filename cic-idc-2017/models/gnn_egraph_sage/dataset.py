import torch
from torch.utils.data import Dataset, DataLoader

class EdgeBatchDataset(Dataset):
    def __init__(self, graph, edge_batches, batch_size=16, shuffle=True):
        self.graph = graph
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.edge_batches = edge_batches
        self.number_of_edge_batches = len(self.edge_batches)
        
        if self.shuffle:
            self.edge_batches = self.edge_batches[torch.randperm(self.number_of_edge_batches)]
        
        self.subgraphs = self._precompute_sub_graphs()

    def __len__(self):
        return (self.number_of_edge_batches + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.number_of_edge_batches)
        
        return self.edge_batches[start:end], self.subgraphs[idx]
    
    def _precompute_sub_graphs(self):
        batch_sub_graphs = {}
        for idx, i in enumerate(range(0, len(self.edge_batches), self.batch_size)):
            # Get batch of edges
            edge_batch = torch.cat(self.edge_batches[i:i+self.batch_size])
            # Get source and destination nodes for these edges
            src_nodes, dst_nodes = self.graph.find_edges(edge_batch)
            
            # Combine all nodes involved in these edges
            nodes = torch.cat([src_nodes, dst_nodes]).unique()
            
            # Create subgraph
            subgraph = self.graph.subgraph(nodes)
            
            # Store edge mapping for later use
            batch_sub_graphs[idx] = subgraph
        return batch_sub_graphs


def create_dataloader(g, batched_edges, batch_size, shuffle):
    dataset = EdgeBatchDataset(g, batched_edges, batch_size=batch_size, shuffle=shuffle)
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # We're handling batching in the dataset
        shuffle=False,
    )
    return dataloader