import torch
from torch.utils.data import Dataset, DataLoader
import random

class FlowNodeDataset(Dataset):
    """
    Dataset for batch processing of flow nodes in a heterogeneous graph.
    
    This dataset creates subgraphs centered around batches of flow nodes,
    optionally including previously seen connected flows.
    """
    def __init__(self, graph, flow_node_batches, batch_size=16, shuffle=True, max_connected_flows=30):
        """
        Initialize the FlowNodeDataset.
        
        Args:
            graph: The DGL heterogeneous graph
            flow_node_batches: List of tensors, each containing flow node IDs for a batch
            batch_size: Number of batches to process at once
            shuffle: Whether to shuffle the batches
            max_connected_flows: Maximum number of previously seen flow nodes to include per batch
        """
        self.graph = graph
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flow_node_batches = flow_node_batches
        self.max_connected_flows = max_connected_flows
        self.number_of_node_batches = len(self.flow_node_batches)
        
        # Initialize a set to track all previously seen flow nodes
        self.previously_seen_flows = set()
        
        if self.shuffle:
            # Shuffle the batches, not the nodes within batches
            indices = torch.randperm(self.number_of_node_batches)
            self.flow_node_batches = [self.flow_node_batches[i] for i in indices]
        
        self.subgraphs = self._precompute_sub_graphs()

    def __len__(self):
        return (self.number_of_node_batches + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.number_of_node_batches)
        
        # Return the batches and corresponding precomputed subgraph
        return self.flow_node_batches[start:end], self.subgraphs[idx]
    
    def reset(self):
        """Reset the dataset's internal state, particularly for shuffling and seen nodes."""
        self.previously_seen_flows = set()
        if self.shuffle:
            indices = torch.randperm(self.number_of_node_batches)
            self.flow_node_batches = [self.flow_node_batches[i] for i in indices]
            # Recompute subgraphs with the new batch order
            self.subgraphs = self._precompute_sub_graphs()
    
    def _precompute_sub_graphs(self):
        """
        Precompute subgraphs for each batch of flow nodes.
        Includes host nodes connected to flows and a limited number of
        previously seen flow nodes that are connected to the current batch.
        """
        batch_sub_graphs = {}
        
        for idx, i in enumerate(range(0, len(self.flow_node_batches), self.batch_size)):
            # Get all flow nodes for this batch
            end_idx = min(i + self.batch_size, len(self.flow_node_batches))
            batch_indices = list(range(i, end_idx))
            
            # Concatenate flow nodes from all batches in this group
            flow_nodes = torch.cat([self.flow_node_batches[j] for j in batch_indices])
            
            # Get connected host nodes
            _, to_host_nodes = self.graph.out_edges(flow_nodes, etype='to_')
            from_host_nodes, _ = self.graph.in_edges(flow_nodes, etype='from_')
            host_nodes = torch.cat([to_host_nodes, from_host_nodes]).unique()
            
            # Add current flow nodes to previously seen set
            for node in flow_nodes.tolist():
                self.previously_seen_flows.add(node)
            
            # Find connected flow nodes from both shared_ip_port and temporal edges
            connected_flow_candidates = set()
            
            # Check for each edge type that might connect flows
            edge_types_to_check = []
            for src, etype, dst in self.graph.canonical_etypes:
                if src == 'flow' and dst == 'flow':
                    edge_types_to_check.append(etype)
            
            # Collect all connected flow nodes across all flow-to-flow edge types
            for etype in edge_types_to_check:
                try:
                    # Get outgoing connected flows
                    src_flows, dst_flows = self.graph.out_edges(flow_nodes, etype=(src, etype, dst))
                    for node in dst_flows.tolist():
                        connected_flow_candidates.add(node)
                    
                    # Get incoming connected flows
                    _, incoming_flows = self.graph.in_edges(flow_nodes, etype=(src, etype, dst))
                    for node in incoming_flows.tolist():
                        connected_flow_candidates.add(node)
                except:
                    # Skip if edge type doesn't exist or there's another error
                    continue
            
            # Filter to only include previously seen flows (excluding current batch)
            connected_flow_nodes = []
            for node in connected_flow_candidates:
                if node in self.previously_seen_flows and node not in flow_nodes.tolist():
                    connected_flow_nodes.append(node)
            
            # Limit the number of connected flows if needed
            if len(connected_flow_nodes) > self.max_connected_flows:
                connected_flow_nodes = random.sample(connected_flow_nodes, self.max_connected_flows)
            
            # Convert to tensor
            if connected_flow_nodes:
                connected_flow_tensor = torch.tensor(connected_flow_nodes, device=flow_nodes.device)
                all_flow_nodes = torch.cat([flow_nodes, connected_flow_tensor])
            else:
                all_flow_nodes = flow_nodes
            
            # Create node dictionary for subgraph
            nodes_dict = {'flow': all_flow_nodes, 'host': host_nodes}
            
            # Create the subgraph
            subgraph = self.graph.subgraph(nodes_dict)
            batch_sub_graphs[idx] = subgraph
            
        return batch_sub_graphs
    # def _precompute_sub_graphs(self):
    #     batch_sub_graphs = {}
    #     for idx,i in enumerate(range(0,len(self.flow_node_batches),self.batch_size)):
    #         flow_node_batches_i = torch.cat(self.flow_node_batches[i:i+self.batch_size])
    #         _, to_host_nodes = self.graph.out_edges(flow_node_batches_i, etype=('flow', 'to_', 'host'))
    #         from_host_nodes, _ = self.graph.in_edges(flow_node_batches_i, etype=('host', 'from_', 'flow'))
    #         host_nodes = torch.cat([to_host_nodes, from_host_nodes]).unique()
    #         nodes_dict = {'flow': flow_node_batches_i, 'host': host_nodes}
    #         subgraph = self.graph.subgraph(nodes_dict)
    #         batch_sub_graphs[idx] = subgraph
    #     return batch_sub_graphs



def create_datalaoder(g, batched_flow_nodes, batch_size, shuffle, num_workers=0, max_connected_flows=30):
    """
    Create a DataLoader for processing batches of flow nodes.
    
    Args:
        g: The DGL heterogeneous graph
        batched_flow_nodes: List of tensors, each containing flow node IDs for a batch
        batch_size: Number of batches to process at once
        shuffle: Whether to shuffle the batches
        num_workers: Number of worker processes for data loading
        max_connected_flows: Maximum number of previously seen connected flows to include
        
    Returns:
        A DataLoader that yields (batch_indices, subgraph) pairs
    """
    dataset = FlowNodeDataset(
        g, 
        batched_flow_nodes, 
        batch_size=batch_size, 
        shuffle=shuffle,
        max_connected_flows=max_connected_flows
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # We're already batching at the dataset level
        shuffle=False,    # Shuffling is handled in the dataset if needed
    )
    
    return dataloader