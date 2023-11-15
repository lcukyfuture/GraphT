
from WL_gpu import WL
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def compute_kernel_for_batch(batch_data, device, iteration=3):
    sub_kernels = []
    X = batch_data.x[batch_data.subgraph_node_index].argmax(-1).to(device)
    E = batch_data.subgraph_edge_index.to(device)
    B = batch_data.subgraph_indicator.to(device)
    # print(iteration)
    wl = WL(iteration)
    wl.fit((X,E,B))
    kernel_out = wl.transform((X,E,B))
    num_nodes = torch.diff(batch_data.ptr).tolist()
    start_idx = 0
    del X, E, B, batch_data
    torch.cuda.empty_cache() 
    for num in num_nodes:
        sub_kernels.append(kernel_out[start_idx:start_idx+num, start_idx:start_idx+num])
        start_idx += num
    # print(sub_kernels[0])
    return sub_kernels


