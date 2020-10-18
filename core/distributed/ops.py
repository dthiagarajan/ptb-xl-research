import torch
import torch.distributed as dist

def all_gather_op(single_node_output):
    all_node_outputs = [
        torch.zeros_like(single_node_output).cuda() if torch.cuda.is_available()
        else torch.zeros_like(single_node_output)
        for rank in range(dist.get_world_size())
    ]
    if torch.cuda.is_available():
        single_node_output = single_node_output.cuda()
    dist.all_gather_multigpu(all_node_outputs, single_node_output)
    dist.barrier()
    return torch.cat(all_node_outputs)
