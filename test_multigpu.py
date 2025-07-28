import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if __name__ == "__main__":
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    print(f"Process {local_rank} using GPU {torch.cuda.current_device()}")
    print(f"Available memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
    
    # Properly cleanup
    dist.destroy_process_group()
