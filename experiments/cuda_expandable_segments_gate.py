"""PB-only pinned-container qualification of the CUDA allocator option."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
assert os.environ.get('PYTORCH_ALLOC_CONF')=='expandable_segments:True'
torch.set_num_threads(1)
tensor=torch.arange(2**21,dtype=torch.int32,device='cuda')
returned=tensor.cpu()
assert torch.equal(returned,torch.arange(2**21,dtype=torch.int32))
segments=torch.cuda.memory_snapshot()
assert any(s.get('is_expandable') is True for s in segments), 'allocator option unsupported or inactive'
result={'schema':'prismaquant.cuda_expandable_segments_gate.v1','torch':torch.__version__,
        'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),
        'backend':torch.cuda.get_allocator_backend(),'allocator_conf':os.environ['PYTORCH_ALLOC_CONF'],
        'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'verified_values':tensor.numel(),'value_sha256':hashlib.sha256(returned.numpy().tobytes()).hexdigest(),
        'expandable_segments':sum(s.get('is_expandable') is True for s in segments),
        'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}
del tensor
torch.cuda.empty_cache()
result['allocated_after_release']=torch.cuda.memory_allocated()
result['reserved_after_release']=torch.cuda.memory_reserved()
assert result['allocated_after_release']==0
args.out.parent.mkdir(parents=True,exist_ok=True)
with args.out.open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result),flush=True)
