import ctypes,time,json,statistics
import torch
lib=ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')
def call(name,*args):
    status=getattr(lib,name)(*args)
    if status: raise RuntimeError(name,status)
torch.cuda.init()
stream=ctypes.c_void_p(); event=ctypes.c_void_p()
call('cudaStreamCreateWithFlags',ctypes.byref(stream),1)
call('cudaEventCreateWithFlags',ctypes.byref(event),3)
src=torch.empty(64*1024*1024,dtype=torch.uint8,pin_memory=True).fill_(123)
dst=torch.empty_like(src,device='cuda')
results={}
for mode in ['stream','event','event','stream']:
    walls=[]; cpus=[]
    for i in range(101):
        wall=time.perf_counter(); cpu=time.thread_time()
        call('cudaMemcpyAsync',ctypes.c_void_p(dst.data_ptr()),ctypes.c_void_p(src.data_ptr()),ctypes.c_size_t(src.numel()),1,stream)
        if mode=='stream': call('cudaStreamSynchronize',stream)
        else:
            call('cudaEventRecord',event,stream)
            call('cudaEventSynchronize',event)
        if i:
            walls.append(time.perf_counter()-wall);cpus.append(time.thread_time()-cpu)
    results.setdefault(mode,[]).append(dict(wall_median=statistics.median(walls),thread_cpu_median=statistics.median(cpus)))
assert torch.equal(dst.cpu(),src)
call('cudaEventDestroy',event);call('cudaStreamDestroy',stream)
print(json.dumps(results,indent=2))
