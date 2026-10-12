"""Generate reproducible end-to-end references with upstream OIDN v2.5.1.
The native Julia port does NOT load this library; it is only the test oracle."""
import argparse, ctypes as C, pathlib
import numpy as np
from safetensors.numpy import save_file

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--library",required=True)
    p.add_argument("--models",type=pathlib.Path,required=True)
    p.add_argument("--output",type=pathlib.Path,required=True)
    p.add_argument("--width",type=int,default=61)
    p.add_argument("--height",type=int,default=47)
    args=p.parse_args()
    lib=C.CDLL(args.library)
    signatures={
      "oidnNewDevice":(C.c_void_p,[C.c_int]),
      "oidnCommitDevice":(None,[C.c_void_p]),
      "oidnNewFilter":(C.c_void_p,[C.c_void_p,C.c_char_p]),
      "oidnSetFilterInt":(None,[C.c_void_p,C.c_char_p,C.c_int]),
      "oidnSetFilterBool":(None,[C.c_void_p,C.c_char_p,C.c_bool]),
      "oidnSetSharedFilterData":(None,[C.c_void_p,C.c_char_p,C.c_void_p,C.c_size_t]),
      "oidnSetSharedFilterImage":(None,[C.c_void_p,C.c_char_p,C.c_void_p,C.c_int,
        C.c_size_t,C.c_size_t,C.c_size_t,C.c_size_t,C.c_size_t]),
      "oidnCommitFilter":(None,[C.c_void_p]),
      "oidnExecuteFilter":(None,[C.c_void_p]),
      "oidnReleaseFilter":(None,[C.c_void_p]),
      "oidnReleaseDevice":(None,[C.c_void_p]),
      "oidnGetDeviceError":(C.c_int,[C.c_void_p,C.POINTER(C.c_char_p)])}
    for name,(restype,argtypes) in signatures.items():
        fn=getattr(lib,name);fn.restype=restype;fn.argtypes=argtypes
    dev=lib.oidnNewDevice(1);lib.oidnCommitDevice(dev)
    def check():
        msg=C.c_char_p();err=lib.oidnGetDeviceError(dev,C.byref(msg))
        if err:raise RuntimeError((err,msg.value))
    def run(features,kind):
        out=np.zeros_like(next(iter(features.values())))
        f=lib.oidnNewFilter(dev,b"RT")
        blob=(args.models/(kind+".tza")).read_bytes();weights=C.create_string_buffer(blob)
        lib.oidnSetSharedFilterData(f,b"weights",weights,len(blob))
        lib.oidnSetFilterInt(f,b"quality",6)
        lib.oidnSetFilterBool(f,b"hdr",kind.startswith("rt_hdr"))
        lib.oidnSetFilterBool(f,b"cleanAux",True)
        for name,image in {**features,"output":out}.items():
            h,w,c=image.shape
            lib.oidnSetSharedFilterImage(f,name.encode(),image.ctypes.data,3,w,h,0,12,w*12)
        lib.oidnCommitFilter(f);check()
        lib.oidnExecuteFilter(f);check();lib.oidnReleaseFilter(f)
        return out
    rng=np.random.default_rng(123)
    h,w=args.height,args.width
    y,x=np.mgrid[:h,:w].astype(np.float32)
    color=np.stack((2+x/4,1+y/4,0.3+(x+y)/20),axis=-1)
    color*=rng.uniform(0.5,1.5,(h,w,1)).astype(np.float32)
    albedo=np.stack((0.2+x/w*0.5,0.1+y/h*0.6,np.full((h,w),.3,np.float32)),axis=-1)
    normal=np.zeros((h,w,3),np.float32);normal[:,:,2]=1
    normal[:,:,0]=rng.normal(0,.05,(h,w))
    normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
    af=run({"albedo":albedo},"rt_alb_large")
    nf=run({"normal":normal},"rt_nrm_large")
    output=run({"color":color,"albedo":af,"normal":nf},"rt_hdr_calb_cnrm_large")
    # Safetensors reader reverses dimensions: HWC -> Julia CWH. Store NCHW
    # instead, so all tensors read directly as WHCN, matching the graph runtime.
    def nchw(a):return np.ascontiguousarray(a.transpose(2,0,1)[None])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_file({k:nchw(v) for k,v in {"color":color,"albedo":albedo,"normal":normal,
        "filtered_albedo":af,"filtered_normal":nf,"output":output}.items()},str(args.output))
    lib.oidnReleaseDevice(dev)
    print("upstream full-pipeline reference generated",flush=True)
if __name__=="__main__":main()
