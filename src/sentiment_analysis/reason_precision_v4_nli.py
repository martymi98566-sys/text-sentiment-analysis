"""The original NLI score formula with bounded CUDA workspace for this experiment."""
import json,time
import numpy as np
import torch

def low_memory_scores(model,tok,pairs):
    if not pairs:return np.zeros(0,dtype=np.float32)
    started=time.perf_counter()
    enc=tok([p[0] for p in pairs],[p[1] for p in pairs],truncation=False,padding=False,verbose=False)
    lengths=np.array([len(x) for x in enc['input_ids']]);assert lengths.max()<=512
    order=np.argsort(lengths,kind='stable');out=np.zeros(len(pairs),dtype=np.float32);cursor=0;last_log=0
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        while cursor<len(order):
            length=int(lengths[order[min(cursor+31,len(order)-1)]])
            batch=32 if length<=128 else 16 if length<=256 else 4
            ids=order[cursor:cursor+batch]
            entries=[{k:enc[k][int(i)] for k in enc} for i in ids]
            tensor=tok.pad(entries,padding=True,return_tensors='pt').to('cuda')
            logits=model(**tensor).logits.float();out[ids]=logits.softmax(-1)[:,0].cpu().numpy()
            del tensor,logits;torch.cuda.empty_cache();cursor+=len(ids)
            if cursor-last_log>=1000 or cursor==len(order):
                print(json.dumps({'stage':'nli_pairs','done':cursor,'total':len(order),'seconds':round(time.perf_counter()-started,1)}),flush=True);last_log=cursor
    torch.cuda.empty_cache()
    return out
