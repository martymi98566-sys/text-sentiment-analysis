"""Checkpointed research annotation, with the frozen policy and exact evidence spans."""
import gc,json,time
from pathlib import Path
import numpy as np
import torch
from sentiment_analysis.paths import ROOT
from sentiment_analysis.reason_training import load_json,write_json,digest
from sentiment_analysis.reason_repair import load_model,hypotheses,log
from sentiment_analysis.reason_context_trial import setup,cap_categories
from sentiment_analysis.reason_semantic_pipeline import safe_fragments,context_view,aspects,QUALITY_HYPOTHESES,SCOPE_HYPOTHESES
from sentiment_analysis.reason_verified_context import hypothesis


def sorted_scores(model,tok,pairs):
    """Length grouping changes padding only. No truncation or candidate approximation."""
    if not pairs:return np.zeros(0,dtype=np.float32)
    enc=tok([p[0] for p in pairs],[p[1] for p in pairs],truncation=False,padding=False,verbose=False)
    lengths=np.array([len(x) for x in enc['input_ids']]);assert lengths.max()<=512
    order=np.argsort(lengths,kind='stable');out=np.zeros(len(pairs),dtype=np.float32)
    cursor=0
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        while cursor<len(order):
            length=int(lengths[order[min(cursor+95,len(order)-1)]])
            batch=96 if length<=128 else 48 if length<=256 else 16
            ids=order[cursor:cursor+batch]
            entries=[{k:enc[k][int(i)] for k in enc} for i in ids]
            tensor=tok.pad(entries,padding=True,return_tensors='pt').to('cuda')
            out[ids]=model(**tensor).logits.float().softmax(-1)[:,0].cpu().numpy()
            cursor+=len(ids)
    return out


class PrecisionAnnotator:
    def __init__(self,policy):
        setup();self.policy=policy
        self.names=load_json(ROOT/'data/reason_training/v2_2000/split_manifest.json')['label_names']
        self.tok,self.local=load_model(ROOT/'data/reason_training/repair_20261119/nli_evidence/model')
        _,self.context=load_model(ROOT/'data/reason_training/context_trial_20261119/model')
        _,self.gate=load_model(ROOT/'data/pretrained/topic_nli')
        self.local_hs=hypotheses(self.names);self.context_hs=[hypothesis(n,aspects()) for n in self.names]
        self.longest=max(self.context_hs+SCOPE_HYPOTHESES,key=lambda h:len(self.tok(h)['input_ids']))

    def score(self,records,local_screen=None):
        started=time.perf_counter();p=self.policy;t=p['threshold'];fs=[];owners=[]
        for ri,row in enumerate(records):
            fragments=safe_fragments(row['review_id'],row['text'],self.tok)
            fs.extend(fragments);owners.extend([ri]*len(fragments))
        owners=np.array(owners,dtype=np.int32)
        # Deduplicate identical local evidence only. Full review context is never deduplicated away.
        unique=list(dict.fromkeys(f['text'] for f in fs));lookup={x:i for i,x in enumerate(unique)}
        local_pairs=[(text,h) for text in unique for h in self.local_hs]
        proposed=np.ones(len(local_pairs),dtype=bool) if local_screen is None else local_screen(local_pairs)
        local_flat=np.zeros(len(local_pairs),dtype=np.float32)
        local_flat[proposed]=sorted_scores(self.local,self.tok,[pair for pair,keep in zip(local_pairs,proposed) if keep])
        local_unique=local_flat.reshape(-1,22)
        local=local_unique[[lookup[f['text']] for f in fs]] if fs else np.zeros((0,22),np.float32)
        promising=np.flatnonzero(local.max(1)>=t)
        q=np.zeros((len(fs),3),np.float32);scope=np.zeros((len(fs),2),np.float32);context=np.zeros_like(local)
        unique_q=list(dict.fromkeys(fs[int(i)]['text'] for i in promising));ql={x:i for i,x in enumerate(unique_q)}
        if local_screen is not None and hasattr(local_screen,'quality_cache'):
            uq=np.asarray([local_screen.quality_cache[text] for text in unique_q],dtype=np.float32).reshape(-1,3)
        else:
            uq=sorted_scores(self.gate,self.tok,[(text,h) for text in unique_q for h in QUALITY_HYPOTHESES]).reshape(-1,3)
        if len(promising):q[promising]=uq[[ql[fs[int(i)]['text']] for i in promising]]
        non=np.maximum(q[:,1],q[:,2]);allowed=(q[:,0]>=p['quality_min'])&~((non>=p['non_evaluation_reject'])&(non>q[:,0]))
        qpass=[int(i) for i in promising if allowed[int(i)]]
        for i in qpass:
            f=fs[i];raw=records[int(owners[i])]['text']
            f['context_text'],f['context_window']=context_view(raw,f['start'],f['end'],self.tok,self.longest)
        sc=sorted_scores(self.gate,self.tok,[(fs[i]['context_text'],h) for i in qpass for h in SCOPE_HYPOTHESES]).reshape(-1,2)
        if qpass:scope[qpass]=sc
        positions=[(i,int(j)) for i in qpass if scope[i,1]<p['external_veto'] for j in np.flatnonzero(local[i]>=t)]
        values=sorted_scores(self.context,self.tok,[(fs[i]['context_text'],self.context_hs[j]) for i,j in positions])
        for (i,j),v in zip(positions,values):context[i,j]=v
        score=np.minimum(local,context);flags=(score>=t)&allowed[:,None]&(scope[:,1]<p['external_veto'])[:,None]
        flags[:,20:] &= ~flags[:,:20].any(1,keepdims=True)
        accepted=np.where(flags,score,0);agg=np.zeros((len(records),22));np.maximum.at(agg,owners,accepted)
        pred=cap_categories(agg,agg>=t,p['max_categories'])
        results=[]
        for ri,row in enumerate(records):
            evidence=[];labels=[]
            for j in np.flatnonzero(pred[ri]):
                candidates=np.flatnonzero((owners==ri)&flags[:,j]);best=int(candidates[np.argmax(score[candidates,j])]);f=fs[best]
                assert row['text'][f['start']:f['end']]==f['text']
                labels.append(self.names[int(j)])
                evidence.append({'label':self.names[int(j)],'start':f['start'],'end':f['end'],'quote':f['text'],
                  'local_score':float(local[best,j]),'context_score':float(context[best,j]),'score':float(score[best,j]),
                  'specific_evaluation_score':float(q[best,0]),'external_work_score':float(scope[best,1]),'context_window':f['context_window']})
            results.append({**row,'reason_categories':list(dict.fromkeys(n.split(':')[0] for n in labels)),
               'reason_labels':labels,'reason_evidence':evidence,'reason_status':'supported' if labels else 'abstained',
               'label_source':'local_model_precision_v2_weak_annotation','human_reviewed':False,
               'overall_sentiment_generated':False})
        stats={'reviews':len(records),'fragments':len(fs),'unique_local_fragments':len(unique),'promising_fragments':len(promising),
          'quality_pass':len(qpass),'context_candidate_pairs':len(positions),'seconds':time.perf_counter()-started,
          'local_pairs_total':len(local_pairs),'local_pairs_proposed':int(proposed.sum()),
          'windowed_scored_fragments':sum(f.get('context_window',{}).get('windowed',False) for f in fs),
          'max_context_tokens':max([f.get('context_window',{}).get('pair_tokens',0) for f in fs]+[0]),'silent_truncation':False}
        return results,stats,{'local':local,'context_after':context,'quality':q,'scope':scope,'owners':owners},fs
