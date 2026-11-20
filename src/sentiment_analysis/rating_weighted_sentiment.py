"""Real ratings weight classification losses; neither features nor predictions."""
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup

from sentiment_analysis.paths import ROOT
from sentiment_analysis.evaluation import classification_metrics


def rating_weights(ratings, alpha=.5):
    values=np.asarray(ratings,dtype=np.float64)
    if not np.isfinite(values).all() or not np.isin(values,[1,2,3,4,7,8,9,10]).all():
        raise ValueError("Expected observed integer ratings 1-4 or 7-10.")
    if not 0<=alpha<=1:
        raise ValueError("alpha must be between 0 and 1.")
    return 1+alpha*(np.abs(values-5.5)-1.5)/3


def sha256(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def weighted_mean_loss(losses,weights,denominator=None):
    if losses.shape!=weights.shape or losses.ndim!=1 or not torch.isfinite(losses).all() or not torch.isfinite(weights).all() or (weights<=0).any():
        raise ValueError("Finite per-example losses and positive aligned weights required.")
    normalizer=weights.sum() if denominator is None else denominator
    if float(normalizer)<=0 or not math.isfinite(float(normalizer)):
        raise ValueError("Invalid loss normalizer.")
    return (losses*weights).sum()/normalizer


def load_data(config):
    frames={}
    for name,item in config['datasets'].items():
        path=ROOT/item['path']
        if sha256(path)!=item['sha256']:raise ValueError("Dataset changed: "+name)
        frame=pd.read_csv(path)
        if len(frame)!=item['rows'] or frame.text.isna().any():raise ValueError("Invalid dataset")
        if not frame.label.isin([0,1]).all() or not np.array_equal((frame.rating>=7).astype(int),frame.label):
            raise ValueError("Rating and sentiment labels disagree.")
        frames[name]=frame
    return frames


def encoded_records(frame,tokenizer,weights=None):
    features=tokenizer(frame.text.tolist(),truncation=True,max_length=256)
    return [{"input_ids":ids,"attention_mask":mask,"labels":int(label),
             "sample_weight":float(weights[index]) if weights is not None else 1.}
            for index,(ids,mask,label) in enumerate(zip(features['input_ids'],features['attention_mask'],frame.label))]


def evaluate(model,loader,device):
    model.eval(); probabilities=[]; labels=[]; loss_sum=0
    with torch.inference_mode():
        for batch in loader:
            batch.pop('sample_weight',None)
            y=batch.pop('labels').to(device)
            x={k:v.to(device) for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                logits=model(**x).logits.float()
            loss_sum+=float(torch.nn.functional.cross_entropy(logits,y,reduction='sum'))
            probabilities.extend(logits.softmax(-1)[:,1].cpu().tolist());labels.extend(y.cpu().tolist())
    p=np.array(probabilities)
    return {"LOSS":loss_sum/len(labels),**classification_metrics(labels,(p>.5).astype(int))},p


def train(config,mode,device='cuda'):
    output=ROOT/config['output_dir']/mode
    if output.exists():raise FileExistsError("Run exists; no overwrite: "+str(output))
    output.mkdir(parents=True)
    frames=load_data(config)
    torch.manual_seed(config['seed'])
    tokenizer=AutoTokenizer.from_pretrained(ROOT/config['base_model'],local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(ROOT/config['base_model'],local_files_only=True,
        num_labels=2,dtype=torch.float32,id2label={0:'NEGATIVE',1:'POSITIVE'},label2id={'NEGATIVE':0,'POSITIVE':1}).to(device)
    model.config.problem_type='single_label_classification'
    actual=rating_weights(frames['train'].rating,config['alpha']) if mode=='weighted' else np.ones(len(frames['train']))
    pd.DataFrame({'INPUT ROW':frames['train'].input_row,'RATING':frames['train'].rating,'LABEL':frames['train'].label,'WEIGHT':actual}).to_csv(output/'training_weights.csv',index=False)
    collator=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8)
    generator=torch.Generator().manual_seed(config['seed'])
    train_loader=torch.utils.data.DataLoader(encoded_records(frames['train'],tokenizer,actual),batch_size=16,
        shuffle=True,generator=generator,collate_fn=collator)
    val_loader=torch.utils.data.DataLoader(encoded_records(frames['validation'],tokenizer),batch_size=16,collate_fn=collator)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    steps=config['epochs']*math.ceil(len(train_loader)/2)
    scheduler=get_linear_schedule_with_warmup(optimizer,int(.1*steps),steps)
    history=[];best=float('inf')
    for epoch in range(config['epochs']):
        started=time.perf_counter();model.train(); total_weight=0; loss_sum=0
        for step,group in enumerate(itertools.batched(train_loader,2),1):
            denominator=sum(float(batch['sample_weight'].sum()) for batch in group)
            optimizer.zero_grad(set_to_none=True)
            for batch in group:
                weights=batch.pop('sample_weight').to(device).float(); labels=batch.pop('labels').to(device)
                inputs={k:v.to(device) for k,v in batch.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                    logits=model(**inputs).logits.float()
                    per_row=torch.nn.functional.cross_entropy(logits,labels,reduction='none')
                    numerator=(weights*per_row).sum()
                if not torch.isfinite(numerator):raise ValueError("Nonfinite loss")
                weighted_mean_loss(per_row,weights,denominator).backward()
                loss_sum+=float(numerator.detach());total_weight+=float(weights.sum())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step()
            if step%200==0:print(f'{mode} epoch {epoch+1}, update {step}/{math.ceil(len(train_loader)/2)}',flush=True)
        metrics,p=evaluate(model,val_loader,device)
        row={'MODE':mode,'EPOCH':epoch+1,'TRAIN WEIGHTED LOSS':loss_sum/total_weight,
             **{'VALIDATION '+k:v for k,v in metrics.items()},'SECONDS':time.perf_counter()-started}
        history.append(row); print(json.dumps(row),flush=True)
        if metrics['LOSS']<best:
            best=metrics['LOSS']; model.save_pretrained(output/'best_model');tokenizer.save_pretrained(output/'best_model')
            pd.DataFrame({'INPUT ROW':frames['validation'].input_row,'LABEL':frames['validation'].label,
                          'POSITIVE PROBABILITY':p,'PREDICTION':(p>.5).astype(int)}).to_csv(output/'validation_predictions.csv',index=False)
            (output/'best_epoch.json').write_text(json.dumps({'epoch':epoch+1,'validation':metrics}),encoding='utf-8')
        pd.DataFrame(history).to_csv(output/'history.csv',index=False)
    (output/'training_complete.json').write_text(json.dumps({'config':config,'mode':mode,
        'model_sha256':sha256(output/'best_model/model.safetensors'),'test_not_evaluated':True,
        'features':['input_ids','attention_mask'],'rating_role':'train loss weight only; not an input feature or output',
        'torch':torch.__version__},indent=2),encoding='utf-8')


def test_frozen_models(config,device='cuda'):
    # Both checkpoints must be complete and fixed before test scoring.
    root=ROOT/config['output_dir']
    for mode in ['plain','weighted']:
        meta=json.loads((root/mode/'training_complete.json').read_text())
        if meta['config']!=config or meta['model_sha256']!=sha256(root/mode/'best_model/model.safetensors'):
            raise ValueError("Training protocol/checkpoint changed.")
        if (root/mode/'test_predictions.csv').exists():raise FileExistsError("Test already evaluated.")
    comparison=json.loads((ROOT/'configs/rating_weighted_comparison.json').read_text(encoding='utf-8'))
    for item in comparison['sources'].values():
        if sha256(ROOT/item['path'])!=item['sha256']:raise ValueError("Comparison source changed.")
    frame=load_data(config)['test']; summaries=[]
    old=pd.read_csv(ROOT/'data/cleaned/test.csv').iloc[frame.input_row.to_numpy()].reset_index(drop=True)
    if not np.array_equal(old.label,frame.label) or not old.text.map(lambda x:hashlib.sha256(x.encode()).hexdigest()).eq(frame.input_text_sha256).all():
        raise ValueError("Common test review identity mismatch.")
    # Archived models saw original text. New models receive the identical bytes.
    frame=frame.copy();frame['text']=old.text.to_numpy()
    for mode in ['plain','weighted']:
        path=root/mode/'best_model';tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).to(device)
        loader=torch.utils.data.DataLoader(encoded_records(frame,tokenizer),batch_size=16,
                                          collate_fn=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8))
        metrics,p=evaluate(model,loader,device)
        table=pd.DataFrame({'INPUT ROW':frame.input_row,'LABEL':frame.label,'POSITIVE PROBABILITY':p,'PREDICTION':(p>.5).astype(int)})
        table.to_csv(root/mode/'test_predictions.csv',index=False)
        (root/mode/'test_evaluation.json').write_text(json.dumps({
            'checkpoint_sha256':sha256(path/'model.safetensors'),
            'predictions_sha256':sha256(root/mode/'test_predictions.csv'),
            'comparison_protocol':comparison,'rows':len(frame),'metrics':metrics,
            'rating_used_for_test_inference':False},indent=2),encoding='utf-8')
        summaries.append({'MODEL':'DistilBERT '+('rating-weighted CE' if mode=='weighted' else 'unweighted CE (matched control)'),**metrics})
        del model
        if device=='cuda':torch.cuda.empty_cache()
    pd.DataFrame(summaries).to_csv(root/'test_metrics.csv',index=False)
    print(json.dumps(summaries),flush=True)
