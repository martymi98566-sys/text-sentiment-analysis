"""Focus-pooled reason classifier with masked weak-label supervision."""
import json
from pathlib import Path
import torch
from torch import nn
from transformers import AutoConfig,AutoModel


def encode_focus(tokenizer,text,start,end,max_length=224):
    if not 0<=start<end<=len(text):raise ValueError('Invalid focus offsets')
    raw=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,truncation=False,verbose=False)
    offsets=raw['offset_mapping'];selected=[i for i,(a,b) in enumerate(offsets) if b>a and a<end and b>start]
    if not selected:raise ValueError('Empty target token set')
    room=max_length-tokenizer.num_special_tokens_to_add(pair=False)
    if selected[-1]-selected[0]+1>room:raise ValueError('Target itself exceeds token budget; never truncate evidence')
    lo=max(0,selected[0]-(room-len(selected))//2);hi=min(len(offsets),lo+room);lo=max(0,hi-room)
    while True:
        char_a=offsets[lo][0];char_b=offsets[hi-1][1]
        encoded=tokenizer(text[char_a:char_b],add_special_tokens=True,return_offsets_mapping=True,truncation=False,verbose=False)
        if len(encoded['input_ids'])<=max_length:break
        if selected[0]-lo>=hi-selected[-1]-1 and lo<selected[0]:lo+=1
        elif hi>selected[-1]+1:hi-=1
        else:raise ValueError('Target itself exceeds formatted token budget')
    focus=[int(b>a and char_a+a<end and char_a+b>start) for a,b in encoded.pop('offset_mapping')]
    encoded['target_mask']=focus
    encoded.pop('token_type_ids',None)
    if not any(focus):raise ValueError('Target lost during input formatting')
    return dict(encoded)


def collate_focus(tokenizer,rows):
    masks=[r['target_mask'] for r in rows]
    data=tokenizer.pad([{k:v for k,v in r.items() if k!='target_mask'} for r in rows],padding=True,return_tensors='pt')
    n=data['input_ids'].shape[1]
    data['target_mask']=torch.tensor([m+[0]*(n-len(m)) for m in masks],dtype=torch.float32)
    return data


def partial_category_targets(positive,negative,label_names):
    if set(positive)&set(negative):raise ValueError('Conflicting category supervision')
    targets=[float(c in positive) for c in label_names]
    known=[float(c in positive or c in negative) for c in label_names]
    return targets,known


def collapse_signed_verified(labels):
    positive=set();negative=set()
    for c in {name.split(':')[0] for name in labels}:
        if any(labels.get(c+':'+s)==0 for s in ['positive','negative']):positive.add(c)
        elif all(labels.get(c+':'+s)==1 for s in ['positive','negative']):negative.add(c)
    return positive,negative


class TargetReasonModel(nn.Module):
    def __init__(self,encoder,labels):
        super().__init__();self.encoder=encoder;self.labels=list(labels)
        size=encoder.config.hidden_size
        self.norm=nn.LayerNorm(size*2);self.dropout=nn.Dropout(.1)
        self.classifier=nn.Linear(size*2,len(labels));self.specific=nn.Linear(size*2,1)

    def forward(self,input_ids,attention_mask,target_mask):
        hidden=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        weights=target_mask.to(hidden.dtype)*attention_mask.to(hidden.dtype)
        focused=(hidden*weights.unsqueeze(-1)).sum(1)/weights.sum(1,keepdim=True).clamp_min(1)
        representation=self.dropout(self.norm(torch.cat([focused,hidden[:,0]],dim=-1)))
        return self.classifier(representation),self.specific(representation).squeeze(-1)

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        self.encoder.save_pretrained(directory/'encoder')
        torch.save({'norm':self.norm.state_dict(),'classifier':self.classifier.state_dict(),'specific':self.specific.state_dict(),'labels':self.labels},directory/'heads.pt')

    @classmethod
    def load(cls,directory,device='cuda'):
        directory=Path(directory);heads=torch.load(directory/'heads.pt',map_location='cpu',weights_only=True)
        obj=cls(AutoModel.from_pretrained(directory/'encoder',local_files_only=True,dtype=torch.float32),heads['labels'])
        obj.norm.load_state_dict(heads['norm']);obj.classifier.load_state_dict(heads['classifier']);obj.specific.load_state_dict(heads['specific'])
        return obj.to(device).eval()


def masked_loss(logits,gate,targets,known,gate_targets,gate_known,negative_gate_weight=1.,sparsity=.02,gate_weight=.25,cardinality_weight=0.):
    per=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),targets,reduction='none')
    category=(per*known).sum()/known.sum().clamp_min(1)
    unknown=1-known
    regularization=(logits.float().sigmoid()*unknown).sum()/unknown.sum().clamp_min(1)
    gw=gate_known*torch.where(gate_targets>0,1.,negative_gate_weight)
    gl=torch.nn.functional.binary_cross_entropy_with_logits(gate.float(),gate_targets,reduction='none')
    partial=(known.sum(1)<known.shape[1])&(targets.sum(1)>0)
    excess=torch.relu(logits.float().sigmoid().sum(1)-targets.sum(1)-.5)
    cardinality=(excess*partial).sum()/partial.sum().clamp_min(1)
    return category+sparsity*regularization+gate_weight*(gl*gw).sum()/gw.sum().clamp_min(1)+cardinality_weight*cardinality
