"""Known-label masks, signed-category collapse, and actual target pooling invariants."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from transformers import AutoTokenizer
from sentiment_analysis.reason_fragment_model import (TargetReasonModel,encode_focus,partial_category_targets,collapse_signed_verified,masked_loss)
ROOT=Path(__file__).resolve().parents[1]


class FragmentModelTests(unittest.TestCase):
    def test_missing_category_is_unknown_not_negative(self):
        target,mask=partial_category_targets({'acting'},set(),['plot','acting','sound'])
        self.assertEqual(target,[0.,1.,0.]);self.assertEqual(mask,[0.,1.,0.])
    def test_rejecting_one_polarity_is_not_rejecting_category(self):
        self.assertEqual(collapse_signed_verified({'acting:negative':1}),(set(),set()))
        self.assertEqual(collapse_signed_verified({'acting:negative':1,'acting:positive':0}),({'acting'},set()))
        self.assertEqual(collapse_signed_verified({'acting:negative':1,'acting:positive':1}),(set(),{'acting'}))
    def test_unknown_logits_do_not_receive_supervised_negative_gradient(self):
        logits=torch.tensor([[0.,0.,0.]],requires_grad=True)
        loss=masked_loss(logits,torch.tensor([0.]),torch.tensor([[0.,1.,0.]]),torch.tensor([[0.,1.,0.]]),
                         torch.tensor([1.]),torch.tensor([0.]),sparsity=0.)
        loss.backward();self.assertEqual(logits.grad[0,0].item(),0.);self.assertLess(logits.grad[0,1].item(),0.);self.assertEqual(logits.grad[0,2].item(),0.)
    def test_cardinality_prior_penalizes_excess_only_for_partial_positives(self):
        logits=torch.full((1,3),5.,requires_grad=True)
        common=(torch.zeros(1),torch.tensor([[1.,0.,0.]]),torch.tensor([[1.,0.,0.]]),torch.ones(1),torch.zeros(1))
        base=masked_loss(logits,*common,sparsity=0.,cardinality_weight=0.)
        sparse=masked_loss(logits,*common,sparsity=0.,cardinality_weight=.15)
        self.assertGreater(sparse.item(),base.item())
        full=(common[0],common[1],torch.ones(1,3),common[3],common[4])
        self.assertAlmostEqual(masked_loss(logits,*full,sparsity=0.,cardinality_weight=0.).item(),
                               masked_loss(logits,*full,sparsity=0.,cardinality_weight=.15).item())
    @unittest.skip("Private annotation corpus is not distributed; original integrity audit retained locally")
    def test_v2_review_and_target_phrase_isolation(self):
        rows=json.loads((ROOT/'data/reason_training/target_fragments_v2_20261120/dataset.json').read_text(encoding='utf-8'))
        train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='development']
        self.assertFalse({r['group_id'] for r in train}&{r['group_id'] for r in dev})
        canon=lambda s:' '.join(s.lower().split())
        self.assertFalse({canon(r['quote']) for r in train}&{canon(r['quote']) for r in dev})
        self.assertTrue(all(r['text'][r['start']:r['end']]==r['quote'] for r in rows))
    def test_same_context_different_focus_changes_representation(self):
        class Encoder(nn.Module):
            config=SimpleNamespace(hidden_size=2)
            def forward(self,input_ids,attention_mask):
                return SimpleNamespace(last_hidden_state=torch.tensor([[[1.,0.],[0.,1.]]]).expand(input_ids.shape[0],-1,-1))
        model=TargetReasonModel(Encoder(),['acting']).eval()
        with torch.no_grad():model.classifier.weight.zero_();model.classifier.weight[0,0]=1.;model.classifier.weight[0,1]=-1.;model.classifier.bias.zero_()
        x={'input_ids':torch.tensor([[1,2]]),'attention_mask':torch.ones(1,2)}
        a=model(**x,target_mask=torch.tensor([[1.,0.]]))[0];b=model(**x,target_mask=torch.tensor([[0.,1.]]))[0]
        self.assertFalse(torch.equal(a,b))
    @unittest.skip("Private annotation corpus is not distributed; original integrity audit retained locally")
    def test_training_group_isolation_and_no_benchmark_reuse(self):
        cfg=json.loads((ROOT/'configs/reason_fragment_training_v1.json').read_text(encoding='utf-8'))
        rows=json.loads((ROOT/cfg['output_dir']/'dataset.json').read_text(encoding='utf-8'));bench=json.loads((ROOT/'data/reason_composition/context_benchmark_v1_20261120/samples.json').read_text(encoding='utf-8'))
        train={r['group_id'] for r in rows if r['split']=='train'};dev={r['group_id'] for r in rows if r['split']=='development'}
        self.assertFalse(train&dev);quotes={' '.join(r['quote'].lower().split()) for r in rows}
        self.assertFalse(quotes&{' '.join(r['quote'].lower().split()) for r in bench})
    def test_tokenization_keeps_focus_and_bounds_context(self):
        tokenizer=AutoTokenizer.from_pretrained(ROOT/'data/pretrained/topic_nli',local_files_only=True)
        text=('Background facts. '*100)+'The cast performs beautifully.'+(' Later events. '*100)
        quote='The cast performs beautifully.';start=text.index(quote)
        enc=encode_focus(tokenizer,text,start,start+len(quote),64)
        self.assertLessEqual(len(enc['input_ids']),64);self.assertGreater(sum(enc['target_mask']),0)
        self.assertEqual(enc['target_mask'][0],0);self.assertEqual(enc['target_mask'][-1],0)
        with self.assertRaises(ValueError):encode_focus(tokenizer,text,0,len(text),64)


if __name__=='__main__':unittest.main()
