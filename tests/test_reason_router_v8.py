"""Contrastive target protocol, unchanged text/polarity, fresh reference isolation."""
import copy,hashlib,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sentiment_analysis.reason_router_v8 import assign,prompt,support,apply_router


class RouterV8Tests(unittest.TestCase):
    def setUp(self):
        self.policy={'name':'test','learned_veto':.5,'semantic_min':.25,'mode':'union'}
        text='The story is clever and the music is lovely.'
        self.unit={'unit_id':'x','text':text,'quote':text,'start':0,'end':len(text),'status':'specific_evaluation','reason_categories':['plot'],
            'reason_classification':{'reason':'ranked_best_fit'},'polarity':'positive','intensity':'ordinary',
            'local_aspect_scores':{'plot:positive':.9,'sound:positive':.3},'reason_type_scores':{}}
        self.pred={'route':{'status':'scored','categories':['sound']},'learned':{'status':'scored','specific_score':.99}}
    def test_reference_labels_are_not_in_model_prompt(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):return json.dumps(messages)
        rendered=prompt(Tokenizer(),dict(self.unit,categories=['GOLD_LABEL_MUST_NOT_LEAK']))
        self.assertNotIn('GOLD_LABEL_MUST_NOT_LEAK',rendered)
        self.assertIn('EXACT TARGET TO LABEL',rendered)
    def test_semantic_union_preserves_original_fields(self):
        original=copy.deepcopy(self.unit);self.assertEqual(assign(self.unit,self.pred,self.policy),['plot','sound']);self.assertEqual(self.unit,original)
    def test_strict_original_labels_and_output_failures_are_preserved(self):
        u=dict(self.unit,reason_classification={'reason':'preserve_existing_supported_categories'})
        p=copy.deepcopy(self.pred);p['learned']['specific_score']=0.
        self.assertEqual(assign(u,p,self.policy),['plot'])
        p['route']['status']='abstain_context_too_long';self.assertEqual(assign(self.unit,p,self.policy),['plot'])
    def test_only_model_selection_changes_not_pie_intensity(self):
        row={'text':self.unit['text'],'emotion_units':[self.unit],'reason_policy':'semantic_backfill_v2'}
        with patch('sentiment_analysis.reason_router_v8.score_records',return_value=[self.pred]):out=apply_router(row,self.policy)
        for key in ['text','quote','start','end','polarity','intensity']:
            self.assertEqual(out['emotion_units'][0][key],self.unit[key])
        self.assertAlmostEqual(sum(s['percent'] for s in out['composition']['slices']),100.)
        self.assertAlmostEqual(sum(s['word_credit'] for s in out['composition']['slices']),out['composition']['covered_words'])
    def test_appearance_and_rap_boundaries_use_existing_categories(self):
        self.assertFalse(support(dict(self.unit,quote='The actress has extraordinary beauty.'),'acting',.25))
        self.assertTrue(support(dict(self.unit,quote='The rapping and lyrics are dreadful.'),'sound',.25))
    @unittest.skip("Private annotation corpus is not distributed; original integrity audit retained locally")
    def test_final_references_are_frozen_and_review_disjoint(self):
        out=ROOT/'data/reason_training/reason_router_v8_20261120';final=json.loads((out/'final_references.json').read_text(encoding='utf-8'))
        protocol=json.loads((out/'final_reference_protocol.json').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256((out/'final_references.json').read_bytes()).hexdigest(),protocol['reference_sha256'])
        prior=[]
        for path in ['data/reason_annotation/pilot_20261119/desktop_handoff_2000/all_2000_inputs.json',
                     'data/reason_training/natural_focus_v3_20261120/annotations.json','data/reason_composition/context_benchmark_v1_20261120/samples.json']:
            prior+=json.loads((ROOT/path).read_text(encoding='utf-8'))
        canon=lambda s:' '.join(s.lower().split())
        self.assertFalse({canon(r['text']) for r in final}&{canon(r['text']) for r in prior})
        self.assertTrue(all(r['quote']==r['text'][r['start']:r['end']] for r in final))
        self.assertEqual(len({r['sample_id'] for r in final}),len(final))
    def test_runtime_model_failure_keeps_previous_result(self):
        row={'emotion_units':[self.unit],'reason_policy':'semantic_backfill_v2'}
        with patch('sentiment_analysis.reason_router_v8.score_records',side_effect=OSError('missing model')):out=apply_router(row,self.policy)
        self.assertEqual(out['emotion_units'],row['emotion_units']);self.assertEqual(out['reason_policy'],'semantic_backfill_v2')


if __name__=='__main__':unittest.main()
