"""Regression tests for broken shared predicates and polarity label routing."""
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sentence_first_tested',ROOT/'src/sentiment_analysis/reason_sentence_first.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SentenceFirstTests(unittest.TestCase):
    def test_shared_predicate_is_not_split(self):
        text='The acting and the music are excellent.'
        self.assertEqual(m.clause_spans(text,0,len(text)),[(0,len(text))])
    def test_enumerated_objects_keep_their_verb(self):
        text='Photography captures the time, the violence, and flavor of rural Punjab very well.'
        self.assertEqual(m.clause_spans(text,0,len(text)),[(0,len(text))])
    def test_independent_clauses_are_split(self):
        text='The acting is excellent, the plot is terrible, and the music is pleasant.'
        spans=m.clause_spans(text,0,len(text))
        self.assertEqual(len(spans),3)
        self.assertTrue(all(m.FINITE.search(text[a:b]) for a,b in spans))
    def test_negative_nli_is_never_routed_to_positive(self):
        u={'status':'specific_evaluation','polarity_nli_scores':[.001,.99,.1],'local_aspect_scores':{}}
        self.assertEqual(m.decide_local_sign(u)[0],'negative')
    def test_fact_and_unclear_are_not_forced_to_a_direction(self):
        u={'status':'factual','polarity_nli_scores':[.01,.99,.1],'local_aspect_scores':{}}
        self.assertIsNone(m.decide_local_sign(u)[0])
        u.update(status='uncertain')
        self.assertIsNone(m.decide_local_sign(u)[0])

if __name__=='__main__':unittest.main()
