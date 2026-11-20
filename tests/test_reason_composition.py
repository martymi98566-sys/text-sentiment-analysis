"""Word-budget invariants and transparent single-review weighting."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("composition_tested", ROOT/"src/sentiment_analysis/reason_composition.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
WEIGHTS = {"mild": 1., "ordinary": 1.5, "strong": 2.}


def evidence(text, quote, category="acting", polarity="positive", intensity="ordinary"):
    start = text.index(quote)
    return {"start": start, "end": start+len(quote), "quote": quote,
            "category": category, "polarity": polarity, "intensity": intensity}


class CompositionTests(unittest.TestCase):
    def test_same_type_two_polarities_survive(self):
        text = "Good acting. Bad acting."
        r = m.composition(text, [evidence(text,"Good acting."), evidence(text,"Bad acting.",polarity="negative")], WEIGHTS)
        self.assertEqual({x["polarity"] for x in r["slices"]}, {"positive", "negative"})
        self.assertAlmostEqual(sum(x["percent"] for x in r["slices"]), 100)
        self.assertEqual(r["covered_words"],4)

    def test_duplicate_quotes_do_not_inflate_length(self):
        text = "The acting is excellent."
        e = evidence(text,text)
        a = m.composition(text,[e],WEIGHTS)
        b = m.composition(text,[e,e,e],WEIGHTS)
        self.assertEqual(a,b)

    def test_two_aspects_share_same_word_budget(self):
        text = "Acting and music are excellent."
        r = m.composition(text,[evidence(text,text),evidence(text,text,category="sound")],WEIGHTS)
        self.assertEqual(r["covered_words"],5)
        self.assertAlmostEqual(sum(x["word_credit"] for x in r["slices"]),5)
        self.assertAlmostEqual(r["total_weight"],7.5)
        self.assertEqual([x["percent"] for x in r["slices"]],[50.,50.])

    def test_length_times_tone_and_no_confidence_multiplier(self):
        text = "Acting good. Plot very bad."
        e1=evidence(text,"Acting good.",intensity="mild")
        e2=evidence(text,"Plot very bad.",category="plot",polarity="negative",intensity="strong")
        e1['confidence']=.01;e2['confidence']=.99
        r=m.composition(text,[e1,e2],WEIGHTS)
        self.assertEqual(r["total_weight"],8)
        self.assertEqual(r["slices"][0]["percent"],75)

    def test_facts_not_in_denominator(self):
        text="A detective walks through London. Acting good."
        r=m.composition(text,[evidence(text,"Acting good.")],WEIGHTS)
        self.assertEqual(r["covered_words"],2)
        self.assertEqual(r["total_words"],7)
        self.assertEqual(r["slices"][0]["percent"],100)
        self.assertLess(r["evidence_word_coverage"],1)

    def test_empty_is_not_an_other_slice(self):
        r=m.composition("Great, recommended.",[],WEIGHTS)
        self.assertEqual(r["slices"],[])
        self.assertEqual(r["status"],"no_supported_reason")

    def test_evidence_offset_is_checked(self):
        text="The acting is excellent."
        e=evidence(text,text);e['end']-=1
        with self.assertRaises(ValueError):m.composition(text,[e],WEIGHTS)

    def test_split_keeps_source_and_opposite_clauses(self):
        text="The acting is good, the plot is bad, and the music is lovely."
        spans=m.split_clause_spans(text,0,len(text))
        self.assertEqual(len(spans),3)
        self.assertTrue(all(a<b for a,b in spans))
        self.assertTrue(all(spans[i][1]<=spans[i+1][0] for i in range(2)))
        self.assertIn('plot is bad',text[spans[1][0]:spans[1][1]])


if __name__=='__main__':unittest.main()
