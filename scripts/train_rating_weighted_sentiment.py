"""Run a separate rating-weighted sentiment experiment; no score regression."""
import argparse
import json
from sentiment_analysis.paths import ROOT
from sentiment_analysis.rating_weighted_sentiment import train,test_frozen_models


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['plain','weighted','test'],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args()
    config=json.loads((ROOT/'configs/rating_weighted_sentiment.json').read_text(encoding='utf-8'))
    if args.mode=='test':test_frozen_models(config,args.device)
    else:train(config,args.mode,args.device)


if __name__=='__main__':main()
