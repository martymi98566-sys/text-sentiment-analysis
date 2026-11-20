"""Run the fixed binary sentiment/rating-band auxiliary experiment."""
import argparse
import json
from sentiment_analysis.paths import ROOT
from sentiment_analysis.rating_auxiliary_sentiment import pilot, confirm, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pilot", "confirm", "test", "all"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/rating_auxiliary_sentiment.json").read_text(encoding="utf-8"))
    for stage, function in [("pilot", pilot), ("confirm", confirm), ("test", test)]:
        if args.stage in [stage, "all"]:
            function(config, args.device)


if __name__ == "__main__":
    main()
