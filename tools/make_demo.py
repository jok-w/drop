"""Create a harmless textured moving-square clip for pipeline verification."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_tracking import synthetic_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="New directory for demo.avi and ground_truth.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    truth = synthetic_video(output / "demo.avi")
    (output / "ground_truth.json").write_text(json.dumps(truth), encoding="utf-8")
    print(f"Created {output.resolve() / 'demo.avi'}; initial ROI: 40 100 32 32")


if __name__ == "__main__":
    main()
