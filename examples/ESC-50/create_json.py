import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import List, Set


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create input files for ESC-50 dataset")
    parser.add_argument(
        "--csv", type=str, required=True, help="Path to `esc50.csv` from ESC-50 dataset"
    )
    parser.add_argument(
        "--wav-dir", type=str, required=True, help="Path to `audio` directory from ESC-50 dataset"
    )
    parser.add_argument("--json", type=str, default="data.json", help="Path to output jsonl file")
    parser.add_argument(
        "--class_names",
        type=str,
        default="class_names.txt",
        help="Path to output class_names.txt file",
    )
    return parser


@dataclass
class Record:
    audio_path: str
    category: str


def write_records(records: List[Record], output: str):
    with open(output, "w") as f:
        for record in records:
            f.write(
                json.dumps(
                    {"audio_path": record.audio_path, "category": record.category},
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_class_names(lines: Set[str], output: str) -> None:
    with open(output, "w") as f:
        for line in lines:
            f.write(f"{line}\n")


def main():
    args = get_parser().parse_args()

    wav_dir = os.path.abspath(args.wav_dir)
    records = []
    categories = set()
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = f"[{row['category']}]"
            records.append(Record(f"{wav_dir}/{row['filename']}", category))
            categories.add(category)

    write_records(records, args.json)
    write_class_names(categories, args.class_names)


if __name__ == "__main__":
    main()
