import argparse
import json
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
import whisper
from tqdm import tqdm
from whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from whisper.model import Whisper
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, Tokenizer, get_tokenizer


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-shot Audio Classification using Whisper")
    parser.add_argument("--audio", type=str, help="Path to an audio file to classify")
    parser.add_argument("--json", type=str, help="Path to a jsonl file containing audio paths")
    parser.add_argument(
        "--class_names", type=str, required=True, help="Path to a txt file containing class names"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--device", default="cuda", help="Device to use for inference")
    parser.add_argument(
        "--model",
        default="large",
        choices=whisper.available_models(),
        help="Name of the Whisper model to use",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="Language of the class names",
    )
    parser.add_argument("--output", type=str, default="result.json", help="Path to the output file")
    return parser


def calculate_audio_features(audio_path: str, model: Whisper) -> torch.Tensor:
    mel = log_mel_spectrogram(audio_path)
    segment = pad_or_trim(mel, N_FRAMES).to(model.device)
    return model.embed_audio(segment.unsqueeze(0))


@torch.no_grad()
def classify(
    model: Whisper,
    audio_path: str,
    class_names: List[str],
    tokenizer: Tokenizer,
    verbose: bool = False,
) -> str:
    initial_tokens = (
        torch.tensor(tokenizer.sot_sequence_including_notimestamps).unsqueeze(0).to(model.device)
    )
    eot_token = torch.tensor([tokenizer.eot]).unsqueeze(0).to(model.device)
    audio_features = calculate_audio_features(audio_path, model)

    average_logprobs = []
    for class_name in class_names:
        class_name_tokens = (
            torch.tensor(tokenizer.encode(" " + class_name)).unsqueeze(0).to(model.device)
        )
        input_tokens = torch.cat([initial_tokens, class_name_tokens, eot_token], dim=1)

        logits = model.logits(input_tokens, audio_features)  # (1, T, V)
        logprobs = F.log_softmax(logits, dim=-1).squeeze(0)  # (T, V)
        logprobs = logprobs[len(tokenizer.sot_sequence_including_notimestamps) - 1 : -1]  # (T', V)
        logprobs = torch.gather(logprobs, dim=-1, index=class_name_tokens.view(-1, 1))  # (T', 1)
        average_logprob = logprobs.mean().item()
        average_logprobs.append(average_logprob)

    sorted_indices = sorted(
        range(len(average_logprobs)), key=lambda i: average_logprobs[i], reverse=True
    )
    if verbose:
        tqdm.write("  Average log probabilities for each class:")
        for i in sorted_indices:
            tqdm.write(f"    {class_names[i]}: {average_logprobs[i]}")

    return class_names[sorted_indices[0]]


@dataclass
class AudioData:
    audio_path: str
    category: Optional[str] = None


def read_json(path: str) -> List[AudioData]:
    records = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            record = AudioData(
                audio_path=data["audio_path"],
                category=data["category"] if "category" in data else None,
            )
            records.append(record)
    return records


def read_class_names(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f]


def write_result(records: List[AudioData], results: List[str], path: str) -> None:
    with open(path, "w") as f:
        for record, result in zip(records, results):
            f.write(
                json.dumps(
                    {
                        "audio_path": record.audio_path,
                        "category": record.category,
                        "recognized": result,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def calc_accuracy(records: List[AudioData], results: List[str]) -> float:
    correct = 0
    for record, result in zip(records, results):
        if record.category == result:
            correct += 1
    return correct / len(records)


def main():
    args = get_parser().parse_args()

    if args.audio is None and args.json is None:
        raise ValueError("Either --audio or --json must be specified")

    if args.audio is not None:
        records = [AudioData(audio_path=args.audio)]
    else:
        records = read_json(args.json)

    class_names = read_class_names(args.class_names)

    tokenizer = get_tokenizer(multilingual=".en" not in args.model, language=args.language)
    model = whisper.load_model(args.model, args.device)

    results = []
    for record in tqdm(records):
        tqdm.write(f"processing {record.audio_path} (class: {record.category})")
        result = classify(
            model=model,
            audio_path=record.audio_path,
            class_names=class_names,
            tokenizer=tokenizer,
            verbose=args.verbose,
        )
        results.append(result)
        tqdm.write(f"  predicted: {result}")

    write_result(records, results, args.output)
    accuracy = calc_accuracy(records, results)
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
