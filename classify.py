import argparse
import json
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
import whisper
from tqdm import tqdm
from whisper.audio import N_FRAMES, N_MELS, log_mel_spectrogram, pad_or_trim
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
    parser.add_argument("--subtract_internal_lm_score", action="store_true")
    parser.add_argument(
        "--no_subtract_internal_lm_score", action="store_false", dest="subtract_internal_lm_score"
    )
    parser.set_defaults(subtract_internal_lm_score=True)
    parser.add_argument("--output", type=str, default="result.json", help="Path to the output file")
    return parser


@torch.no_grad()
def calculate_audio_features(audio_path: Optional[str], model: Whisper) -> torch.Tensor:
    if audio_path is None:
        segment = torch.zeros((N_MELS, N_FRAMES), dtype=torch.float32).to(model.device)
    else:
        mel = log_mel_spectrogram(audio_path)
        segment = pad_or_trim(mel, N_FRAMES).to(model.device)
    return model.embed_audio(segment.unsqueeze(0))


@torch.no_grad()
def calculate_average_logprobs(
    model: Whisper,
    audio_features: torch.Tensor,
    class_names: List[str],
    tokenizer: Tokenizer,
) -> torch.Tensor:
    initial_tokens = (
        torch.tensor(tokenizer.sot_sequence_including_notimestamps).unsqueeze(0).to(model.device)
    )
    eot_token = torch.tensor([tokenizer.eot]).unsqueeze(0).to(model.device)

    average_logprobs = torch.zeros(len(class_names))
    for i, class_name in enumerate(class_names):
        class_name_tokens = (
            torch.tensor(tokenizer.encode(" " + class_name)).unsqueeze(0).to(model.device)
        )
        input_tokens = torch.cat([initial_tokens, class_name_tokens, eot_token], dim=1)

        logits = model.logits(input_tokens, audio_features)  # (1, T, V)
        logprobs = F.log_softmax(logits, dim=-1).squeeze(0)  # (T, V)
        logprobs = logprobs[len(tokenizer.sot_sequence_including_notimestamps) - 1 : -1]  # (T', V)
        logprobs = torch.gather(logprobs, dim=-1, index=class_name_tokens.view(-1, 1))  # (T', 1)
        average_logprob = logprobs.mean().item()
        average_logprobs[i] = average_logprob

    return average_logprobs


def classify(
    model: Whisper,
    audio_path: str,
    class_names: List[str],
    tokenizer: Tokenizer,
    internal_lm_average_logprobs: Optional[torch.Tensor],
    verbose: bool = False,
) -> str:
    audio_features = calculate_audio_features(audio_path, model)

    average_logprobs = calculate_average_logprobs(
        model=model,
        audio_features=audio_features,
        class_names=class_names,
        tokenizer=tokenizer,
    )
    if internal_lm_average_logprobs is not None:
        average_logprobs -= internal_lm_average_logprobs

    sorted_indices = sorted(
        range(len(class_names)), key=lambda i: average_logprobs[i], reverse=True
    )
    if verbose:
        tqdm.write("  Average log probabilities for each class:")
        for i in sorted_indices:
            tqdm.write(f"    {class_names[i]}: {average_logprobs[i]:.3f}")

    return class_names[sorted_indices[0]]


def calculate_internal_lm_average_logprobs(
    model: Whisper,
    class_names: List[str],
    tokenizer: Tokenizer,
    verbose: bool = False,
) -> torch.Tensor:
    audio_features_from_empty_input = calculate_audio_features(None, model)
    average_logprobs = calculate_average_logprobs(
        model=model,
        audio_features=audio_features_from_empty_input,
        class_names=class_names,
        tokenizer=tokenizer,
    )
    if verbose:
        print("Internal LM average log probabilities for each class:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {average_logprobs[i]:.3f}")
    return average_logprobs


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

    internal_lm_average_logprobs = None
    if args.subtract_internal_lm_score:
        internal_lm_average_logprobs = calculate_internal_lm_average_logprobs(
            model=model,
            class_names=class_names,
            tokenizer=tokenizer,
            verbose=args.verbose,
        )

    results = []
    for record in tqdm(records):
        tqdm.write(f"processing {record.audio_path} (class: {record.category})")
        result = classify(
            model=model,
            audio_path=record.audio_path,
            class_names=class_names,
            tokenizer=tokenizer,
            internal_lm_average_logprobs=internal_lm_average_logprobs,
            verbose=args.verbose,
        )
        results.append(result)
        tqdm.write(f"  predicted: {result}")

    write_result(records, results, args.output)
    accuracy = calc_accuracy(records, results)
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
