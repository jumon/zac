# ZAC
ZAC (Zero-shot Audio Classification using Whisper) allows you to assign audio files to ANY class you want without training.

Demo: https://huggingface.co/spaces/Jumon/whisper-zero-shot-audio-classification

## Quick Start
1. Install [Whisper](https://github.com/openai/whisper). Follow the instructions [here](https://github.com/openai/whisper#setup) to install Whisper.

2. Create a `class_names.txt` file containing the class names of the dataset you want to classify.
It is recommended to enclose the class names with square brackets (`[` and `]`) to get better results.
An example for the [ESC-50](https://github.com/karolpiczak/ESC-50) dataset can be found in [`examples/ESC-50/class_names.txt`](https://github.com/jumon/zac/blob/main/examples/ESC-50/class_names.txt).

3. Run the following command:
    ```
    python classify.py --audio <audio_file_path> --class_names <class_names_file_path>
    ```
    For all available options, see `classify.py`.

## How does it work?
Whisper is an automatic speech recognition (ASR) model trained on a massive amount of labeled audio data collected from the internet.
It is assumed that the training data includes closed-caption-like subtitles describing non-speech sounds, such as "[dog barking]", "[car engine]", or "[water running]".
This allows Whisper to recognize non-speech sounds and classify them into predefined classes.
For each class, we can calculate the average log probability that Whisper generates the class name text given an audio sample.
The class with the highest average log probability will be the predicted class.

However, this approach makes it more difficult to recognize rare class names than common ones.
For instance, it generally assigns higher probabilities to "[laughing]" or "[footsteps]" than to "[hen]" or "[hand_saw]".
We can improve this by subtracting "internal language model scores" for each class, which are the average log probabilities for class names given an empty input.

## Results
To evaluate our method, we used the ESC-50 dataset, which contains 2000 audio samples from 50 classes (40 samples per class).
We used the class names from the dataset as is, but added square brackets (`[` and `]`) around the names.
This achieved an accuracy of 31.80% (636/2000).

## Limitations
- The results are highly dependent on the dataset used to train Whisper, which is not publicly available. Class names that are not present in the training data will not be recognized well.
- The accuracy of the classification depends on the class names used. For example, using "[birds chirping]" instead of "[chirping_birds]" may produce better results, as the former is probably more common in Whisper's training data.
