# eval_script.py
import torch
from joblib import load
import argparse
from script import (
    EncoderRNN,
    AttnDecoderRNN,
    tensorFromSentence,
    evaluate,
    Lang,  # Needed for depickling process
    normalizeString,
)
from pathlib import Path
import sys
import traceback


DATA_DIR = Path(__file__).resolve().parent / "data"

DATA_DIR.mkdir(exist_ok=True)

# Set this to the device you are using, 'cpu' or 'cuda'
device = torch.device("cpu")


# Function to load the model
def load_model(file_name):
    path = DATA_DIR / file_name

    # Check if the file exists
    if not path.is_file():
        print(f"The model file {path} was not found.")
        sys.exit(1)

    model_data = load(path)
    hyperparameters = model_data["hyperparameters"]

    encoder = EncoderRNN(
        hyperparameters["input_size"],
        hyperparameters["hidden_size"],
        hyperparameters["num_gru_layers"],
        hyperparameters["dropout_p"],
    ).to(device)

    decoder = AttnDecoderRNN(
        hyperparameters["hidden_size"],
        hyperparameters["output_size"],
        hyperparameters["dropout_p"],
    ).to(device)

    encoder.load_state_dict(model_data["encoder_state_dict"])
    decoder.load_state_dict(model_data["decoder_state_dict"])
    input_lang = model_data["input_lang"]
    output_lang = model_data["output_lang"]

    encoder.eval()
    decoder.eval()

    return encoder, decoder, input_lang, output_lang


def translate(encoder, decoder, input_lang, output_lang, sentence):
    for word in sentence.split():
        if word not in input_lang.word2index:
            print(f"Warning: '{word}' not in vocabulary.")

    with torch.no_grad():
        output_words, attentions = evaluate(
            encoder, decoder, sentence, input_lang, output_lang
        )
        return " ".join(output_words)


# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate a given sentence.")
    parser.add_argument("sentence", type=str, help="The sentence to translate.")
    args = parser.parse_args()

    # Preprocess the sentence using the same normalization used during training
    normalized_sentence = normalizeString(args.sentence)

    # Load the model and language objects
    try:
        encoder, decoder, input_lang, output_lang = load_model(
            "model_checkpoint.joblib"
        )
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)

    # Translate the sentence
    try:
        translation = translate(
            encoder,
            decoder,
            input_lang,
            output_lang,
            normalized_sentence,
        )
        print(f"Translated sentence: {translation}")
    except Exception as e:
        print(f"Error during translation: {e}")
        traceback.print_exc()
